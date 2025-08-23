import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.Modules import ConvBNGeLU, ConvBN, DepthwiseSeparableConv

class DWStack(nn.Module):
    """
    Depthwise conv stack used inside FM/BM. Implements focal levels with
    increasing receptive fields (k=7,9). Pointwise identity is left to the caller.

    Tuning Tips:
    levels=2 and base_k=7 match the paper’s increasing kernel schedule (7,9). Increase levels to 3 (7,9,11) if memory allows.

    If training is unstable, start by disabling frequency-path FSM (comment yhf…ylb) and keep only spatial FSM; re-enable later.

    To supervise the mask m (optional), expose it as an auxiliary output and add a small-weighted BCE loss against the GT saliency.

    """
    def __init__(self, channels: int, levels: int = 2, base_k: int = 7):
        super().__init__()
        ks = [base_k + 2*i for i in range(levels)]
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, k, padding=k//2, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for k in ks
        ])
        # include identity at l=0 (as in G^{L+1} in paper)
        self.include_identity = True

    def forward(self, x):
        outs = []
        if self.include_identity:
            outs.append(x)
        cur = x
        for b in self.blocks:
            cur = b(cur)
            outs.append(cur)
        # outs: list length L+1, each [B,C,H,W]
        return outs  # return all focal levels


class Gating(nn.Module):
    """
    Produces per-level gates G^l \in R^{H×W×(L+1)} using a 1×1 conv.
    """
    def __init__(self, channels: int, levels_plus_one: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, levels_plus_one, kernel_size=1, bias=True)

    def forward(self, x, H, W):
        g = torch.sigmoid(self.conv(x))               # [B,L+1,H,W]
        return g                                      # channel-less gates


class FSMBlock(nn.Module):
    """
    One FSM that takes split features (foreground or background) and produces
    gated focal aggregation result.
    """
    def __init__(self, channels: int, levels: int = 2, base_k: int = 7):
        super().__init__()
        self.levels = levels
        self.dwstack = DWStack(channels, levels=levels, base_k=base_k)
        self.gating = Gating(channels, levels_plus_one=levels+1)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.out_bn = nn.BatchNorm2d(channels)

    def forward(self, x, context_feat):
        # x: [B,C,H,W] features to modulate; context_feat used to compute gates
        B, C, H, W = x.shape
        outs = self.dwstack(x)                        # list of L+1 maps
        G = self.gating(context_feat, H, W)           # [B,L+1,H,W]
        # Weighted sum over levels (broadcast gates to channel dim)
        agg = 0.0
        for l, z in enumerate(outs):
            gl = G[:, l:l+1, :, :]                   # [B,1,H,W]
            agg = agg + z * gl
        y = self.out_proj(agg)
        y = self.out_bn(y)
        return y


class FSM_FFM(nn.Module):
    """
    CamoFocus-style Feature Split and Modulation fused into your FFM role.
    Signature and output shape match your FFM:
        forward(x, high, low) -> x_out with same [B, C, H, W] as x
    """
    def __init__(self, channel: int, levels: int = 2, base_k: int = 7):
        super().__init__()

        # 1) align frequency channels (same as your FFM)
        self.high_reconv = ConvBNGeLU(in_channels=96, out_channels=channel, kernel_size=1)
        self.low_reconv  = ConvBNGeLU(in_channels=96, out_channels=channel, kernel_size=1)

        # 2) small fusers (keep your concat->1×1 behavior)
        self.high_fuse = ConvBN(in_channels=channel*2, out_channels=channel, kernel_size=1)
        self.low_fuse  = ConvBN(in_channels=channel*2, out_channels=channel, kernel_size=1)

        # 3) mask generator m \in [0,1] from spatial+freq context
        self.mask_gen = nn.Sequential(
            nn.Conv2d(channel*3, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 4) Foreground/Background modulators (FM/BM) for spatial path
        self.fm_spatial = FSMBlock(channel, levels=levels, base_k=base_k)
        self.bm_spatial = FSMBlock(channel, levels=levels, base_k=base_k)

        # 5) Also modulate high/low frequency branches
        self.fm_high = FSMBlock(channel, levels=levels, base_k=base_k)
        self.bm_high = FSMBlock(channel, levels=levels, base_k=base_k)
        self.fm_low  = FSMBlock(channel, levels=levels, base_k=base_k)
        self.bm_low  = FSMBlock(channel, levels=levels, base_k=base_k)

        # 6) final mix (keeps channels=channel)
        self.act = nn.GELU()
        self.out_conv = ConvBN(in_channels=channel, out_channels=channel, kernel_size=1)

    def forward(self, x, high, low):
        # resize frequency maps to match x
        Hx, Wx = x.shape[2], x.shape[3]
        high = F.interpolate(high, size=(Hx, Wx), mode='bilinear', align_corners=False)
        low  = F.interpolate(low,  size=(Hx, Wx), mode='bilinear', align_corners=False)

        # project frequency channels
        high = self.high_reconv(high)                 # [B,C,H,W]
        low  = self.low_reconv(low)                   # [B,C,H,W]

        # quick fusions with spatial x (as in your FFM)
        high_x = self.high_fuse(torch.cat([high, x], dim=1))   # [B,C,H,W]
        low_x  = self.low_fuse(torch.cat([low,  x], dim=1))    # [B,C,H,W]

        # build mask m from spatial + freq context
        mask_ctx = torch.cat([x, high, low], dim=1)            # [B,3C,H,W]
        m = self.mask_gen(mask_ctx)                            # [B,1,H,W]

        # split features
        xf = x * m
        xb = x * (1.0 - m)

        hf = high_x * m
        hb = high_x * (1.0 - m)

        lf = low_x * m
        lb = low_x * (1.0 - m)

        # focal modulation (use x as context for gating on spatial path,
        # and high_x/low_x respectively for frequency paths)
        ysf = self.fm_spatial(xf, context_feat=x)              # spatial foreground
        ysb = self.bm_spatial(xb, context_feat=x)              # spatial background

        yhf = self.fm_high(hf, context_feat=high_x)
        yhb = self.bm_high(hb, context_feat=high_x)

        ylf = self.fm_low(lf, context_feat=low_x)
        ylb = self.bm_low(lb, context_feat=low_x)

        # aggregate: spatial FM/BM + freq FM/BM, mirroring your sum of high/low
        spatial_out = ysf + ysb
        freq_out    = (yhf + yhb + ylf + ylb) / 2.0            # keep scale similar

        out = self.act(spatial_out + freq_out)
        out = self.out_conv(out)                                # ConvBN 1×1, preserves channel count

        return out

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DSConvBNReLU(nn.Module):
    """Depthwise separable 3x3 conv -> BN -> ReLU, with configurable dilation."""
    def __init__(self, channels, dilation: int = 1):
        super().__init__()
        padding = dilation
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, stride=1,
                            padding=padding, dilation=dilation, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class CRM(nn.Module):
    """
    Context Refinement Module (paper-style, Fig.4)
    Inputs:
      - x_cur: tensor (B, C, H, W), current-stage modulated feature x_n'
      - x_coarse_up: tensor (B, C, H, W) or None, upsampled previous stage feature x_{n+1}'↑
    Steps:
      1) If a coarse feature is provided, spatially match (should already match) and concatenate [x_cur, x_coarse_up] on channels.
      2) 1x1 fuse to C channels.
      3) Split into 4 equal chunks along channels.
      4) Apply 4 parallel 3x3 DS-Convs with dilations 1, 2, 4, 8 respectively, each on its own chunk.
      5) Elementwise sum the 4 outputs -> f_3.
      6) Final 1x1 conv to mix channels -> refined feature (B, C, H, W).
      7) Optional 1x1 prediction head to produce P_n (B, 1, H, W).
    """
    def __init__(self, channels: int, make_side_head: bool = True):
        super().__init__()
        self.channels = channels
        self.make_side_head = make_side_head

        # Fuse concatenated inputs back to C
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # Four parallel DS-Conv branches with different dilations
        quarter = channels // 4
        # If C not divisible by 4, distribute remainder to first chunks to preserve sum=channels
        splits = [quarter, quarter, quarter, channels - 3 * quarter]
        self.splits = splits  # used for channel split

        self.br1 = _DSConvBNReLU(splits[0], dilation=1)
        self.br2 = _DSConvBNReLU(splits[1], dilation=2)
        self.br3 = _DSConvBNReLU(splits[2], dilation=4)
        self.br4 = _DSConvBNReLU(splits[3], dilation=8)

        # Final 1x1 to mix after element-wise sum
        self.mix = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # Optional side head for deep supervision (P_n)
        if self.make_side_head:
            self.side_head = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        else:
            self.side_head = None

    def forward(self, x_cur: torch.Tensor, x_coarse_up: torch.Tensor | None = None):
        b, c, h, w = x_cur.shape
        assert c == self.channels, f"Expected channels={self.channels}, got {c}"

        if x_coarse_up is None:
            # If coarser feature not provided (topmost stage), use zeros like a skip-less second input
            x_coarse_up = torch.zeros_like(x_cur)
        else:
            # Ensure spatial match (should already match if caller upsamples)
            if x_coarse_up.shape[2:] != (h, w):
                x_coarse_up = F.interpolate(x_coarse_up, size=(h, w), mode='bilinear', align_corners=False)

        # 1x1 fusion after concat
        z = torch.cat([x_cur, x_coarse_up], dim=1)  # (B, 2C, H, W)
        z = self.fuse(z)                            # (B, C, H, W)

        # Channel split into 4 chunks
        s1, s2, s3, s4 = self.splits
        z1, z2, z3, z4 = torch.split(z, [s1, s2, s3, s4], dim=1)

        # Parallel DS-Conv with dilations 1,2,4,8
        y1 = self.br1(z1)
        y2 = self.br2(z2)
        y3 = self.br3(z3)
        y4 = self.br4(z4)

        # Element-wise sum (concat is in the paper figure for display, but the text specifies sum)
        # If you prefer strict “sum after concat-of-branches”, first concat then 1x1; here we sum directly to C:
        # Bring all to C via zero-pad-and-concat-like behavior: we’ll put them back in their original channel slots.
        # Simpler and faithful: concatenate branch outputs and then 1x1; but paper shows sum (⊕). We implement sum over re-concatenated tensor.
        y = torch.cat([y1, y2, y3, y4], dim=1)  # still (B, C, H, W) due to preserved per-branch channels
        # Final 1x1 mix
        y = self.mix(y)  # refined feature (B, C, H, W)

        # Side head (optional)
        p = self.side_head(y) if self.side_head is not None else None
        return y, p
