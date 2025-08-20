import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.Modules import ConvBNGeLU, ConvBN, DepthwiseSeparableConv

class DWStack(nn.Module):
    """
    Depthwise conv stack used inside FM/BM. Implements focal levels with
    increasing receptive fields (k=7,9). Pointwise identity is left to the caller.
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
