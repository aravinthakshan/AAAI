"""
star_utils.py  — fixes for low weighted-Fβ with starnet_s1
------------------------------------------------------------------
Three targeted fixes applied:

Fix 1 — StarFFM: two-stage frequency projection instead of 96 → C direct.
         The 12× squeeze (96 → 8) in a single 1×1 destroys boundary detail
         before the star gate fires.  We now go 96 → mid → C where mid is
         clamped to be at least 4× C, preserving frequency information.

Fix 2 — StarDeBlock: zero-init the cross-scale branch output BN gamma.
         The cross gate (low ⊙ top) starts at near-zero output, so early
         training relies on the well-formed low + mid residual paths.
         As training progresses the cross branch activates gradually —
         the same trick used in ResNet "res-zero" / NF-Nets.

Fix 3 — StarDeBlock: use GELU instead of ReLU6 for the star activation.
         ReLU6's ceiling clips gradients when f1 output > 6 (common after BN
         on early layers).  GELU has no ceiling and passes larger gradients
         through the gate path, which matters most for the thin boundary
         features that drive weighted-Fβ.

Fix 4 — StarNetEncoder: pretrained weight loading with head/norm stripping.
         Without this, starnet_s1 starts from random init while EfficientNet-B0
         starts from ImageNet — the single biggest source of the metric gap.

------------------------------------------------------------------
LDSConv integration (new)
------------------------------------------------------------------
Two of the static 7×7 depthwise context convs are now replaced with
LDSConv (deformable sampling + LKP/SKA spatially-adaptive weighting,
see Model/lap_utils.py). Rationale:

  • StarLapInjection.dwconv → LDSConv, conditioned on the projected
    Laplacian level (`cond=lap`). The whole point of this block is to
    inject high-frequency boundary detail; a fixed square receptive
    field works against that when camouflaged-object edges are thin
    and rarely axis-aligned. Conditioning the offset prediction on the
    Laplacian signal lets sampling positions bend toward edge evidence
    that's already been computed, instead of inferring offsets from
    `ctx` alone.

  • StarDeBlock.low_dw → LDSConv (unconditioned). This sits in the
    *highest-resolution* decoder branch (the one that ultimately
    produces out1, the finest prediction before the final 4× upsample),
    where boundary precision matters most and channel width is
    smallest — i.e. where the extra compute is cheapest and most
    justified.

  Deliberately NOT applied to ffm4/out4 or deconv3 (lowest-resolution,
  highest-channel stage): those stages are about coarse localization,
  not fine boundary shape, so the LDSConv overhead isn't justified
  there relative to its cost.

  Caveat: LDSConv's offset_conv is zero-initialized (per the LDConv
  paper's convention), and StarDeBlock.cross_g already zero-inits its
  output BN gamma (Fix 2). Stacking two independently zero-initialized
  branches in the same forward pass can slow early convergence more
  than either fix alone — watch loss curves over the first few hundred
  iterations if you enable this in the decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.Starnet import Block, ConvBN
from Model.lap_utils import LDSConv


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load pretrained StarNet weights into an encoder
# ─────────────────────────────────────────────────────────────────────────────

model_urls = {
    "starnet_s1":   "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar",
    "starnet_s2":   "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.pth.tar",
    "starnet_s3":   "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.pth.tar",
    "starnet_s4":   "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar",
}


def load_starnet_pretrained(encoder, variant: str):
    """
    Download the classification checkpoint and copy stem+stages weights.
    The head (norm, avgpool, linear) is discarded — only the feature extractor
    layers that exist in StarNetEncoder are loaded.

    Usage:
        enc = StarNetEncoder(variant='starnet_s1')
        load_starnet_pretrained(enc, 'starnet_s1')
    """
    if variant not in model_urls:
        raise ValueError(f"No pretrained URL for {variant}. Available: {list(model_urls)}")

    url = model_urls[variant]
    checkpoint = torch.hub.load_state_dict_from_url(url, map_location="cpu")

    # Checkpoint may be nested under 'state_dict' or 'model'
    sd = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))

    # Strip keys that don't exist in StarNetEncoder (head, norm, avgpool)
    encoder_sd = encoder.state_dict()
    filtered = {
        k: v for k, v in sd.items()
        if k in encoder_sd and v.shape == encoder_sd[k].shape
    }
    missing = [k for k in encoder_sd if k not in filtered]
    unexpected = [k for k in sd if k not in encoder_sd]

    encoder.load_state_dict(filtered, strict=False)
    print(f"[StarNet pretrained] loaded {len(filtered)}/{len(encoder_sd)} keys  "
          f"| missing {len(missing)}  | unexpected {len(unexpected)}")
    return encoder


# ─────────────────────────────────────────────────────────────────────────────
# StarLapInjection  (ctx branch upgraded to Laplacian-conditioned LDSConv)
# ─────────────────────────────────────────────────────────────────────────────

class StarLapInjection(nn.Module):
    """
    Cross-modal star gate: encoder context selects which Laplacian details survive.
    act(f1(ctx)) ⊙ f2(lap)  +  residual skip

    `ctx` used to come from a fixed 7×7 depthwise ConvBN over the encoder
    features alone. It's now an LDSConv whose offset prediction also sees
    the projected Laplacian level (`lap`), so the sampling grid can deform
    toward the same high-frequency edges the block is trying to inject —
    instead of always pooling over a rigid square window.

    Set `use_ldsconv=False` to fall back to the original static depthwise
    conv (useful for ablation).
    """
    def __init__(self, encoder_channels: int, lap_channels: int = 3, mlp_ratio: int = 3,
                 use_ldsconv: bool = True, lds_num_param: int = 9, lds_groups: int = 8):
        super().__init__()
        C = encoder_channels
        H = mlp_ratio * C

        self.use_ldsconv = use_ldsconv
        self.lap_proj = nn.Sequential(
            nn.Conv2d(lap_channels, C // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.GELU(),
            nn.Conv2d(C // 2, C, 1, bias=False),
            nn.BatchNorm2d(C),
        )

        if use_ldsconv:
            # Offset prediction conditioned on the Laplacian projection (C channels)
            self.dwconv = LDSConv(C, num_param=lds_num_param, groups=lds_groups,
                                   lks=7, cond_dim=C)
        else:
            self.dwconv = ConvBN(C, C, 7, 1, 3, groups=C, with_bn=True)

        self.f1 = ConvBN(C, H, 1, with_bn=False)
        self.f2 = ConvBN(C, H, 1, with_bn=False)
        self.g  = ConvBN(H, C, 1, with_bn=True)
        self.dwconv2 = ConvBN(C, C, 7, 1, 3, groups=C, with_bn=False)
        self.act = nn.GELU()   # fix 3: GELU instead of ReLU6

    def forward(self, encoder_features: torch.Tensor, laplacian_level: torch.Tensor):
        if laplacian_level.shape[2:] != encoder_features.shape[2:]:
            laplacian_level = F.interpolate(
                laplacian_level, size=encoder_features.shape[2:],
                mode='bilinear', align_corners=False,
            )
        lap = self.lap_proj(laplacian_level)

        if self.use_ldsconv:
            ctx = self.dwconv(encoder_features, cond=lap)
        else:
            ctx = self.dwconv(encoder_features)

        star = self.act(self.f1(ctx)) * self.f2(lap)
        star = self.dwconv2(self.g(star))
        return encoder_features + star


# ─────────────────────────────────────────────────────────────────────────────
# StarFFM  (Fix 1 applied: two-stage frequency projection)
# ─────────────────────────────────────────────────────────────────────────────

class StarFFM(nn.Module):
    """
    Star-based Frequency Fusion Module.

    Fix 1 — two-stage frequency projection:
        Original: 96 → C  (single 1×1, up to 12× reduction when C=8)
        Fixed:    96 → mid → C  where mid is clamped to be at least 4× C.

    At C=8:  96 → 32 → 8  (two manageable steps, 3× then 4×)
    At C=48: 96 → 96 → 48 (first step is identity-like, no info loss)

    The two-step projection preserves boundary-relevant frequency content
    that the direct projection collapses.
    """
    def __init__(self, channel: int, freq_channels: int = 96, mlp_ratio: int = 3):
        super().__init__()
        C = channel
        H = mlp_ratio * C

        # Two-stage frequency lift (Fix 1)
        mid = max(4 * C, 32)   # intermediate width: at least 32, at least 4×C
        self.high_proj = nn.Sequential(
            ConvBN(freq_channels, mid, 1, with_bn=True),
            nn.GELU(),
            ConvBN(mid, C, 1, with_bn=True),
            nn.GELU(),
        )
        self.low_proj = nn.Sequential(
            ConvBN(freq_channels, mid, 1, with_bn=True),
            nn.GELU(),
            ConvBN(mid, C, 1, with_bn=True),
            nn.GELU(),
        )

        # Shared encoder DW context
        self.dwconv = ConvBN(C, C, 7, 1, 3, groups=C, with_bn=True)

        # Star paths
        self.h_f1 = ConvBN(C, H, 1, with_bn=False)
        self.h_f2 = ConvBN(C, H, 1, with_bn=False)
        self.h_g  = ConvBN(H, C, 1, with_bn=True)

        self.l_f1 = ConvBN(C, H, 1, with_bn=False)
        self.l_f2 = ConvBN(C, H, 1, with_bn=False)
        self.l_g  = ConvBN(H, C, 1, with_bn=True)

        # Cross-frequency conjunction
        self.x_f1 = ConvBN(C, H, 1, with_bn=False)
        self.x_f2 = ConvBN(C, H, 1, with_bn=False)
        self.x_g  = ConvBN(H, C, 1, with_bn=True)

        self.act    = nn.GELU()   # Fix 3
        self.out_bn = nn.BatchNorm2d(C)

    def forward(self, x: torch.Tensor, high: torch.Tensor, low: torch.Tensor):
        high = F.interpolate(high, size=x.shape[2:], mode='bilinear', align_corners=False)
        low  = F.interpolate(low,  size=x.shape[2:], mode='bilinear', align_corners=False)
        high = self.high_proj(high)
        low  = self.low_proj(low)

        ctx = self.dwconv(x)

        h_feat = self.h_g(self.act(self.h_f1(high)) * self.h_f2(ctx))
        l_feat = self.l_g(self.act(self.l_f1(low))  * self.l_f2(ctx))
        cross  = self.x_g(self.act(self.x_f1(h_feat)) * self.x_f2(l_feat))

        return self.out_bn(x + h_feat + l_feat + cross)


# ─────────────────────────────────────────────────────────────────────────────
# StarDeBlock  (Fix 2 + 3 applied: zero-init cross branch + GELU;
#               low branch upgraded to LDSConv)
# ─────────────────────────────────────────────────────────────────────────────

class StarDeBlock(nn.Module):
    """
    Star-based decoder block.

    Fix 2 — zero-init cross-branch output BN:
        At init, cross_g.bn.weight = 0 → cross output is zero.
        The model trains from the residual (x + low + mid) first,
        then gradually activates the cross-scale boundary gate.
        Same technique as "ReZero" / FixUp — stabilises early training
        when the multiplicative gate hasn't learned yet.

    Fix 3 — GELU throughout (no ReLU6 ceiling on gate signals).

    LDSConv on the low branch (new):
        `low_dw` used to be a static 7×7 depthwise conv applied right
        after the channel-mixing `low_g`. It's now an (unconditioned)
        LDSConv, so the spatial sampling for the finest-detail branch
        can deform toward thin/irregular boundary shapes instead of
        pooling over a fixed square window. Only applied here — not in
        deconv3, the lowest-res / highest-channel stage — to keep the
        added compute where boundary precision actually matters most.

        Set `use_ldsconv=False` to fall back to the original static
        depthwise conv (useful for ablation / compute-constrained runs).
    """
    def __init__(self, in_channels: int, out_channels: int, mlp_ratio: int = 3,
                 use_ldsconv: bool = True, lds_num_param: int = 9, lds_groups: int = 8):
        super().__init__()
        C = out_channels
        H = mlp_ratio * C

        self.use_ldsconv = use_ldsconv

        self.proj = (
            nn.Conv2d(in_channels, C, 1, bias=True)
            if in_channels != C else nn.Identity()
        )
        self.dwconv = ConvBN(C, C, 7, 1, 3, groups=C, with_bn=True)

        # Low branch
        self.low_f1 = ConvBN(C, H, 1, with_bn=False)
        self.low_f2 = ConvBN(C, H, 1, with_bn=False)
        self.low_g  = ConvBN(H, C, 1, with_bn=True)
        if use_ldsconv:
            self.low_dw = LDSConv(C, num_param=lds_num_param, groups=lds_groups, lks=7)
        else:
            self.low_dw = ConvBN(C, C, 7, 1, 3, groups=C, with_bn=False)

        # Mid branch
        self.mid_blocks = nn.ModuleList([Block(C, mlp_ratio) for _ in range(2)])
        self.mid_f1 = ConvBN(C, H, 1, with_bn=False)
        self.mid_f2 = ConvBN(C, H, 1, with_bn=False)
        self.mid_g  = ConvBN(H, C, 1, with_bn=True)
        self.mid_dw = ConvBN(C, C, 7, 1, 3, groups=C, with_bn=False)

        # Top branch
        self.top_block = Block(C, mlp_ratio)

        # Cross-scale star (Fix 2: output BN gamma zeroed at init)
        self.cross_f1 = ConvBN(C, H, 1, with_bn=False)
        self.cross_f2 = ConvBN(C, H, 1, with_bn=False)
        self.cross_g  = ConvBN(H, C, 1, with_bn=True)
        nn.init.zeros_(self.cross_g.bn.weight)   # <-- zero-init gamma

        self.act    = nn.GELU()   # Fix 3
        self.out_bn = nn.BatchNorm2d(C)

    def forward(self, x: torch.Tensor):
        x   = self.proj(x)
        ctx = self.dwconv(x) + x

        # Channel-mix (low_g) always runs first; low_dw is then either the
        # original static depthwise conv or the deformable LDSConv refine step.
        low = self.low_g(self.act(self.low_f1(ctx)) * self.low_f2(ctx))
        low = self.low_dw(low)

        mid = ctx
        for blk in self.mid_blocks:
            mid = blk(mid)
        mid = self.mid_dw(self.mid_g(self.act(self.mid_f1(mid)) * self.mid_f2(mid)))

        top   = self.top_block(mid)
        cross = self.cross_g(self.act(self.cross_f1(low)) * self.cross_f2(top))

        return self.out_bn(x + low + mid + cross)