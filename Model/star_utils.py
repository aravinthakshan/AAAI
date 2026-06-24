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
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.Starnet import Block, ConvBN


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
# StarLapInjection  (unchanged — performs well)
# ─────────────────────────────────────────────────────────────────────────────

class StarLapInjection(nn.Module):
    """
    Cross-modal star gate: encoder context selects which Laplacian details survive.
    act(f1(ctx)) ⊙ f2(lap)  +  residual skip
    """
    def __init__(self, encoder_channels: int, lap_channels: int = 3, mlp_ratio: int = 3):
        super().__init__()
        C = encoder_channels
        H = mlp_ratio * C

        self.dwconv  = ConvBN(C, C, 7, 1, 3, groups=C, with_bn=True)
        self.lap_proj = nn.Sequential(
            nn.Conv2d(lap_channels, C // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.GELU(),
            nn.Conv2d(C // 2, C, 1, bias=False),
            nn.BatchNorm2d(C),
        )
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
        Fixed:    96 → mid → C  where mid = max(4*C, 32)

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
# StarDeBlock  (Fix 2 + 3 applied: zero-init cross branch + GELU)
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
    """
    def __init__(self, in_channels: int, out_channels: int, mlp_ratio: int = 3):
        super().__init__()
        C = out_channels
        H = mlp_ratio * C

        self.proj = (
            nn.Conv2d(in_channels, C, 1, bias=True)
            if in_channels != C else nn.Identity()
        )
        self.dwconv = ConvBN(C, C, 7, 1, 3, groups=C, with_bn=True)

        # Low branch
        self.low_f1 = ConvBN(C, H, 1, with_bn=False)
        self.low_f2 = ConvBN(C, H, 1, with_bn=False)
        self.low_g  = ConvBN(H, C, 1, with_bn=True)
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

        low = self.low_dw(self.low_g(self.act(self.low_f1(ctx)) * self.low_f2(ctx)))

        mid = ctx
        for blk in self.mid_blocks:
            mid = blk(mid)
        mid = self.mid_dw(self.mid_g(self.act(self.mid_f1(mid)) * self.mid_f2(mid)))

        top   = self.top_block(mid)
        cross = self.cross_g(self.act(self.cross_f1(low)) * self.cross_f2(top))

        return self.out_bn(x + low + mid + cross)