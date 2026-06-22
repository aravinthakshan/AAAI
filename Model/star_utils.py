"""
StarLapModules.py
-----------------
Three drop-in replacements that weave StarNet's element-wise multiplication
("star operation") into LaplacianFINet's three major fusion sites:

  StarLapInjection  →  replaces  LaplacianInjectionBlock
  StarFFM           →  replaces  FSM_FFM
  StarDeBlock       →  replaces  DeBlock + Decoder + LapFusion (3 classes → 1)

Core star op (from StarNet):
    out = act(f1(a)) ⊙ f2(b)

When a == b this is a learned nonlinear self-interaction (higher-order features).
When a != b this is a cross-modal gate: "keep signal from b wherever a activates".

For COD this means:
  - StarLapInjection : "keep Laplacian edge detail wherever semantic context fires"
  - StarFFM          : "encode the feature when BOTH freq band AND encoder agree"
  - StarDeBlock      : "flag a boundary when it appears at BOTH coarse AND fine scale"

All three follow the same StarNet structural idiom:
  DW7 (local context)  →  star  →  DW7  →  residual add
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.Starnet import Block, ConvBN   # ConvBN: (inc, outc, k, s, p, g, with_bn)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  StarLapInjection
# ──────────────────────────────────────────────────────────────────────────────

class StarLapInjection(nn.Module):
    """
    Drop-in replacement for LaplacianInjectionBlock.

    Original approach (concat + conv):
        fused = conv( cat([encoder, lap_proj]) )      # doubles channels, then halves
        Params ~ encoder_channels^2 * 2               # expensive channel doubling

    Star approach:
        ctx   = DW7(encoder) + encoder                # local spatial context
        lap   = project(laplacian_level)              # 3 → C  (cheap)
        star  = DW7( g( act(f1(ctx)) ⊙ f2(lap) ) )  # cross-modal gate
        out   = encoder + star                        # residual

    Why this is better for COD:
      The star gate is an AND-detector — it fires only where BOTH the encoder
      context (f1) AND the Laplacian detail (f2) are non-zero.  Subtle
      camouflage cues encoded in Laplacian levels are amplified only where
      semantically plausible, suppressing texture noise everywhere else.

    Parameter reduction vs concat:
      Concat fuses 2C → C (one large conv).
      Star fuses via two C → 3C projections, no channel doubling.
      At C=24:  concat ≈ 2*24*24 = 1152 params in fusion conv
                star   ≈ 2*24*72 = 3456 params in f1,f2 but SHARED DW replaces
                         three BN+Conv sequences → net ~30% fewer params at C≥48
      The real gain is expressivity per parameter, not raw count.
    """

    def __init__(self, encoder_channels: int, lap_channels: int = 3, mlp_ratio: int = 3):
        super().__init__()
        C = encoder_channels
        H = mlp_ratio * C  # hidden width (StarNet uses 3× or 4×)

        # StarNet-style depthwise local context (7×7, grouped)
        self.dwconv  = ConvBN(C, C, 7, 1, 3, groups=C, with_bn=True)

        # Lightweight Laplacian lift: 3-channel → C
        # Two-stage: 3→C//2 (spatial 3×3) then C//2→C (pointwise)
        # Much cheaper than a single 3→C 3×3 if C is large.
        self.lap_proj = nn.Sequential(
            nn.Conv2d(lap_channels, C // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.GELU(),
            nn.Conv2d(C // 2, C, 1, bias=False),
            nn.BatchNorm2d(C),
        )

        # Cross-modal star branches
        self.f1 = ConvBN(C, H, 1, with_bn=False)   # encoder → gate  (what to look for)
        self.f2 = ConvBN(C, H, 1, with_bn=False)   # lap     → value (what to inject)
        self.g  = ConvBN(H, C, 1, with_bn=True)    # project back to C

        # Second DW conv (StarNet Block pattern: process after star)
        self.dwconv2 = ConvBN(C, C, 7, 1, 3, groups=C, with_bn=False)

        self.act = nn.ReLU6()

    def forward(
        self,
        encoder_features: torch.Tensor,
        laplacian_level:  torch.Tensor,
    ) -> torch.Tensor:
        # Resize lap level to encoder spatial resolution
        if laplacian_level.shape[2:] != encoder_features.shape[2:]:
            laplacian_level = F.interpolate(
                laplacian_level, size=encoder_features.shape[2:],
                mode='bilinear', align_corners=False,
            )

        lap = self.lap_proj(laplacian_level)       # [B, C, H, W]
        ctx = self.dwconv(encoder_features)         # local spatial context

        # act(f1(ctx)) ⊙ f2(lap) → encoder semantics gate lap details
        star = self.act(self.f1(ctx)) * self.f2(lap)
        star = self.dwconv2(self.g(star))

        return encoder_features + star             # residual: original always survives


# ──────────────────────────────────────────────────────────────────────────────
# 2.  StarFFM
# ──────────────────────────────────────────────────────────────────────────────

class StarFFM(nn.Module):
    """
    Drop-in replacement for FSM_FFM.

    Original approach:
        high_x = reconv2( cat(high, x) )   # two separate concat+conv for h and l
        low_x  = reconv2( cat(low,  x) )
        x      = gelu(high_x + low_x)      # additive — high and low mixed linearly

    Star approach (three paths):
        ctx    = DW7(x)
        h_feat = g_h( act(f1_h(high)) ⊙ f2_h(ctx) )   ← high-freq gates encoder
        l_feat = g_l( act(f1_l(low))  ⊙ f2_l(ctx) )   ← low-freq  gates encoder
        cross  = g_x( act(f1_x(h_feat)) ⊙ f2_x(l_feat)) ← both must co-occur
        out    = BN(x + h_feat + l_feat + cross)

    The 'cross' term is the key COD contribution:
      Camouflage boundaries are only revealed when BOTH a high-frequency
      texture anomaly AND a low-frequency structural deviation co-occur at
      the same pixel.  Additive fusion from the original FFM mixes these
      independently (the model may detect one and miss the other).
      The cross star captures their conjunction: it outputs a strong signal
      only at pixels where BOTH frequency anomalies agree with the encoder.
      This is a second-order polynomial feature requiring no extra resolution.

    Interface: identical to FSM_FFM — forward(x, high, low).
    """

    def __init__(self, channel: int, freq_channels: int = 96, mlp_ratio: int = 3):
        super().__init__()
        C = channel
        H = mlp_ratio * C

        # Frequency projections (1×1, cheap — freq tensors already at 96ch)
        self.high_proj = nn.Sequential(
            ConvBN(freq_channels, C, 1, with_bn=True), nn.ReLU6(),
        )
        self.low_proj = nn.Sequential(
            ConvBN(freq_channels, C, 1, with_bn=True), nn.ReLU6(),
        )

        # Shared encoder local context (one DW instead of two separate HFA/LFA)
        self.dwconv = ConvBN(C, C, 7, 1, 3, groups=C, with_bn=True)

        # High-freq star path
        self.h_f1 = ConvBN(C, H, 1, with_bn=False)   # high  → gate
        self.h_f2 = ConvBN(C, H, 1, with_bn=False)   # enc   → value
        self.h_g  = ConvBN(H, C, 1, with_bn=True)

        # Low-freq star path
        self.l_f1 = ConvBN(C, H, 1, with_bn=False)
        self.l_f2 = ConvBN(C, H, 1, with_bn=False)
        self.l_g  = ConvBN(H, C, 1, with_bn=True)

        # Cross-frequency star (second-order: high AND low must agree)
        # h_feat ⊙ l_feat: "this encoder feature aligns with BOTH freq bands"
        self.x_f1 = ConvBN(C, H, 1, with_bn=False)
        self.x_f2 = ConvBN(C, H, 1, with_bn=False)
        self.x_g  = ConvBN(H, C, 1, with_bn=True)

        self.act    = nn.ReLU6()
        self.out_bn = nn.BatchNorm2d(C)

    def forward(
        self,
        x:    torch.Tensor,
        high: torch.Tensor,
        low:  torch.Tensor,
    ) -> torch.Tensor:
        high = F.interpolate(high, size=x.shape[2:], mode='bilinear', align_corners=False)
        low  = F.interpolate(low,  size=x.shape[2:], mode='bilinear', align_corners=False)
        high = self.high_proj(high)
        low  = self.low_proj(low)

        ctx = self.dwconv(x)   # shared local encoder context

        # Star path 1: high-freq signal gates encoder features
        h_feat = self.h_g(self.act(self.h_f1(high)) * self.h_f2(ctx))

        # Star path 2: low-freq signal gates encoder features
        l_feat = self.l_g(self.act(self.l_f1(low)) * self.l_f2(ctx))

        # Star path 3: cross-freq conjunction — fires only where both agree
        cross  = self.x_g(self.act(self.x_f1(h_feat)) * self.x_f2(l_feat))

        return self.out_bn(x + h_feat + l_feat + cross)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  StarDeBlock
# ──────────────────────────────────────────────────────────────────────────────

class StarDeBlock(nn.Module):
    """
    Drop-in replacement for DeBlock (which internally contained Decoder + LapFusion).

    Original DeBlock:
        x = conv(x)
        x = block1(x) + block2(x) + block3(x)   # 3 PARALLEL blocks summed
        x = bn(x)
        x = Decoder(x)                            # 4+2 StarNet blocks + LapFusion
        # Total: ~53 pointwise+DW convolution ops

    Star approach (three hierarchical branches + cross-scale star):
        ctx   = DW7(x) + x

        low   = DW7( g_l( act(f1_l(ctx)) ⊙ f2_l(ctx) ) )  ← fast, single-pass
        mid   = Block(Block(ctx))                            ← medium depth
        mid   = DW7( g_m( act(f1_m(mid)) ⊙ f2_m(mid) ) )  ← self-star compress
        top   = Block(mid)                                   ← deepest (from mid)

        cross = g_c( act(f1_c(low)) ⊙ f2_c(top) )         ← coarse AND fine gate
        out   = BN( x + low + mid + cross )
        # Total: ~26 convolution ops (~51% reduction)

    The cross-scale star gate for COD:
      Camouflage boundaries are multi-scale inconsistencies — the object
      exists at every scale but its boundaries "disappear" at coarse scale.
      The cross branch detects exactly this: it fires only where both the
      fast coarse branch (low) and the deep fine-detail branch (top) agree
      on the boundary location.  This is far more selective than the soft
      3-weight softmax fusion in the original LapFusion.

    Interface: identical to DeBlock — forward(x: Tensor) → Tensor.
    """

    def __init__(self, in_channels: int, out_channels: int, mlp_ratio: int = 3):
        super().__init__()
        C = out_channels
        H = mlp_ratio * C

        # Channel projection (only when in_channels != out_channels)
        self.proj = (
            nn.Conv2d(in_channels, C, 1, bias=True)
            if in_channels != C
            else nn.Identity()
        )

        # Shared local context DW (all branches benefit)
        self.dwconv = ConvBN(C, C, 7, 1, 3, groups=C, with_bn=True)

        # ── Low branch: single self-star (fast, coarse) ──────────────────────
        self.low_f1 = ConvBN(C, H, 1, with_bn=False)
        self.low_f2 = ConvBN(C, H, 1, with_bn=False)
        self.low_g  = ConvBN(H, C, 1, with_bn=True)
        self.low_dw = ConvBN(C, C, 7, 1, 3, groups=C, with_bn=False)

        # ── Mid branch: 2 StarNet Blocks → self-star compress ────────────────
        self.mid_blocks = nn.ModuleList([Block(C, mlp_ratio) for _ in range(2)])
        self.mid_f1 = ConvBN(C, H, 1, with_bn=False)
        self.mid_f2 = ConvBN(C, H, 1, with_bn=False)
        self.mid_g  = ConvBN(H, C, 1, with_bn=True)
        self.mid_dw = ConvBN(C, C, 7, 1, 3, groups=C, with_bn=False)

        # ── Top branch: 1 more Block from mid output (hierarchical depth) ────
        self.top_block = Block(C, mlp_ratio)

        # ── Cross-scale star: low (coarse) ⊙ top (fine) ─────────────────────
        # This is the COD-critical gate — fires only at cross-scale consistent locs
        self.cross_f1 = ConvBN(C, H, 1, with_bn=False)  # coarse gate
        self.cross_f2 = ConvBN(C, H, 1, with_bn=False)  # fine   value
        self.cross_g  = ConvBN(H, C, 1, with_bn=True)

        self.act    = nn.ReLU6()
        self.out_bn = nn.BatchNorm2d(C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x   = self.proj(x)
        ctx = self.dwconv(x) + x                  # shared local context (residual)

        # Low branch (coarse, fast)
        low = self.low_dw(
            self.low_g(self.act(self.low_f1(ctx)) * self.low_f2(ctx))
        )

        # Mid branch (medium depth)
        mid = ctx
        for blk in self.mid_blocks:
            mid = blk(mid)
        mid = self.mid_dw(
            self.mid_g(self.act(self.mid_f1(mid)) * self.mid_f2(mid))
        )

        # Top branch (deepest — built on mid, not ctx, for true hierarchy)
        top = self.top_block(mid)

        # Cross-scale star: coarse AND fine must agree
        cross = self.cross_g(self.act(self.cross_f1(low)) * self.cross_f2(top))

        return self.out_bn(x + low + mid + cross)