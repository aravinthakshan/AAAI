"""
LaFINet.py  (updated — star-integrated version)
------------------------------------------------
Changes vs original:
  1.  lap_injection1/2/3  →  StarLapInjection   (from StarLapModules)
  2.  ffm1/2/3/4          →  StarFFM             (from StarLapModules)
  3.  deconv1/2/3         →  StarDeBlock         (from StarLapModules)

  Removed imports / dead classes:
      - FSM_FFM              (was from Model.Replacements)
      - LaplacianInjectionBlock (was from Model.lap_utils)
      - Decoder, DeBlock, LapFusion  (all replaced by StarDeBlock)
      - LFA, HFA, FFM        (replaced by StarFFM's unified star paths)

Forward method is UNCHANGED — all interfaces are drop-in compatible.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.EfficientNet   import EfficientNet_B0
from Model.TinyNet        import TinyNetA
from Model.Starnet        import StarNetEncoder, Block
from Model.Demonet        import DemoNetEncoder
from Model.Modules        import ConvBNGeLU, ConvBN, DepthwiseSeparableConv
from Model.lap_utils      import (
    LaplacianPyramid,
    LDConv,
    asf_attention_model,
    ScalSeq,
    GOLDYOLO_Attention,
    top_Block,
)
from Model.star_utils import StarLapInjection, StarFFM, StarDeBlock


# ──────────────────────────────────────────────────────────────────────────────
# Backbone factory (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def build_lafinet_backbone(backbone: str):
    if backbone == 'efficientb0':
        return EfficientNet_B0()
    if backbone == 'tinynet-a':
        return TinyNetA()
    if backbone.startswith('starnet_'):
        return StarNetEncoder(variant=backbone)
    if backbone.startswith('demonet_'):
        parts = backbone.split('_')
        if len(parts) != 4 or not parts[1].startswith('d') or not parts[2].startswith('w'):
            raise ValueError(f"Invalid DemoNet backbone name: {backbone}")
        depth = int(parts[1][1:])
        dim   = int(parts[2][1:])
        mode  = parts[3]
        return DemoNetEncoder(depth=depth, dim=dim, mode=mode)
    raise ValueError(f"Unsupported LaFINet backbone: {backbone}")


# ──────────────────────────────────────────────────────────────────────────────
# LaplacianFINet  (star-integrated)
# ──────────────────────────────────────────────────────────────────────────────

class LaplacianFINet(nn.Module):
    """
    LaplacianFINet with StarNet element-wise multiplication integrated
    at all three major fusion sites.

    Star-operation summary
    ──────────────────────
    Site 1 – Laplacian injection (StarLapInjection × 3):
        act(f1(encoder_ctx)) ⊙ f2(lap_level)
        "amplify Laplacian edge detail only where semantic context activates"

    Site 2 – Frequency fusion (StarFFM × 4):
        h_feat = act(f1_h(high)) ⊙ f2_h(enc)    high-freq gate
        l_feat = act(f1_l(low))  ⊙ f2_l(enc)    low-freq  gate
        cross  = act(f1_x(h))    ⊙ f2_x(l)      both-must-agree gate
        "detect camouflage where freq anomaly AND semantic content co-occur"

    Site 3 – Decoder (StarDeBlock × 3):
        cross  = act(f1(low_branch)) ⊙ f2(top_branch)
        "flag a boundary only when it's consistent across coarse AND fine scale"
    """

    def __init__(self, backbone: str = 'efficientb0', channels: tuple = (8, 12, 24, 48)):
        super().__init__()

        self.encoder = build_lafinet_backbone(backbone)

        # 3-level Laplacian pyramid (unchanged)
        self.laplacian_pyramid = LaplacianPyramid(num_levels=3)

        stage_ch = self.encoder.get_stage_channels()  # [C0, C1, C2, C3, C4]

        # ── Site 1: Star-based Laplacian injection ────────────────────────────
        # StarLapInjection replaces LaplacianInjectionBlock.
        # Interface: forward(encoder_features, laplacian_level) → fused_features
        # Stages 1-3 receive lap levels L0, L1, L2. Stage 4 unchanged.
        self.lap_injection1 = StarLapInjection(stage_ch[1], lap_channels=3, mlp_ratio=3)
        self.lap_injection2 = StarLapInjection(stage_ch[2], lap_channels=3, mlp_ratio=3)
        self.lap_injection3 = StarLapInjection(stage_ch[3], lap_channels=3, mlp_ratio=3)

        # Channel reduction (unchanged)
        self.re_conv1 = ConvBNGeLU(stage_ch[1], channels[0], kernel_size=1)
        self.re_conv2 = ConvBNGeLU(stage_ch[2], channels[1], kernel_size=1)
        self.re_conv3 = ConvBNGeLU(stage_ch[3], channels[2], kernel_size=1)
        self.re_conv4 = ConvBNGeLU(stage_ch[4], channels[3], kernel_size=1)

        # ── Site 2: Star-based frequency fusion ──────────────────────────────
        # StarFFM replaces FSM_FFM.
        # Interface: forward(x, high, low) → freq-fused features
        # freq_channels=96 matches the external high/low frequency tensor width.
        self.ffm1 = StarFFM(channels[0], freq_channels=96, mlp_ratio=3)
        self.ffm2 = StarFFM(channels[1], freq_channels=96, mlp_ratio=3)
        self.ffm3 = StarFFM(channels[2], freq_channels=96, mlp_ratio=3)
        self.ffm4 = StarFFM(channels[3], freq_channels=96, mlp_ratio=3)

        self.gelu = nn.GELU()

        # ── Site 3: Star-based decoder ────────────────────────────────────────
        # StarDeBlock replaces DeBlock (which contained Decoder + LapFusion).
        # Interface: forward(x) → decoded features  (identical to DeBlock)
        self.deconv3 = StarDeBlock(channels[3], channels[2], mlp_ratio=3)
        self.deconv2 = StarDeBlock(channels[2], channels[1], mlp_ratio=3)
        self.deconv1 = StarDeBlock(channels[1], channels[0], mlp_ratio=3)

        # Output convolutions (unchanged)
        self.out_conv1 = nn.Conv2d(channels[0], 1, kernel_size=3, padding=1)
        self.out_conv2 = nn.Conv2d(channels[1], 1, kernel_size=3, padding=1)
        self.out_conv3 = nn.Conv2d(channels[2], 1, kernel_size=3, padding=1)
        self.out_conv4 = nn.Conv2d(channels[3], 1, kernel_size=3, padding=1)

        # # ASF attention + ScalSeq (unchanged)
        self.asf4      = asf_attention_model(channels[3])
        self.asf3      = asf_attention_model(channels[2])
        self.asf2      = asf_attention_model(channels[1])
        self.asf1      = asf_attention_model(channels[0])
        self.asf_proj3 = nn.Conv2d(channels[3], channels[2], kernel_size=1)
        self.asf_proj2 = nn.Conv2d(channels[2], channels[1], kernel_size=1)
        self.asf_proj1 = nn.Conv2d(channels[1], channels[0], kernel_size=1)
        self.ssff      = ScalSeq([channels[0], channels[1], channels[2]], channels[3])

    # ──────────────────────────────────────────────────────────────────────────
    # Forward  (logic unchanged — only module calls differ)
    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x:    torch.Tensor,
        high: torch.Tensor,
        low:  torch.Tensor,
    ):
        # Laplacian pyramid from input (3 levels)
        lap_levels = self.laplacian_pyramid(x)   # [L0, L1, L2]

        # Backbone encoder
        x0, x1, x2, x3, x4 = self.encoder(x)

        # ── Site 1: Inject Laplacian levels via star gates ────────────────────
        x1 = self.lap_injection1(x1, lap_levels[0])   # L0 finest → stage 1
        x2 = self.lap_injection2(x2, lap_levels[1])   # L1        → stage 2
        x3 = self.lap_injection3(x3, lap_levels[2])   # L2 coarser → stage 3
        # x4 unchanged (no Laplacian injection at deepest stage)

        # Channel reduction
        x1 = self.re_conv1(x1)
        x2 = self.re_conv2(x2)
        x3 = self.re_conv3(x3)
        x4 = self.re_conv4(x4)

        # ── Site 2: Frequency star fusion at all 4 stages ─────────────────────
        x1   = self.ffm1(x=x1, high=high, low=low)
        x2   = self.ffm2(x=x2, high=high, low=low)
        x3   = self.ffm3(x=x3, high=high, low=low)
        out4 = self.ffm4(x=x4, high=high, low=low)

        # ── Site 3: Star decoder (top-down) ───────────────────────────────────
        out3 = self.gelu(
            self.deconv3(
                F.interpolate(out4, size=x3.shape[2:], mode='bilinear', align_corners=False)
            ) + x3
        )
        out2 = self.gelu(
            self.deconv2(
                F.interpolate(out3, size=x2.shape[2:], mode='bilinear', align_corners=False)
            ) + x2
        )
        out1 = self.gelu(
            self.deconv1(
                F.interpolate(out2, size=x1.shape[2:], mode='bilinear', align_corners=False)
            ) + x1
        )

        # ScalSeq multi-scale aggregation + ASF attention (unchanged)
        fused = self.ssff([x1, x2, x3])
        fused = F.interpolate(fused, size=out4.shape[2:], mode='bilinear', align_corners=False)

        out4 = self.asf4([out4, fused])
        out3 = self.asf3([
            out3,
            self.asf_proj3(F.interpolate(out4, size=out3.shape[2:], mode='bilinear', align_corners=False))
        ])
        out2 = self.asf2([
            out2,
            self.asf_proj2(F.interpolate(out3, size=out2.shape[2:], mode='bilinear', align_corners=False))
        ])
        out1 = self.asf1([
            out1,
            self.asf_proj1(F.interpolate(out2, size=out1.shape[2:], mode='bilinear', align_corners=False))
        ])

        # Multi-scale outputs → upsample to 4× of out1 resolution
        out1 = self.out_conv1(out1)
        out2 = self.out_conv2(out2)
        out3 = self.out_conv3(out3)
        out4 = self.out_conv4(out4)

        size = (out1.shape[2] * 4, out1.shape[3] * 4)
        out1 = F.interpolate(out1, size=size, mode='bilinear', align_corners=False)
        out2 = F.interpolate(out2, size=size, mode='bilinear', align_corners=False)
        out3 = F.interpolate(out3, size=size, mode='bilinear', align_corners=False)
        out4 = F.interpolate(out4, size=size, mode='bilinear', align_corners=False)

        return out1, out2, out3, out4
