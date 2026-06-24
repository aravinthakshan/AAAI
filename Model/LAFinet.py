"""
LaFinet.py  — backbone-aware channel scaling + controlled ablation
----------------------------------------------------------------------
Key change: LaplacianFINet now accepts channels=None, which triggers
automatic scaling based on the backbone's stage channel widths.

For starnet_s1  [24, 48, 96, 192]  → auto channels = (24, 48, 96, 192)
For efficientb0 [stage-dependent]  → auto channels = (derived from backbone)

This prevents the aggressive bottleneck (192 → 48 → 8) that was the
primary cause of low weighted-Fβ with the larger backbone.

Ablation flags let you isolate which change is responsible for any
metric difference: backbone, star modules, or channel width.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.EfficientNet   import EfficientNet_B0
from Model.TinyNet        import TinyNetA
from Model.Starnet        import StarNetEncoder, Block
from Model.Demonet        import DemoNetEncoder
from Model.Modules        import ConvBNGeLU, ConvBN
from Model.lap_utils      import (
    LaplacianPyramid, LaplacianInjectionBlock,
    asf_attention_model, ScalSeq,
)
from Model.star_utils import (
    StarLapInjection, StarFFM, StarDeBlock, load_starnet_pretrained,
)

# Original modules kept for ablation
try:
    from Model.Replacements import FSM_FFM
except ImportError:
    FSM_FFM = None   # may not exist in all setups

try:
    from Model.lap_utils import DeBlock   # if kept in lap_utils
except ImportError:
    DeBlock = None


def build_lafinet_backbone(backbone: str, pretrained: bool = False):
    if backbone == 'efficientb0':
        return EfficientNet_B0()
    if backbone == 'tinynet-a':
        return TinyNetA()
    if backbone.startswith('starnet_'):
        enc = StarNetEncoder(variant=backbone)
        if pretrained:
            load_starnet_pretrained(enc, backbone)   # Fix 4
        return enc
    if backbone.startswith('demonet_'):
        parts = backbone.split('_')
        if len(parts) != 4 or not parts[1].startswith('d') or not parts[2].startswith('w'):
            raise ValueError(f"Invalid DemoNet backbone name: {backbone}")
        return DemoNetEncoder(depth=int(parts[1][1:]), dim=int(parts[2][1:]), mode=parts[3])
    raise ValueError(f"Unsupported backbone: {backbone}")


def _auto_channels(stage_channels: list, scale: float = 1.0):
    """
    Derive decoder channel widths directly from backbone stage widths.
    stage_channels = encoder.get_stage_channels() → [C0, C1, C2, C3, C4]
    We use stages 1–4 (indices 1..4) for the four decoder widths.
    scale < 1.0 reduces if you need fewer params.
    """
    return tuple(max(8, int(c * scale)) for c in stage_channels[1:])


class LaplacianFINet(nn.Module):
    """
    channels=None  → auto-scale decoder widths to backbone capacity (recommended)
    channels=tuple → explicit widths (original behaviour, useful for ablation)

    use_star=True  → StarLapInjection + StarFFM + StarDeBlock  (new)
    use_star=False → LaplacianInjectionBlock + FSM_FFM + DeBlock (original)

    pretrained=True → load ImageNet weights for StarNet backbones
    """

    def __init__(
        self,
        backbone:    str   = 'starnet_s1',
        channels:    tuple = None,
        use_star:    bool  = True,
        pretrained:  bool  = False,
        channel_scale: float = 1.0,
    ):
        super().__init__()

        self.encoder = build_lafinet_backbone(backbone, pretrained=pretrained)
        stage_ch = self.encoder.get_stage_channels()  # [C0, C1, C2, C3, C4]

        # ── Channel widths ────────────────────────────────────────────────────
        if channels is None:
            channels = _auto_channels(stage_ch, scale=channel_scale)
            print(f"[LaFINet] auto channels {channels}  (backbone stages {stage_ch})")
        self.channels = channels

        # ── Laplacian pyramid ─────────────────────────────────────────────────
        self.laplacian_pyramid = LaplacianPyramid(num_levels=3)

        # ── Injection site (star or original) ────────────────────────────────
        if use_star:
            self.lap_injection1 = StarLapInjection(stage_ch[1], 3)
            self.lap_injection2 = StarLapInjection(stage_ch[2], 3)
            self.lap_injection3 = StarLapInjection(stage_ch[3], 3)
        else:
            self.lap_injection1 = LaplacianInjectionBlock(stage_ch[1], 3, stage_ch[1])
            self.lap_injection2 = LaplacianInjectionBlock(stage_ch[2], 3, stage_ch[2])
            self.lap_injection3 = LaplacianInjectionBlock(stage_ch[3], 3, stage_ch[3])

        # ── Channel reduction ─────────────────────────────────────────────────
        self.re_conv1 = ConvBNGeLU(stage_ch[1], channels[0], kernel_size=1)
        self.re_conv2 = ConvBNGeLU(stage_ch[2], channels[1], kernel_size=1)
        self.re_conv3 = ConvBNGeLU(stage_ch[3], channels[2], kernel_size=1)
        self.re_conv4 = ConvBNGeLU(stage_ch[4], channels[3], kernel_size=1)

        # ── Frequency fusion (star or original) ───────────────────────────────
        if use_star:
            self.ffm1 = StarFFM(channels[0], freq_channels=96)
            self.ffm2 = StarFFM(channels[1], freq_channels=96)
            self.ffm3 = StarFFM(channels[2], freq_channels=96)
            self.ffm4 = StarFFM(channels[3], freq_channels=96)
        else:
            assert FSM_FFM is not None, "FSM_FFM not available"
            self.ffm1 = FSM_FFM(channels[0])
            self.ffm2 = FSM_FFM(channels[1])
            self.ffm3 = FSM_FFM(channels[2])
            self.ffm4 = FSM_FFM(channels[3])

        self.gelu = nn.GELU()

        # ── Decoder (star or original) ────────────────────────────────────────
        if use_star:
            self.deconv3 = StarDeBlock(channels[3], channels[2])
            self.deconv2 = StarDeBlock(channels[2], channels[1])
            self.deconv1 = StarDeBlock(channels[1], channels[0])
        else:
            assert DeBlock is not None, "DeBlock not available"
            self.deconv3 = DeBlock(channels[3], channels[2])
            self.deconv2 = DeBlock(channels[2], channels[1])
            self.deconv1 = DeBlock(channels[1], channels[0])

        # ── Output heads ──────────────────────────────────────────────────────
        self.out_conv1 = nn.Conv2d(channels[0], 1, 3, padding=1)
        self.out_conv2 = nn.Conv2d(channels[1], 1, 3, padding=1)
        self.out_conv3 = nn.Conv2d(channels[2], 1, 3, padding=1)
        self.out_conv4 = nn.Conv2d(channels[3], 1, 3, padding=1)

        # ── ASF + ScalSeq (unchanged) ─────────────────────────────────────────
        self.asf4      = asf_attention_model(channels[3])
        self.asf3      = asf_attention_model(channels[2])
        self.asf2      = asf_attention_model(channels[1])
        self.asf1      = asf_attention_model(channels[0])
        self.asf_proj3 = nn.Conv2d(channels[3], channels[2], 1)
        self.asf_proj2 = nn.Conv2d(channels[2], channels[1], 1)
        self.asf_proj1 = nn.Conv2d(channels[1], channels[0], 1)
        self.ssff      = ScalSeq([channels[0], channels[1], channels[2]], channels[3])

    def forward(self, x, high, low):
        lap_levels = self.laplacian_pyramid(x)
        x0, x1, x2, x3, x4 = self.encoder(x)

        x1 = self.lap_injection1(x1, lap_levels[0])
        x2 = self.lap_injection2(x2, lap_levels[1])
        x3 = self.lap_injection3(x3, lap_levels[2])

        x1 = self.re_conv1(x1)
        x2 = self.re_conv2(x2)
        x3 = self.re_conv3(x3)
        x4 = self.re_conv4(x4)

        x1   = self.ffm1(x=x1, high=high, low=low)
        x2   = self.ffm2(x=x2, high=high, low=low)
        x3   = self.ffm3(x=x3, high=high, low=low)
        out4 = self.ffm4(x=x4, high=high, low=low)

        out3 = self.gelu(
            self.deconv3(F.interpolate(out4, size=x3.shape[2:], mode='bilinear', align_corners=False)) + x3)
        out2 = self.gelu(
            self.deconv2(F.interpolate(out3, size=x2.shape[2:], mode='bilinear', align_corners=False)) + x2)
        out1 = self.gelu(
            self.deconv1(F.interpolate(out2, size=x1.shape[2:], mode='bilinear', align_corners=False)) + x1)

        fused = self.ssff([x1, x2, x3])
        fused = F.interpolate(fused, size=out4.shape[2:], mode='bilinear', align_corners=False)

        out4 = self.asf4([out4, fused])
        out3 = self.asf3([out3, self.asf_proj3(F.interpolate(out4, size=out3.shape[2:], mode='bilinear', align_corners=False))])
        out2 = self.asf2([out2, self.asf_proj2(F.interpolate(out3, size=out2.shape[2:], mode='bilinear', align_corners=False))])
        out1 = self.asf1([out1, self.asf_proj1(F.interpolate(out2, size=out1.shape[2:], mode='bilinear', align_corners=False))])

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