import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.EfficientNet import EfficientNet_B0
from Model.TinyNet import TinyNetA
from Model.Starnet import StarNetEncoder
from Model.Demonet import DemoNetEncoder
from Model.Modules import ConvBNGeLU, ConvBN, DepthwiseSeparableConv
from Model.lap_utils import LaplacianPyramid, LaplacianInjectionBlock, LDConv, asf_attention_model, ScalSeq, GOLDYOLO_Attention, top_Block
from Model.Replacements import FSM_FFM
from Model.Starnet import Block

def build_lafinet_backbone(backbone):
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
        dim = int(parts[2][1:])
        mode = parts[3]
        return DemoNetEncoder(depth=depth, dim=dim, mode=mode)
    raise ValueError(f"Unsupported LaFINet backbone: {backbone}")

class LapFusion(nn.Module):
    """
    Learnable 3-branch Laplacian fusion with channel + spatial attention.
    Replaces naive low + mid + top sum.
    """
    def __init__(self, channels):
        super().__init__()

        # lightweight attention: produces 3 attention maps (per pixel)
        self.att = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, 3, kernel_size=1, bias=False),  # 3 branch weights
            nn.Softmax(dim=1)  # normalize across branches
        )

    def forward(self, low, mid, top):
        # concat spatially
        x = torch.cat([low, mid, top], dim=1)

        # attention maps shape: [B, 3, H, W]
        w = self.att(x)

        # split weights
        w1, w2, w3 = w[:, 0:1], w[:, 1:2], w[:, 2:3]

        # weighted fusion
        out = w1 * low + w2 * mid + w3 * top
        return out


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        self.input_proj = nn.Identity()
        if in_channels != out_channels:
            self.input_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Low branch:
        self.low_branch = nn.Sequential(
            Block(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        # Middle branch:
        self.middle_branch = nn.ModuleList([
            Block(out_channels)
            for _ in range(4)
        ])

        # Top branch:
        self.top_branch = nn.ModuleList([
            Block(out_channels)
            for _ in range(2)
        ])

        # Final lap outputs
        self.final_mid = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.final_top = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.fusion = LapFusion(out_channels)

    def forward(self, x):
        x = self.input_proj(x)

        # Low
        low = self.low_branch(x)

        # Middle residual stack
        mid = x
        for block in self.middle_branch:
            mid = block(mid)
        mid_out = self.final_mid(mid)

        # Top residual stack
        top = mid
        for block in self.top_branch:
            top = block(top)
        top_out = self.final_top(top)

        out = self.fusion(low, mid_out, top_out)

        return out


class DeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=True)
        self.block1 = Block(out_channels)
        self.block2 = Block(out_channels)
        self.block3 = Block(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lap_decoder = Decoder(out_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.block1(x) + self.block2(x) + self.block3(x)
        x = self.bn(x)
        # Apply LAP decoder
        x = self.lap_decoder(x)
        # Combine all branches (hierarchical upsampling)
        #combined = low_out + middle_out + top_out
        return x

class LFA(nn.Module):
    """Low Frequency Injection Module (unchanged from original)"""
    def __init__(self, channels):
        super(LFA, self).__init__()

        # local_att
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.GELU(),
            nn.Conv2d(channels // 2, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels)
        )

        # global_att
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.GELU(),
            nn.Conv2d(channels // 2, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

        self.conv = ConvBN(in_channels=channels, out_channels=channels, kernel_size=1)

    def forward(self, x):
        x = self.local_att(x) + self.global_att(x) * x
        x = self.conv(x)
        return x


class HFA(nn.Module):
    """High Frequency Injection Module (unchanged from original)"""
    def __init__(self, channels):
        super(HFA, self).__init__()

        # local_att
        self.local_att = nn.Sequential(
            DepthwiseSeparableConv(channels, channels, kernel_size=3, padding=1, bias=False, stride=2),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparableConv(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        # global_att
        self.global_att = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparableConv(1, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        self.conv = ConvBN(in_channels=channels, out_channels=channels, kernel_size=1)

    def forward(self, x):
        x = self.local_att(x) + self.global_att(torch.mean(x, dim=1, keepdim=True)) * x
        x = self.conv(x)
        return x

class FFM(nn.Module): # This is the FIM, it takes input from the encoder and high/low frequency features
    """
    Frequency Injection Module
    """

    def __init__(self, channel):
        super(FFM, self).__init__()

        self.high_reconv = ConvBNGeLU(in_channels=96, out_channels=channel, kernel_size=1)
        self.low_reconv = ConvBNGeLU(in_channels=96, out_channels=channel, kernel_size=1)

        self.high_reconv2 = ConvBN(in_channels=channel * 2, out_channels=channel, kernel_size=1)
        self.low_reconv2 = ConvBN(in_channels=channel * 2, out_channels=channel, kernel_size=1)

        self.high_msca = HFA(channels=channel)
        self.low_msca = LFA(channels=channel)

        self.gelu = nn.GELU()
        self.conv = ConvBN(in_channels=channel, out_channels=channel, kernel_size=1)

    def forward(self, x, high, low):
        high = F.interpolate(high, size=x.shape[2:], mode='bilinear', align_corners=False)
        low = F.interpolate(low, size=x.shape[2:], mode='bilinear', align_corners=False)
        high = self.high_reconv(high)
        low = self.low_reconv(low)

        high_x = self.high_reconv2(torch.cat((high, x), dim=1))
        low_x = self.low_reconv2(torch.cat((low, x), dim=1))

        high_x = self.high_msca(high_x)
        low_x = self.low_msca(low_x)

        x = self.gelu(high_x + low_x)
        x = self.conv(x)

        return x


class LaplacianFINet(nn.Module):
    """FINet with 3-layer Laplacian Pyramid Integration and No Decoder"""
    
    def __init__(self, backbone='efficientb0', channels=(8, 12, 24, 48)):
        super(LaplacianFINet, self).__init__()

        self.encoder = build_lafinet_backbone(backbone)

        # Laplacian Pyramid decomposition with only 3 levels
        self.laplacian_pyramid = LaplacianPyramid(num_levels=3)
        
        stage_channels = self.encoder.get_stage_channels()
        
        # Laplacian injection blocks for first 3 encoder stages only
        self.lap_injection1 = LaplacianInjectionBlock(stage_channels[1], 3, stage_channels[1])
        self.lap_injection2 = LaplacianInjectionBlock(stage_channels[2], 3, stage_channels[2])
        self.lap_injection3 = LaplacianInjectionBlock(stage_channels[3], 3, stage_channels[3])
        
        # Channel reduction - only for stages that will be processed
        self.re_conv1 = ConvBNGeLU(in_channels=stage_channels[1], out_channels=channels[0], kernel_size=1)
        self.re_conv2 = ConvBNGeLU(in_channels=stage_channels[2], out_channels=channels[1], kernel_size=1)
        self.re_conv3 = ConvBNGeLU(in_channels=stage_channels[3], out_channels=channels[2], kernel_size=1)
        self.re_conv4 = ConvBNGeLU(in_channels=stage_channels[4], out_channels=channels[3], kernel_size=1)
        
        # Enhanced frequency fusion modules - only for 3 stages + final stage
        self.ffm1 = FSM_FFM(channels[0])
        self.ffm2 = FSM_FFM(channels[1])
        self.ffm3 = FSM_FFM(channels[2])
        self.ffm4 = FSM_FFM(channels[3])
        
        # activation
        self.gelu = nn.GELU()

        self.deconv3 = DeBlock(channels[3], channels[2]) 
        self.deconv2 = DeBlock(channels[2], channels[1])
        self.deconv1 = DeBlock(channels[1], channels[0])
        # Output convolutions - simplified
        self.out_conv1 = nn.Conv2d(channels[0], 1, kernel_size=3, padding=1)
        self.out_conv2 = nn.Conv2d(channels[1], 1, kernel_size=3, padding=1)
        self.out_conv3 = nn.Conv2d(channels[2], 1, kernel_size=3, padding=1)
        self.out_conv4 = nn.Conv2d(channels[3], 1, kernel_size=3, padding=1)

        
        #self.rcca_out4 = RCCAModule(channels[3]) # might cause dimensions issues.
        #self.rcca_out3 = RCCAModule(channels[2]) # this will cause overhead for sure.

        self.asf4 = asf_attention_model(channels[3])
        self.asf3 = asf_attention_model(channels[2])
        self.asf2 = asf_attention_model(channels[1])
        self.asf1 = asf_attention_model(channels[0])
        self.asf_proj3 = nn.Conv2d(channels[3], channels[2], kernel_size=1)
        self.asf_proj2 = nn.Conv2d(channels[2], channels[1], kernel_size=1)
        self.asf_proj1 = nn.Conv2d(channels[1], channels[0], kernel_size=1)
        self.ssff = ScalSeq([channels[0], channels[1], channels[2]], channels[3])

        #self.gold4 = top_Block(channels[3])
        #self.gold3 = top_Block(channels[2])
        #self.gold2 = top_Block(channels[1])
        #self.gold1 = top_Block(channels[0])

    def forward(self, x, high, low):
        # Generate 3-level Laplacian pyramid from input
        laplacian_levels = self.laplacian_pyramid(x)
        
        # Forward through encoder
        x0, x1, x2, x3, x4 = self.encoder(x)
        
        # Inject Laplacian levels at corresponding scales (only first 3 levels)
        x1 = self.lap_injection1(x1, laplacian_levels[0])  # L0 -> stage 1
        x2 = self.lap_injection2(x2, laplacian_levels[1])  # L1 -> stage 2
        x3 = self.lap_injection3(x3, laplacian_levels[2])  # L2 -> stage 3
        # x4 remains unchanged (no Laplacian injection)
        
        # Channel reduction
        x1 = self.re_conv1(x1)
        x2 = self.re_conv2(x2)
        x3 = self.re_conv3(x3)
        x4 = self.re_conv4(x4)
        
        # Enhanced frequency fusion
        x1 = self.ffm1(x=x1, high=high, low=low)
        x2 = self.ffm2(x=x2, high=high, low=low)
        x3 = self.ffm3(x=x3, high=high, low=low)
        out4 = self.ffm4(x=x4, high=high, low=low)

        #x3 = self.rcca_out3(x3) # added rcca here
        # RCCA Block, this could add lots of overhead remove later
        #out4 = self.rcca_out4(out4) # assuming num_classes=1 for binary segmentation

        #x1 = self.gold1(x1)
        #x2 = self.gold2(x2)
        #x3 = self.gold3(x3)
        #out4 = self.gold4(out4)

        out3 = self.gelu(
            self.deconv3(F.interpolate(out4, size=x3.shape[2:], mode='bilinear', align_corners=False)) + x3)
        out2 = self.gelu(
            self.deconv2(F.interpolate(out3, size=x2.shape[2:], mode='bilinear', align_corners=False)) + x2)
        out1 = self.gelu(
            self.deconv1(F.interpolate(out2, size=x1.shape[2:], mode='bilinear', align_corners=False)) + x1)

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


        # Generate outputs at multiple scales
        out1 = self.out_conv1(out1)
        out2 = self.out_conv2(out2)
        out3 = self.out_conv3(out3)
        out4 = self.out_conv4(out4)

        # Upsample all outputs to same resolution
        size = (out1.shape[2] * 4, out1.shape[3] * 4)
        out1 = F.interpolate(out1, size=size, mode='bilinear', align_corners=False)
        out2 = F.interpolate(out2, size=size, mode='bilinear', align_corners=False)
        out3 = F.interpolate(out3, size=size, mode='bilinear', align_corners=False)
        out4 = F.interpolate(out4, size=size, mode='bilinear', align_corners=False)

        return out1, out2, out3, out4
