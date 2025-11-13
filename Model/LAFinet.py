import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.EfficientNet import EfficientNet_B0
from Model.TinyNet import TinyNetA
from Model.Modules import ConvBNGeLU, ConvBN, DepthwiseSeparableConv
from Model.lap_utils import LaplacianPyramid, LaplacianInjectionBlock, LDConv, asf_attention_model, ScalSeq, GOLDYOLO_Attention
from Model.Replacements import FSM_FFM
from Model.ccnet import RCCAModule  

class Decoder(nn.Module):
    """Lap decoder with Low, Middle, and Top branches"""
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        
        # Low Branch - generates primary segmentation mask
        self.low_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Middle Branch - reconstructs high-resolution residuals
        self.middle_branch = nn.ModuleList([
            self._make_residual_block(in_channels) for _ in range(7)
        ])
        
        # Top Branch - final refinement
        self.top_branch = nn.ModuleList([
            self._make_residual_block(in_channels) for _ in range(2)
        ])
        
        # Final convolution for upsampling
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
    def _make_residual_block(self, channels):
        """Create a residual block with LeakyReLU between conv layers"""
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        """
        Forward pass through LAqua decoder
        Returns: (low_out, middle_out, top_out)
        """
        # Low Branch - primary segmentation
        low_out = self.low_branch(x)
        
        # Middle Branch - residual reconstruction
        middle_x = x
        for block in self.middle_branch:
            residual = block(middle_x)
            middle_x = middle_x + F.leaky_relu(residual, 0.2)
        
        # Top Branch - final refinement
        top_x = middle_x
        for block in self.top_branch:
            residual = block(top_x)
            top_x = top_x + F.leaky_relu(residual, 0.2)
        
        # Final outputs
        middle_out = self.final_conv(middle_x)
        top_out = self.final_conv(top_x)
        
        return low_out, middle_out, top_out

class DeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=True)
        self.conv1 = DepthwiseSeparableConv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1,
                                              bias=True)
        self.conv2 = DepthwiseSeparableConv(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 3),
                                              padding=(0, 1), bias=True)
        self.conv3 = DepthwiseSeparableConv(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1),
                                              padding=(1, 0), bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lap_decoder = Decoder(out_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv1(x) + self.conv2(x) + self.conv3(x)
        x = self.bn(x)
        # Apply LAP decoder
        low_out, middle_out, top_out = self.lap_decoder(x)
        # Combine all branches (hierarchical upsampling)
        combined = low_out + middle_out + top_out
        return combined

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

        if backbone == 'efficientb0':
            self.encoder = EfficientNet_B0()
        elif backbone == 'tinynet-a':
            self.encoder = TinyNetA()
        else:
            print('backbone error')
            return

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

        #self.asf4 = asf_attention_model(channels[3])
        #self.asf3 = asf_attention_model(channels[2])
        #self.asf2 = asf_attention_model(channels[1])
        #self.asf1 = asf_attention_model(channels[0])
        #self.ssff = ScalSeq([channels[0], channels[1], channels[2]], channels[3])

        self.gold4 = GOLDYOLO_Attention(channels[3])
        self.gold3 = GOLDYOLO_Attention(channels[2])
        self.gold2 = GOLDYOLO_Attention(channels[1])
        self.gold1 = GOLDYOLO_Attention(channels[0])

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

        #fused = self.ssff([x1, x2, x3])
        #fused = F.interpolate(fused, size=out4.shape[2:], mode='bilinear', align_corners=False)

        #out4 = self.asf4([out4, fused])

        x1 = self.gold1(x1)
        x2 = self.gold1(x2)
        x3 = self.gold1(x3)
        out4 = self.gold1(out4)

        out3 = self.gelu(
            self.deconv3(F.interpolate(out4, size=x3.shape[2:], mode='bilinear', align_corners=False)) + x3)
        out2 = self.gelu(
            self.deconv2(F.interpolate(out3, size=x2.shape[2:], mode='bilinear', align_corners=False)) + x2)
        out1 = self.gelu(
            self.deconv1(F.interpolate(out2, size=x1.shape[2:], mode='bilinear', align_corners=False)) + x1)

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
