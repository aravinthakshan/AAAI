import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.EfficientNet import EfficientNet_B0
from Model.TinyNet import TinyNetA
from Model.Modules import ConvBNGeLU, ConvBN, DepthwiseSeparableConv
from Model.lap_utils import LaplacianPyramid, LaplacianInjectionBlock
from Model.lap_utils import LapDecoder
from Model.Replacements import FSM_FFM
from Model.LDconv import LDConv

class DeBlock(nn.Module):
    # Decoder Block with LAPDecoder integration
    def __init__(self, in_channels, out_channels):
        super(DeBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=True)
        self.conv1 = DepthwiseSeparableConv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True)
        self.conv2 = DepthwiseSeparableConv(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 3), padding=(0, 1), bias=True)
        self.conv3 = DepthwiseSeparableConv(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1), padding=(1, 0), bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        
        # Add LAP decoder
        self.lap_decoder = LapDecoder(out_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv1(x) + self.conv2(x) + self.conv3(x)
        x = self.bn(x)
        
        # Apply LAP decoder
        low_out, middle_out, top_out = self.lap_decoder(x)
        
        combined = low_out + middle_out + top_out
        
        return combined

class LFA(nn.Module): # This is the LFIM
    """
    Low Frequency Injection Module
    """
    def __init__(self, channels):
        super(LFA, self).__init__()

        # local_att
        self.local_att = nn.Sequential(
            # keep spatial dimension
            # squeeze
            nn.Conv2d(channels, channels // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.GELU(),
            # excitation
            nn.Conv2d(channels // 2, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels)
        )

        # global_att
        self.global_att = nn.Sequential(
            # squeeze spatial dimension
            nn.AdaptiveAvgPool2d(1),
            # squeeze channel
            nn.Conv2d(channels, channels // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.GELU(),
            # excite channel
            nn.Conv2d(channels // 2, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

        self.conv = ConvBN(in_channels=channels, out_channels=channels, kernel_size=1)

    def forward(self, x):
        x = self.local_att(x) + self.global_att(x) * x
        x = self.conv(x)
        return x

class HFA(nn.Module): # This is the HFIM 
    """
    High Frequency Injection Module
    """
    def __init__(self, channels):
        super(HFA, self).__init__()

        # local_att
        self.local_att = nn.Sequential(
            # keep channel dimension
            # squeeze
            DepthwiseSeparableConv(channels, channels, kernel_size=3, padding=1, bias=False, stride=2),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            # excitation
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparableConv(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        # global_att
        self.global_att = nn.Sequential(
            # squeeze channel dimension in forward function
            # squeeze spatial
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.GELU(),
            # excitation spatial
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparableConv(1, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        self.conv = ConvBN(in_channels=channels, out_channels=channels, kernel_size=1)

    def forward(self, x):
        x = self.local_att(x) + self.global_att(torch.mean(x, dim=1, keepdim=True)) * x
        x = self.conv(x)
        return x


class FFM(nn.Module):
    """Enhanced Frequency Injection Module with Laplacian integration"""
    def __init__(self, channel):
        super(FFM, self).__init__()

        # Original
        self.high_reconv = ConvBNGeLU(in_channels=96, out_channels=channel, kernel_size=1)
        self.low_reconv = ConvBNGeLU(in_channels=96, out_channels=channel, kernel_size=1)

        self.high_reconv2 = ConvBN(in_channels=channel * 2, out_channels=channel, kernel_size=1)
        self.low_reconv2 = ConvBN(in_channels=channel * 2, out_channels=channel, kernel_size=1)

        self.high_msca = HFA(channels=channel)
        self.low_msca = LFA(channels=channel)

        # Fusion with more channels for Laplacian features ( Might need to change this later )
        self.enhanced_fusion = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.GELU(),
            nn.Conv2d(channel, channel, kernel_size=1)
        )

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

        # Enhanced fusion
        combined = torch.cat([high_x, low_x], dim=1)
        x = self.enhanced_fusion(combined)
        x = self.gelu(x)
        x = self.conv(x)

        return x


class LaFINet(nn.Module):    
    def __init__(self, backbone='efficientb0', channels=(8, 12, 24, 48), use_conv='LDConv'):
        super(LaFINet, self).__init__()

        self.use_conv = use_conv

        if backbone == 'efficientb0':
            self.encoder = EfficientNet_B0()
        elif backbone == 'tinynet-a':
            self.encoder = TinyNetA()
        else:
            print('backbone error')
            return

        self.laplacian_pyramid = LaplacianPyramid(num_levels=5)
        stage_channels = self.encoder.get_stage_channels()

        self.lap_injection1 = LaplacianInjectionBlock(stage_channels[0], 3, stage_channels[0])
        self.lap_injection2 = LaplacianInjectionBlock(stage_channels[1], 3, stage_channels[1])
        self.lap_injection3 = LaplacianInjectionBlock(stage_channels[2], 3, stage_channels[2])
        self.lap_injection4 = LaplacianInjectionBlock(stage_channels[3], 3, stage_channels[3])
        self.lap_injection5 = LaplacianInjectionBlock(stage_channels[4], 3, stage_channels[4])
        
        self.re_conv1 = ConvBNGeLU(in_channels=stage_channels[1], out_channels=channels[0], kernel_size=1)
        self.re_conv2 = ConvBNGeLU(in_channels=stage_channels[2], out_channels=channels[1], kernel_size=1)
        self.re_conv3 = ConvBNGeLU(in_channels=stage_channels[3], out_channels=channels[2], kernel_size=1)
        self.re_conv4 = ConvBNGeLU(in_channels=stage_channels[4], out_channels=channels[3], kernel_size=1)
        
        self.ffm1 = FSM_FFM(channels[0]) 
        self.ffm2 = FSM_FFM(channels[1])
        self.ffm3 = FSM_FFM(channels[2])
        self.ffm4 = FSM_FFM(channels[3])
        
        self.gelu = nn.GELU()
        
        # ---- DecoderBlocks ----
        self.deconv3 = DeBlock(channels[3], channels[2])
        self.deconv2 = DeBlock(channels[2], channels[1])
        self.deconv1 = DeBlock(channels[1], channels[0])
        
        if use_conv == 'LDConv':
            from Model.LDconv import LDConv
            ConvLayer = LDConv
        else:
            ConvLayer = nn.Conv2d

        self.out_conv1 = ConvLayer(channels[0], 1, kernel_size=3, padding=1)
        self.out_conv2 = ConvLayer(channels[1], 1, kernel_size=3, padding=1)
        self.out_conv3 = ConvLayer(channels[2], 1, kernel_size=3, padding=1)
        self.out_conv4 = ConvLayer(channels[3], 1, kernel_size=3, padding=1)

    def forward(self, x, high, low):
        laplacian_levels = self.laplacian_pyramid(x)
        
        x0, x1, x2, x3, x4 = self.encoder(x)
        
        x1 = self.lap_injection2(x1, laplacian_levels[1])  # L1 -> stage 1
        x2 = self.lap_injection3(x2, laplacian_levels[2])  # L2 -> stage 2
        x3 = self.lap_injection4(x3, laplacian_levels[3])  # L3 -> stage 3
        x4 = self.lap_injection5(x4, laplacian_levels[4])  # L4 -> stage 4
        
        x1 = self.re_conv1(x1)
        x2 = self.re_conv2(x2)
        x3 = self.re_conv3(x3)
        x4 = self.re_conv4(x4)
        
        x1 = self.ffm1(x=x1, high=high, low=low)
        x2 = self.ffm2(x=x2, high=high, low=low)
        x3 = self.ffm3(x=x3, high=high, low=low)
        out4 = self.ffm4(x=x4, high=high, low=low)
        
        out3 = self.gelu(
            self.deconv3(F.interpolate(out4, size=x3.shape[2:], mode='bilinear', align_corners=False)) + x3)
        out2 = self.gelu(
            self.deconv2(F.interpolate(out3, size=x2.shape[2:], mode='bilinear', align_corners=False)) + x2)
        out1 = self.gelu(
            self.deconv1(F.interpolate(out2, size=x1.shape[2:], mode='bilinear', align_corners=False)) + x1)

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
    


if __name__ == '__main__':
    # Select device
    from utils.tools import get_model_complexity

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model and move to device
    model = LaFINet(backbone='efficientb0', channels=(8, 24, 32, 64)).to(device)
    # model = FINet(backbone='tinynet-a', channels=(8,24,32,64)).to(device)

    # Compute FLOPs and Params
    flops, params = get_model_complexity(
        model,
        inputs=(
            torch.randn(1, 3, 384, 384, device=device),
            torch.randn(1, 96, 48, 48, device=device),
            torch.randn(1, 96, 48, 48, device=device),
        ),
        round=3
    )
    print(f"Params: {params}, FLOPs: {flops}")

    # Evaluation mode
    model.eval()

    # Input sizes
    batch_size = 1
    input_height, input_width = 384, 384
    freq_height, freq_width = 48, 48

    # Create sample inputs directly on device
    x = torch.randn(batch_size, 3, input_height, input_width, device=device)
    high = torch.randn(batch_size, 96, freq_height, freq_width, device=device)
    low = torch.randn(batch_size, 96, freq_height, freq_width, device=device)

    # Forward pass
    with torch.no_grad():
        out1, out2, out3, out4 = model(x, high, low)

    # Print shapes
    print(f"Input shape: {x.shape}")
    print(f"High freq shape: {high.shape}")
    print(f"Low freq shape: {low.shape}")
    print(f"Output 1 shape: {out1.shape}")
    print(f"Output 2 shape: {out2.shape}")
    print(f"Output 3 shape: {out3.shape}")
    print(f"Output 4 shape: {out4.shape}")
