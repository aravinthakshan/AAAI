import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.EfficientNet import EfficientNet_B0
from Model.TinyNet import TinyNetA
from Model.Modules import ConvBNGeLU, ConvBN, DepthwiseSeparableConv
from Model.Replacements import FSM_FFM, CRM
from Model.ccnet import RCCAModule  

# replace with DSC - Ghost - Convs
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

    def forward(self, x):
        x = self.conv(x)
        x = self.conv1(x) + self.conv2(x) + self.conv3(x)
        x = self.bn(x)
        return x


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


class FINet(nn.Module):
    def __init__(self, backbone='efficientb0', channels=(8, 12, 24, 48)):
        super().__init__()

        # Backbone
        if backbone == 'efficientb0':
            self.encoder = EfficientNet_B0()
        elif backbone == 'tinynet-a':
            self.encoder = TinyNetA()
        else:
            raise ValueError('Unsupported backbone')

        stage_channels = self.encoder.get_stage_channels()  # [16, 24, 40, 112, 320]

        # 1x1 reductions to target channels
        self.re_conv1 = ConvBNGeLU(stage_channels[1], channels, kernel_size=1)
        self.re_conv2 = ConvBNGeLU(stage_channels[2], channels[1], kernel_size=1)
        self.re_conv3 = ConvBNGeLU(stage_channels[3], channels[2], kernel_size=1)
        self.re_conv4 = ConvBNGeLU(stage_channels[4], channels[3], kernel_size=1)

        # FSM/FFM (feature modulation + frequency injection)
        self.ffm1 = FSM_FFM(channels)
        self.ffm2 = FSM_FFM(channels[1])
        self.ffm3 = FSM_FFM(channels[2])
        self.ffm4 = FSM_FFM(channels[3])

        # RCCA on deep/mid
        self.rcca_out4 = RCCAModule(channels[3])
        self.rcca_out3 = RCCAModule(channels[2])

        # Paper-style CRM blocks (three stages, cross-scale)
        # Each CRM returns (refined_feature, side_pred)
        self.crm4 = CRM(channels[3], make_side_head=True)  # Stage 4: (x4′, None) -> P1
        self.crm3 = CRM(channels[2], make_side_head=True)  # Stage 3: (x3′, up(F4)) -> P2
        self.crm2 = CRM(channels[1], make_side_head=True)  # Stage 2: (x2′, up(F3)) -> P3

        # Decoder
        self.deconv3 = DeBlock(channels[3], channels[2])  # up(F4) + F3 -> out3
        self.deconv2 = DeBlock(channels[2], channels[1])  # up(out3) + F2 -> out2
        self.deconv1 = DeBlock(channels[1], channels)  # up(out2) + x1 -> out1

        # Output heads at each decoder stage (keep your original heads)
        self.out_conv1 = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.out_conv2 = nn.Conv2d(channels[1], 1, kernel_size=3, padding=1)
        self.out_conv3 = nn.Conv2d(channels[2], 1, kernel_size=3, padding=1)
        self.out_conv4 = nn.Conv2d(channels[3], 1, kernel_size=3, padding=1)

        self.act = nn.GELU()

    def forward(self, x, high, low):
        # Backbone features
        _, f1, f2, f3, f4 = self.encoder(x)  # stages 1..4 (we ignore stage0/stem)

        # Channel reduction
        x1 = self.re_conv1(f1)
        x2 = self.re_conv2(f2)
        x3 = self.re_conv3(f3)
        x4 = self.re_conv4(f4)

        # FSM/FFM modulation
        x1p = self.ffm1(x=x1, high=high, low=low)   # H/4
        x2p = self.ffm2(x=x2, high=high, low=low)   # H/8
        x3p = self.ffm3(x=x3, high=high, low=low)   # H/16
        x4p = self.ffm4(x=x4, high=high, low=low)   # H/32 or H/16 depending on encoder stride

        # CRM at Stage 4 (coarsest): (x4′, None)
        F4, P1 = self.crm4(x4p, None)
        # Optional RCCA at deep feature
        F4 = self.rcca_out4(F4)

        # CRM at Stage 3 (cross-scale with up(F4))
        up4 = F.interpolate(F4, size=x3p.shape[2:], mode='bilinear', align_corners=False)
        F3, P2 = self.crm3(x3p, up4)
        F3 = self.rcca_out3(F3)

        # CRM at Stage 2 (cross-scale with up(F3))
        up3 = F.interpolate(F3, size=x2p.shape[2:], mode='bilinear', align_corners=False)
        F2, P3 = self.crm2(x2p, up3)

        # Decoder fusions (use refined features)
        out3 = self.act(self.deconv3(F.interpolate(F4, size=F3.shape[2:], mode='bilinear', align_corners=False)) + F3)
        out2 = self.act(self.deconv2(F.interpolate(out3, size=F2.shape[2:], mode='bilinear', align_corners=False)) + F2)
        out1 = self.act(self.deconv1(F.interpolate(out2, size=x1p.shape[2:], mode='bilinear', align_corners=False)) + x1p)

        # Decoder heads
        y1 = self.out_conv1(out1)                 # H/4
        y2 = self.out_conv2(out2)                 # H/8
        y3 = self.out_conv3(out3)                 # H/16
        y4 = self.out_conv4(F4)                   # H/16 (deep)

        # Upsample all to a common size for loss/metrics
        final_h = y1.shape[2] * 4
        final_w = y1.shape[3] * 4
        size = (final_h, final_w)

        y1 = F.interpolate(y1, size=size, mode='bilinear', align_corners=False)
        y2 = F.interpolate(y2, size=size, mode='bilinear', align_corners=False)
        y3 = F.interpolate(y3, size=size, mode='bilinear', align_corners=False)
        y4 = F.interpolate(y4, size=size, mode='bilinear', align_corners=False)

        # Auxiliary heads from CRM (P1 at Stage4, P2 at Stage3, P3 at Stage2)
        P1 = F.interpolate(P1, size=size, mode='bilinear', align_corners=False)
        P2 = F.interpolate(P2, size=size, mode='bilinear', align_corners=False)
        P3 = F.interpolate(P3, size=size, mode='bilinear', align_corners=False)

        # Return decoder outputs plus CRM side heads for supervision
        return y1, y2, y3, y4, P1, P2, P3


if __name__ == '__main__':
    # Select device
    from utils.tools import get_model_complexity

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model and move to device
    model = FINet(backbone='efficientb0', channels=(8, 24, 32, 64)).to(device)
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
        out1, out2, out3, out4, _, _, _ = model(x, high, low)

    # Print shapes
    print(f"Input shape: {x.shape}")
    print(f"High freq shape: {high.shape}")
    print(f"Low freq shape: {low.shape}")
    print(f"Output 1 shape: {out1.shape}")
    print(f"Output 2 shape: {out2.shape}")
    print(f"Output 3 shape: {out3.shape}")
    print(f"Output 4 shape: {out4.shape}")

# # Original                 3.740 M 
# # Modified                 3.989 M
# # Cross Attention          4.047 M 
# # TinyCod                  4.720 M