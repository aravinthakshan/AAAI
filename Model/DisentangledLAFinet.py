"""
Disentangled LaplacianFINet Integration

Modifies LAFinet to output both:
1. Standard binary segmentation
2. (K+1)-class disentangled predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.EfficientNet import EfficientNet_B0
from Model.TinyNet import TinyNetA
from Model.Modules import ConvBNGeLU, ConvBN, DepthwiseSeparableConv
from Model.lap_utils import LaplacianPyramid, LaplacianInjectionBlock, top_Block
from Model.Replacements import FSM_FFM
from Model.DisentangledCOD import DisentangledHead, DisentanglingLoss, PrototypeBank


class DisentangledDecoder(nn.Module):
    """
    Decoder block that outputs both features and disentangled representations.
    """
    def __init__(self, in_channels, out_channels, embed_dim=64, num_bg_patterns=8):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.conv1 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), bias=True)
        self.conv3 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        
        # Disentangled head
        self.disentangle_head = DisentangledHead(out_channels, embed_dim, num_bg_patterns)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.conv1(x) + self.conv2(x) + self.conv3(x)
        features = self.bn(x)
        
        # Get disentangled outputs
        disentangle_out = self.disentangle_head(features)
        
        return features, disentangle_out


class DisentangledLAFinet(nn.Module):
    """
    LaplacianFINet with (K+1)-class disentangled prediction.
    
    Key modifications:
    1. Each decoder stage outputs embeddings for disentanglement
    2. Multi-scale prototype learning
    3. Foreground-background disentangling loss integration
    """
    
    def __init__(self, backbone='efficientb0', channels=(8, 24, 32, 64),
                 embed_dim=64, num_bg_patterns=8):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_bg_patterns = num_bg_patterns
        self.num_classes = num_bg_patterns + 1
        
        # Backbone encoder
        if backbone == 'efficientb0':
            self.encoder = EfficientNet_B0()
        elif backbone == 'tinynet-a':
            self.encoder = TinyNetA()
        else:
            raise ValueError(f'Unknown backbone: {backbone}')
        
        # Laplacian Pyramid
        self.laplacian_pyramid = LaplacianPyramid(num_levels=3)
        
        stage_channels = self.encoder.get_stage_channels()
        
        # Laplacian injection blocks
        self.lap_injection1 = LaplacianInjectionBlock(stage_channels[1], 3, stage_channels[1])
        self.lap_injection2 = LaplacianInjectionBlock(stage_channels[2], 3, stage_channels[2])
        self.lap_injection3 = LaplacianInjectionBlock(stage_channels[3], 3, stage_channels[3])
        
        # Channel reduction
        self.re_conv1 = ConvBNGeLU(stage_channels[1], channels[0], kernel_size=1)
        self.re_conv2 = ConvBNGeLU(stage_channels[2], channels[1], kernel_size=1)
        self.re_conv3 = ConvBNGeLU(stage_channels[3], channels[2], kernel_size=1)
        self.re_conv4 = ConvBNGeLU(stage_channels[4], channels[3], kernel_size=1)
        
        # Frequency fusion modules
        self.ffm1 = FSM_FFM(channels[0])
        self.ffm2 = FSM_FFM(channels[1])
        self.ffm3 = FSM_FFM(channels[2])
        self.ffm4 = FSM_FFM(channels[3])
        
        # Attention blocks
        self.gold1 = top_Block(channels[0])
        self.gold2 = top_Block(channels[1])
        self.gold3 = top_Block(channels[2])
        self.gold4 = top_Block(channels[3])
        
        # Disentangled decoders
        self.deconv3 = DisentangledDecoder(channels[3], channels[2], embed_dim, num_bg_patterns)
        self.deconv2 = DisentangledDecoder(channels[2], channels[1], embed_dim, num_bg_patterns)
        self.deconv1 = DisentangledDecoder(channels[1], channels[0], embed_dim, num_bg_patterns)
        
        # Final disentangle head for stage 4
        self.disentangle_head4 = DisentangledHead(channels[3], embed_dim, num_bg_patterns)
        
        # Binary output convolutions (for backward compatibility)
        self.out_conv1 = nn.Conv2d(channels[0], 1, kernel_size=3, padding=1)
        self.out_conv2 = nn.Conv2d(channels[1], 1, kernel_size=3, padding=1)
        self.out_conv3 = nn.Conv2d(channels[2], 1, kernel_size=3, padding=1)
        self.out_conv4 = nn.Conv2d(channels[3], 1, kernel_size=3, padding=1)
        
        # Multi-class output convolutions
        self.cls_conv1 = nn.Conv2d(embed_dim, self.num_classes, kernel_size=1)
        self.cls_conv2 = nn.Conv2d(embed_dim, self.num_classes, kernel_size=1)
        self.cls_conv3 = nn.Conv2d(embed_dim, self.num_classes, kernel_size=1)
        self.cls_conv4 = nn.Conv2d(embed_dim, self.num_classes, kernel_size=1)
        
        self.gelu = nn.GELU()
        
    def forward(self, x, high, low, return_disentangled=True):
        """
        Forward pass.
        
        Args:
            x: [B, 3, H, W] input image
            high: [B, 96, h, w] high frequency features
            low: [B, 96, h, w] low frequency features
            return_disentangled: whether to return disentangled outputs
            
        Returns:
            If return_disentangled=False:
                out1, out2, out3, out4 (binary logits, for compatibility)
            If return_disentangled=True:
                dict with binary outputs and disentangled outputs
        """
        # Laplacian pyramid
        laplacian_levels = self.laplacian_pyramid(x)
        
        # Encoder forward
        x0, x1, x2, x3, x4 = self.encoder(x)
        
        # Laplacian injection
        x1 = self.lap_injection1(x1, laplacian_levels[0])
        x2 = self.lap_injection2(x2, laplacian_levels[1])
        x3 = self.lap_injection3(x3, laplacian_levels[2])
        
        # Channel reduction
        x1 = self.re_conv1(x1)
        x2 = self.re_conv2(x2)
        x3 = self.re_conv3(x3)
        x4 = self.re_conv4(x4)
        
        # Frequency fusion
        x1 = self.ffm1(x=x1, high=high, low=low)
        x2 = self.ffm2(x=x2, high=high, low=low)
        x3 = self.ffm3(x=x3, high=high, low=low)
        feat4 = self.ffm4(x=x4, high=high, low=low)
        
        # Attention
        x1 = self.gold1(x1)
        x2 = self.gold2(x2)
        x3 = self.gold3(x3)
        feat4 = self.gold4(feat4)
        
        # Stage 4 disentangled output
        disentangle4 = self.disentangle_head4(feat4)
        
        # Decoder with disentanglement
        feat3_up = F.interpolate(feat4, size=x3.shape[2:], mode='bilinear', align_corners=False)
        feat3, disentangle3 = self.deconv3(feat3_up)
        out3 = self.gelu(feat3 + x3)
        
        feat2_up = F.interpolate(out3, size=x2.shape[2:], mode='bilinear', align_corners=False)
        feat2, disentangle2 = self.deconv2(feat2_up)
        out2 = self.gelu(feat2 + x2)
        
        feat1_up = F.interpolate(out2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        feat1, disentangle1 = self.deconv1(feat1_up)
        out1 = self.gelu(feat1 + x1)
        
        # Binary outputs
        binary1 = self.out_conv1(out1)
        binary2 = self.out_conv2(out2)
        binary3 = self.out_conv3(out3)
        binary4 = self.out_conv4(feat4)
        
        # Upsample to input size
        size = (binary1.shape[2] * 4, binary1.shape[3] * 4)
        binary1 = F.interpolate(binary1, size=size, mode='bilinear', align_corners=False)
        binary2 = F.interpolate(binary2, size=size, mode='bilinear', align_corners=False)
        binary3 = F.interpolate(binary3, size=size, mode='bilinear', align_corners=False)
        binary4 = F.interpolate(binary4, size=size, mode='bilinear', align_corners=False)
        
        if not return_disentangled:
            return binary1, binary2, binary3, binary4
        
        # Return full outputs
        return {
            'binary': {
                'out1': binary1,
                'out2': binary2,
                'out3': binary3,
                'out4': binary4
            },
            'disentangled': {
                'scale1': disentangle1,
                'scale2': disentangle2,
                'scale3': disentangle3,
                'scale4': disentangle4
            },
            'features': {
                'feat1': out1,
                'feat2': out2,
                'feat3': out3,
                'feat4': feat4
            }
        }


class MultiScaleDisentanglingLoss(nn.Module):
    """
    Multi-scale disentangling loss for all decoder stages.
    """
    def __init__(self, num_bg_patterns=8, embed_dim=64,
                 scale_weights=(1.0, 0.8, 0.6, 0.4)):
        super().__init__()
        
        self.scale_weights = scale_weights
        
        # Shared prototype bank across scales
        self.disentangle_loss = DisentanglingLoss(
            num_bg_patterns=num_bg_patterns,
            embed_dim=embed_dim
        )
        
    def forward(self, outputs, mask):
        """
        Compute multi-scale disentangling loss.
        
        Args:
            outputs: dict from DisentangledLAFinet
            mask: [B, 1, H, W] ground truth mask
        """
        total_loss = 0
        loss_dict = {}
        
        disentangled = outputs['disentangled']
        
        for i, (scale_name, scale_out) in enumerate(disentangled.items()):
            weight = self.scale_weights[i] if i < len(self.scale_weights) else 0.4
            
            # Resize mask to match scale
            H, W = scale_out['embeddings'].shape[2:]
            mask_scaled = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)
            
            # Compute loss for this scale
            scale_losses = self.disentangle_loss(scale_out, mask_scaled)
            
            total_loss += weight * scale_losses['total']
            loss_dict[f'{scale_name}_total'] = scale_losses['total'].item()
            loss_dict[f'{scale_name}_contrast'] = scale_losses['contrast'].item()
        
        loss_dict['total'] = total_loss
        
        return total_loss, loss_dict


def get_disentangled_model(backbone='efficientb0', channels=(8, 24, 32, 64),
                           embed_dim=64, num_bg_patterns=8):
    """
    Factory function to create disentangled model and loss.
    """
    model = DisentangledLAFinet(
        backbone=backbone,
        channels=channels,
        embed_dim=embed_dim,
        num_bg_patterns=num_bg_patterns
    )
    
    loss_fn = MultiScaleDisentanglingLoss(
        num_bg_patterns=num_bg_patterns,
        embed_dim=embed_dim
    )
    
    return model, loss_fn


if __name__ == '__main__':
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model, loss_fn = get_disentangled_model()
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    
    # Test inputs
    x = torch.randn(2, 3, 384, 384).to(device)
    high = torch.randn(2, 96, 48, 48).to(device)
    low = torch.randn(2, 96, 48, 48).to(device)
    mask = torch.randint(0, 2, (2, 1, 384, 384)).float().to(device)
    
    # Forward pass
    outputs = model(x, high, low, return_disentangled=True)
    
    print("Binary outputs:")
    for k, v in outputs['binary'].items():
        print(f"  {k}: {v.shape}")
    
    print("\nDisentangled outputs:")
    for k, v in outputs['disentangled'].items():
        print(f"  {k}:")
        for kk, vv in v.items():
            print(f"    {kk}: {vv.shape}")
    
    # Compute loss
    loss, loss_dict = loss_fn(outputs, mask)
    print(f"\nTotal loss: {loss.item():.4f}")
    print("Loss components:")
    for k, v in loss_dict.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    
    # Parameter count
    params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {params:,}")


