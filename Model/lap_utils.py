import torch
import torch.nn as nn
import torch.nn.functional as F

class LaplacianPyramid(nn.Module):
    """
    Laplacian Pyramid decomposition module
    """
    def __init__(self, num_levels=5):
        super(LaplacianPyramid, self).__init__()
        self.num_levels = num_levels
        
        self.register_buffer('gaussian_kernel', self._get_gaussian_kernel())
        
    def _get_gaussian_kernel(self):
        """Create a 5x5 Gaussian kernel"""
        kernel = torch.tensor([
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1]
        ], dtype=torch.float32) / 256.0
        
        # Expand for 3 channels (RGB)
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        return kernel
    
    def _gaussian_blur(self, x):
        # Apply convolution with Gaussian kernel for each channel separately
        return F.conv2d(x, self.gaussian_kernel, padding=2, groups=3)
    
    def _downsample(self, x):
        return F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
    
    def _upsample(self, x, target_size):
        return F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
    
    def forward(self, x):
        #Returns: List of Laplacian levels [L0, L1, L2, L3, L4]
        #L0 is the finest level, L4 is the coarsest
        
        pyramid_levels = []
        gaussian_pyramid = []
        
        current = x
        
        # Build Gaussian pyramid
        for i in range(self.num_levels):
            if i == 0:
                gaussian_pyramid.append(current)
            else:
                # Blur and downsample
                blurred = self._gaussian_blur(current)
                downsampled = self._downsample(blurred)
                gaussian_pyramid.append(downsampled)
                current = downsampled
        
        # Build Laplacian pyramid
        for i in range(self.num_levels - 1):
            # Upsample the next level
            upsampled = self._upsample(gaussian_pyramid[i + 1], gaussian_pyramid[i].shape[2:])
            # Laplacian = current level - upsampled next level
            laplacian = gaussian_pyramid[i] - upsampled
            pyramid_levels.append(laplacian)
        
        # The last level is just the smallest Gaussian level
        pyramid_levels.append(gaussian_pyramid[-1])
        
        return pyramid_levels


class LaplacianInjectionBlock(nn.Module):
    """
    Block for injecting Laplacian pyramid level into encoder features
    """
    def __init__(self, encoder_channels, laplacian_channels=3, output_channels=None):
        super(LaplacianInjectionBlock, self).__init__()
        
        if output_channels is None:
            output_channels = encoder_channels
            
        # Process Laplacian level to match encoder channels
        self.laplacian_conv = nn.Sequential(
            nn.Conv2d(laplacian_channels, encoder_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_channels // 2),
            nn.GELU(),
            nn.Conv2d(encoder_channels // 2, encoder_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_channels)
        )
        
        # Fusion after concatenation
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(encoder_channels * 2, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels),
            nn.GELU()
        )
        
    def forward(self, encoder_features, laplacian_level):
        """
        Inject Laplacian level into encoder features
        Args:
            encoder_features: Features from encoder at current stage
            laplacian_level: Corresponding Laplacian pyramid level
        """
        # Resize Laplacian level to match encoder features
        if laplacian_level.shape[2:] != encoder_features.shape[2:]:
            laplacian_level = F.interpolate(
                laplacian_level, 
                size=encoder_features.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        processed_laplacian = self.laplacian_conv(laplacian_level)
        
        concatenated = torch.cat([encoder_features, processed_laplacian], dim=1)
        fused_features = self.fusion_conv(concatenated)
        
        return fused_features
    
class LapDecoder(nn.Module):
    """Lap decoder with Low, Middle, and Top branches"""
    def __init__(self, in_channels, out_channels):
        super(LapDecoder, self).__init__()
        
        self.low_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.middle_branch = nn.ModuleList([
            self._make_residual_block(in_channels) for _ in range(7)
        ])
        
        self.top_branch = nn.ModuleList([
            self._make_residual_block(in_channels) for _ in range(2)
        ])
        
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
    def _make_residual_block(self, channels):
        """Creates a residual block"""
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
        low_out = self.low_branch(x)
        
        middle_x = x
        for block in self.middle_branch:
            residual = block(middle_x)
            middle_x = middle_x + F.leaky_relu(residual, 0.2)
        
        top_x = middle_x
        for block in self.top_branch:
            residual = block(top_x)
            top_x = top_x + F.leaky_relu(residual, 0.2)
        
        middle_out = self.final_conv(middle_x)
        top_out = self.final_conv(top_x)
        
        return low_out, middle_out, top_out