import torch
import torch.nn as nn
import torch.nn.functional as F

class LaplacianPyramid(nn.Module):
    """
    Laplacian Pyramid decomposition module
    Creates multiple frequency bands at different scales
    """
    def __init__(self, num_levels=5):
        super(LaplacianPyramid, self).__init__()
        self.num_levels = num_levels
        
        # Gaussian kernel for smoothing (5x5 kernel)
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
        """Apply Gaussian blur to input image"""
        # Apply convolution with Gaussian kernel for each channel separately
        return F.conv2d(x, self.gaussian_kernel, padding=2, groups=3)
    
    def _downsample(self, x):
        """Downsample by factor of 2"""
        return F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
    
    def _upsample(self, x, target_size):
        """Upsample to target size"""
        return F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
    
    def forward(self, x):
        """
        Create Laplacian pyramid from input image
        Returns: List of Laplacian levels [L0, L1, L2, L3, L4]
        L0 is the finest level, L4 is the coarsest
        """
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
        
        # Process Laplacian level
        processed_laplacian = self.laplacian_conv(laplacian_level)
        
        # Concatenate and fuse
        concatenated = torch.cat([encoder_features, processed_laplacian], dim=1)
        fused_features = self.fusion_conv(concatenated)
        
        return fused_features