import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

def structure_loss(logits, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(logits, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(logits)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def laplacian_pyramid(img, levels=3):
    pyr = []
    current = img

    for _ in range(levels):
        # Gaussian downsample
        down = F.interpolate(current, scale_factor=0.5, mode='bilinear', align_corners=False)

        # Upsample back to original size
        up = F.interpolate(down, size=current.shape[2:], mode='bilinear', align_corners=False)

        # Laplacian = current - upsampled(gaussian_down)
        lap = current - up
        pyr.append(lap)

        # Move to next level
        current = down

    return pyr


def lap_structure_loss(logits, mask, alpha=0.7, beta=0.3, levels=3):
    # Resize GT mask if necessary
    if mask.shape[2:] != logits.shape[2:]:
        mask = F.interpolate(mask, size=logits.shape[2:], mode='bilinear', align_corners=False)

    # Base structural loss
    struct_loss = structure_loss(logits, mask)

    # Sigmoid for prediction map
    pred = torch.sigmoid(logits)

    # Build Laplacian pyramids for pred & mask
    pred_pyr = laplacian_pyramid(pred, levels)
    mask_pyr = laplacian_pyramid(mask, levels)

    # Multi-scale MSE across each level
    lap_loss = 0.0
    for p, m in zip(pred_pyr, mask_pyr):
        lap_loss += F.mse_loss(p, m)

    # Average over levels
    lap_loss /= levels

    # Combine
    return alpha * struct_loss + beta * lap_loss

class LapLoss(nn.Module):
    def __init__(self, levels=3, alpha=0.7, beta=0.3, use_charbonnier=False):
        super().__init__()
        self.levels = levels
        self.alpha = alpha
        self.beta = beta
        self.use_charbonnier = use_charbonnier

    def laplacian_pyramid(self, img):
        pyr = []
        current = img

        for _ in range(self.levels):
            down = F.interpolate(current, scale_factor=0.5, mode='bilinear', align_corners=False)
            up   = F.interpolate(down, size=current.shape[2:], mode='bilinear', align_corners=False)
            lap  = current - up
            pyr.append(lap)
            current = down

        return pyr

    def lap_mse(self, pred_pyr, mask_pyr):
        lap_loss = 0
        for p, m in zip(pred_pyr, mask_pyr):
            if self.use_charbonnier:
                lap_loss += torch.mean(torch.sqrt((p - m)**2 + 1e-6))
            else:
                lap_loss += F.mse_loss(p, m)
        return lap_loss / self.levels

    def forward(self, logits, mask):
        # Resize GT if required
        if mask.shape[2:] != logits.shape[2:]:
            mask = F.interpolate(mask, size=logits.shape[2:], mode='bilinear', align_corners=False)

        # Structural loss (F3Net)
        struct_loss = structure_loss(logits, mask)

        # Image-space sigmoid
        pred = torch.sigmoid(logits)

        # Pyramids
        pred_pyr = self.laplacian_pyramid(pred)
        mask_pyr = self.laplacian_pyramid(mask)

        # Laplacian MSE loss
        lap_loss = self.lap_mse(pred_pyr, mask_pyr)

        # Combine
        return self.alpha * struct_loss + self.beta * lap_loss



