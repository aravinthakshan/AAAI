import torch
import numpy as np
import torch.nn.functional as F

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

def create_mask_pyramid(mask, output_shapes):
    mask_pyramid = []
    
    for shape in output_shapes:
        h, w = shape
        resized_mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
        mask_pyramid.append(resized_mask)
    
    return mask_pyramid


def lap_structure_loss(logits, mask, alpha=0.7, beta=0.3):
    if mask.shape[2:] != logits.shape[2:]:
        mask = F.interpolate(mask, size=logits.shape[2:], mode='bilinear', align_corners=False)
    
    struct_loss = structure_loss(logits, mask)
    
    laplacian_kernel = torch.tensor([[[[-1, -1, -1],
                                      [-1,  8, -1],
                                      [-1, -1, -1]]]], dtype=torch.float32).to(logits.device)
    
    pred = torch.sigmoid(logits)
    
    pred_edges = F.conv2d(pred, laplacian_kernel, padding=1)
    mask_edges = F.conv2d(mask, laplacian_kernel, padding=1)
    
    edge_loss = F.mse_loss(pred_edges, mask_edges)
    
    return alpha * struct_loss + beta * edge_loss
