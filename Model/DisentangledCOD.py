"""
Disentangled Camouflaged Object Detection Module

Key Idea: Train a (K+1)-class dense predictor with:
- 1 foreground class (camouflaged object)
- K background pattern classes (learned prototypes)

Use disentangling loss to prevent foreground embeddings from 
collapsing into any background cluster.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class PrototypeBank(nn.Module):
    """
    Learnable prototype bank for K background patterns.
    Each prototype represents a distinct background pattern/texture.
    """
    def __init__(self, num_prototypes: int, embed_dim: int, 
                 temperature: float = 0.1, momentum: float = 0.999):
        super().__init__()
        self.num_prototypes = num_prototypes  # K background patterns
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.momentum = momentum
        
        # Learnable background prototypes [K, embed_dim]
        self.register_buffer(
            'prototypes', 
            F.normalize(torch.randn(num_prototypes, embed_dim), dim=1)
        )
        
        # Learnable foreground prototype [1, embed_dim]
        self.register_buffer(
            'fg_prototype',
            F.normalize(torch.randn(1, embed_dim), dim=1)
        )
        
        # Track cluster assignments for analysis
        self.register_buffer('cluster_counts', torch.zeros(num_prototypes))
        
    def update_prototypes(self, embeddings: torch.Tensor, 
                          assignments: torch.Tensor, 
                          mask: torch.Tensor):
        """
        Update prototypes with momentum using current batch.
        
        Args:
            embeddings: [B, C, H, W] dense feature embeddings
            assignments: [B, H, W] cluster assignments (0=fg, 1-K=bg patterns)
            mask: [B, 1, H, W] ground truth binary mask
        """
        B, C, H, W = embeddings.shape
        embeddings_flat = embeddings.permute(0, 2, 3, 1).reshape(-1, C)  # [BHW, C]
        mask_flat = mask.squeeze(1).reshape(-1)  # [BHW]
        assignments_flat = assignments.reshape(-1)  # [BHW]
        
        with torch.no_grad():
            # Update foreground prototype
            fg_mask = mask_flat > 0.5
            if fg_mask.sum() > 0:
                fg_embeddings = embeddings_flat[fg_mask]
                fg_mean = F.normalize(fg_embeddings.mean(dim=0, keepdim=True), dim=1)
                self.fg_prototype = self.momentum * self.fg_prototype + (1 - self.momentum) * fg_mean
                self.fg_prototype = F.normalize(self.fg_prototype, dim=1)
            
            # Update background prototypes
            bg_mask = mask_flat < 0.5
            for k in range(self.num_prototypes):
                cluster_mask = bg_mask & (assignments_flat == (k + 1))
                if cluster_mask.sum() > 10:  # Minimum samples
                    cluster_embeddings = embeddings_flat[cluster_mask]
                    cluster_mean = F.normalize(cluster_embeddings.mean(dim=0, keepdim=True), dim=1)
                    self.prototypes[k] = self.momentum * self.prototypes[k] + (1 - self.momentum) * cluster_mean.squeeze()
                    self.prototypes[k] = F.normalize(self.prototypes[k], dim=0)
                    self.cluster_counts[k] += cluster_mask.sum()
    
    def compute_similarities(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute similarity of embeddings to all prototypes.
        
        Args:
            embeddings: [B, C, H, W] dense features
            
        Returns:
            fg_sim: [B, 1, H, W] similarity to foreground prototype
            bg_sims: [B, K, H, W] similarity to each background prototype
        """
        B, C, H, W = embeddings.shape
        embeddings_norm = F.normalize(embeddings, dim=1)  # [B, C, H, W]
        
        # Reshape for matmul: [B, HW, C]
        embeddings_flat = embeddings_norm.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        # Foreground similarity: [B, HW, 1]
        fg_sim = torch.matmul(embeddings_flat, self.fg_prototype.T) / self.temperature
        fg_sim = fg_sim.reshape(B, H, W, 1).permute(0, 3, 1, 2)  # [B, 1, H, W]
        
        # Background similarities: [B, HW, K]
        bg_sims = torch.matmul(embeddings_flat, self.prototypes.T) / self.temperature
        bg_sims = bg_sims.reshape(B, H, W, self.num_prototypes).permute(0, 3, 1, 2)  # [B, K, H, W]
        
        return fg_sim, bg_sims
    
    def get_assignments(self, bg_sims: torch.Tensor) -> torch.Tensor:
        """Get hard cluster assignments for background pixels"""
        return bg_sims.argmax(dim=1) + 1  # +1 because 0 is foreground


class DisentangledHead(nn.Module):
    """
    Dense prediction head that outputs (K+1) class logits.
    Also produces embeddings for prototype-based learning.
    """
    def __init__(self, in_channels: int, embed_dim: int = 64, 
                 num_bg_patterns: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_bg_patterns = num_bg_patterns
        self.num_classes = num_bg_patterns + 1  # K background + 1 foreground
        
        # Embedding head - produces dense embeddings
        self.embed_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, embed_dim, 1),
        )
        
        # Classification head - produces (K+1) class logits
        self.cls_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, self.num_classes, 1)
        )
        
        # Auxiliary binary head for direct segmentation supervision
        self.binary_head = nn.Conv2d(embed_dim, 1, 1)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W] input features
            
        Returns:
            dict with:
                - embeddings: [B, embed_dim, H, W] dense embeddings
                - class_logits: [B, K+1, H, W] multi-class logits
                - binary_logits: [B, 1, H, W] binary fg/bg logits
        """
        embeddings = self.embed_head(x)
        class_logits = self.cls_head(embeddings)
        binary_logits = self.binary_head(embeddings)
        
        return {
            'embeddings': embeddings,
            'class_logits': class_logits,
            'binary_logits': binary_logits
        }


class DisentanglingLoss(nn.Module):
    """
    Disentangling loss that prevents foreground embeddings from 
    collapsing into any background cluster.
    
    Components:
    1. Contrastive Loss: Push FG away from all BG prototypes
    2. Prototype Separation: Keep BG prototypes distinct from each other
    3. Cluster Compactness: Pull same-cluster embeddings together
    4. Cross-Entropy: Standard classification loss
    5. Binary Supervision: Direct mask supervision
    """
    def __init__(self, num_bg_patterns: int = 8, embed_dim: int = 64,
                 margin: float = 0.5, temperature: float = 0.1,
                 lambda_contrast: float = 1.0,
                 lambda_separation: float = 0.5,
                 lambda_compact: float = 0.3,
                 lambda_entropy: float = 0.1,
                 lambda_binary: float = 1.0):
        super().__init__()
        self.num_bg_patterns = num_bg_patterns
        self.margin = margin
        self.temperature = temperature
        
        # Loss weights
        self.lambda_contrast = lambda_contrast
        self.lambda_separation = lambda_separation
        self.lambda_compact = lambda_compact
        self.lambda_entropy = lambda_entropy
        self.lambda_binary = lambda_binary
        
        # Prototype bank
        self.prototype_bank = PrototypeBank(
            num_prototypes=num_bg_patterns,
            embed_dim=embed_dim,
            temperature=temperature
        )
        
    def forward(self, outputs: Dict[str, torch.Tensor], 
                mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components.
        
        Args:
            outputs: dict from DisentangledHead
            mask: [B, 1, H, W] ground truth binary mask
            
        Returns:
            dict with individual losses and total loss
        """
        embeddings = outputs['embeddings']  # [B, C, H, W]
        class_logits = outputs['class_logits']  # [B, K+1, H, W]
        binary_logits = outputs['binary_logits']  # [B, 1, H, W]
        
        B, C, H, W = embeddings.shape
        
        # Resize mask if needed
        if mask.shape[2:] != embeddings.shape[2:]:
            mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)
        
        # Get prototype similarities
        fg_sim, bg_sims = self.prototype_bank.compute_similarities(embeddings)
        
        # 1. Contrastive Loss: Push FG embeddings away from ALL BG prototypes
        contrast_loss = self._contrastive_loss(embeddings, mask)
        
        # 2. Prototype Separation: Keep BG prototypes distinct
        separation_loss = self._prototype_separation_loss()
        
        # 3. Cluster Compactness: Pull same-cluster BG embeddings together
        compact_loss = self._compactness_loss(embeddings, bg_sims, mask)
        
        # 4. Entropy Regularization: Prevent degenerate solutions
        entropy_loss = self._entropy_regularization(class_logits, mask)
        
        # 5. Binary Segmentation Loss
        binary_loss = self._binary_loss(binary_logits, mask)
        
        # 6. Multi-class Cross Entropy (soft targets from prototypes)
        ce_loss = self._soft_cross_entropy(class_logits, fg_sim, bg_sims, mask)
        
        # Update prototypes
        if self.training:
            assignments = self.prototype_bank.get_assignments(bg_sims)
            self.prototype_bank.update_prototypes(embeddings, assignments, mask)
        
        # Total loss
        total_loss = (
            ce_loss +
            self.lambda_contrast * contrast_loss +
            self.lambda_separation * separation_loss +
            self.lambda_compact * compact_loss +
            self.lambda_entropy * entropy_loss +
            self.lambda_binary * binary_loss
        )
        
        return {
            'total': total_loss,
            'ce': ce_loss,
            'contrast': contrast_loss,
            'separation': separation_loss,
            'compact': compact_loss,
            'entropy': entropy_loss,
            'binary': binary_loss
        }
    
    def _contrastive_loss(self, embeddings: torch.Tensor, 
                          mask: torch.Tensor) -> torch.Tensor:
        """
        Push foreground embeddings away from all background prototypes.
        Uses margin-based triplet-style loss.
        """
        B, C, H, W = embeddings.shape
        embeddings_norm = F.normalize(embeddings, dim=1)
        embeddings_flat = embeddings_norm.permute(0, 2, 3, 1).reshape(-1, C)  # [BHW, C]
        mask_flat = mask.squeeze(1).reshape(-1)  # [BHW]
        
        # Get foreground embeddings
        fg_mask = mask_flat > 0.5
        if fg_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        fg_embeddings = embeddings_flat[fg_mask]  # [N_fg, C]
        
        # Compute similarity to all background prototypes
        # We want: sim(fg, bg_prototype) < -margin (i.e., push apart)
        fg_to_bg_sim = torch.matmul(fg_embeddings, self.prototype_bank.prototypes.T)  # [N_fg, K]
        
        # Margin loss: penalize if fg is too close to any bg prototype
        # loss = max(0, sim + margin)
        margin_violations = F.relu(fg_to_bg_sim + self.margin)
        
        # Take max violation across prototypes (hardest negative)
        max_violation = margin_violations.max(dim=1)[0]
        
        return max_violation.mean()
    
    def _prototype_separation_loss(self) -> torch.Tensor:
        """
        Keep background prototypes separated from each other and from foreground.
        """
        # BG prototypes should be far from each other
        bg_proto_sim = torch.matmul(
            self.prototype_bank.prototypes, 
            self.prototype_bank.prototypes.T
        )
        # Zero out diagonal
        bg_proto_sim = bg_proto_sim - torch.eye(
            self.num_bg_patterns, device=bg_proto_sim.device
        ) * 2
        
        # Penalize high similarity between different prototypes
        bg_separation = F.relu(bg_proto_sim + self.margin).mean()
        
        # FG prototype should be far from all BG prototypes
        fg_to_bg_sim = torch.matmul(
            self.prototype_bank.fg_prototype,
            self.prototype_bank.prototypes.T
        )
        fg_separation = F.relu(fg_to_bg_sim + self.margin).mean()
        
        return bg_separation + fg_separation
    
    def _compactness_loss(self, embeddings: torch.Tensor,
                          bg_sims: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
        """
        Pull embeddings toward their assigned prototype (cluster compactness).
        """
        B, C, H, W = embeddings.shape
        embeddings_norm = F.normalize(embeddings, dim=1)
        embeddings_flat = embeddings_norm.permute(0, 2, 3, 1).reshape(-1, C)
        mask_flat = mask.squeeze(1).reshape(-1)
        
        # Get background embeddings
        bg_mask = mask_flat < 0.5
        if bg_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        bg_embeddings = embeddings_flat[bg_mask]
        
        # Get soft assignments from similarities
        bg_sims_flat = bg_sims.permute(0, 2, 3, 1).reshape(-1, self.num_bg_patterns)
        bg_assignments = F.softmax(bg_sims_flat[bg_mask], dim=1)  # [N_bg, K]
        
        # Compute distance to each prototype
        distances = 1 - torch.matmul(bg_embeddings, self.prototype_bank.prototypes.T)  # [N_bg, K]
        
        # Weighted distance by soft assignment
        weighted_distances = (bg_assignments * distances).sum(dim=1)
        
        return weighted_distances.mean()
    
    def _entropy_regularization(self, class_logits: torch.Tensor,
                                mask: torch.Tensor) -> torch.Tensor:
        """
        Entropy regularization on background regions to encourage
        diverse cluster usage (prevent collapse to single cluster).
        """
        B, K_plus_1, H, W = class_logits.shape
        
        # Get background logits only (exclude foreground class)
        bg_logits = class_logits[:, 1:, :, :]  # [B, K, H, W]
        
        # Flatten
        bg_logits_flat = bg_logits.permute(0, 2, 3, 1).reshape(-1, self.num_bg_patterns)
        mask_flat = mask.squeeze(1).reshape(-1)
        
        # Background pixels only
        bg_mask = mask_flat < 0.5
        if bg_mask.sum() == 0:
            return torch.tensor(0.0, device=class_logits.device)
        
        bg_probs = F.softmax(bg_logits_flat[bg_mask], dim=1)
        
        # Per-pixel entropy (high is good - uncertain assignment)
        pixel_entropy = -(bg_probs * torch.log(bg_probs + 1e-8)).sum(dim=1)
        
        # We want HIGH entropy (diverse assignments), so minimize negative entropy
        # But also want some confidence, so balance
        
        # Global distribution entropy (should be uniform)
        global_probs = bg_probs.mean(dim=0)
        global_entropy = -(global_probs * torch.log(global_probs + 1e-8)).sum()
        max_entropy = torch.log(torch.tensor(self.num_bg_patterns, dtype=torch.float32, device=class_logits.device))
        
        # Penalize low global entropy (cluster collapse)
        entropy_loss = max_entropy - global_entropy
        
        return entropy_loss
    
    def _binary_loss(self, binary_logits: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
        """Standard binary segmentation loss."""
        # Weighted BCE + IoU (same as structure_loss)
        weit = 1 + 5 * torch.abs(
            F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
        )
        wbce = F.binary_cross_entropy_with_logits(binary_logits, mask, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
        
        pred = torch.sigmoid(binary_logits)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        
        return (wbce + wiou).mean()
    
    def _soft_cross_entropy(self, class_logits: torch.Tensor,
                            fg_sim: torch.Tensor,
                            bg_sims: torch.Tensor,
                            mask: torch.Tensor) -> torch.Tensor:
        """
        Cross entropy with soft targets derived from prototype similarities.
        """
        B, K_plus_1, H, W = class_logits.shape
        
        # Construct soft targets
        # Foreground pixels: high weight on class 0
        # Background pixels: soft assignment based on prototype similarity
        
        # Combine similarities: [B, K+1, H, W]
        all_sims = torch.cat([fg_sim, bg_sims], dim=1)
        soft_targets = F.softmax(all_sims, dim=1)
        
        # Use mask to guide: foreground should predict class 0
        fg_target = torch.zeros_like(soft_targets)
        fg_target[:, 0, :, :] = 1.0  # Class 0 is foreground
        
        # Blend based on mask
        mask_expanded = mask.expand_as(soft_targets)
        targets = mask_expanded * fg_target + (1 - mask_expanded) * soft_targets
        
        # Soft cross entropy
        log_probs = F.log_softmax(class_logits, dim=1)
        ce_loss = -(targets * log_probs).sum(dim=1).mean()
        
        return ce_loss


class DisentangledFINet(nn.Module):
    """
    FINet with disentangled (K+1)-class prediction head.
    """
    def __init__(self, base_model: nn.Module, embed_dim: int = 64,
                 num_bg_patterns: int = 8):
        super().__init__()
        self.base_model = base_model
        self.embed_dim = embed_dim
        self.num_bg_patterns = num_bg_patterns
        
        # Get output channels from base model
        # Assuming the base model outputs features before final conv
        # We'll intercept at the decoder output stage
        
        # Disentangled heads for each scale
        # Channels from LaFINet: (8, 24, 32, 64)
        self.disentangle_head1 = DisentangledHead(8, embed_dim, num_bg_patterns)
        self.disentangle_head2 = DisentangledHead(24, embed_dim, num_bg_patterns)
        self.disentangle_head3 = DisentangledHead(32, embed_dim, num_bg_patterns)
        self.disentangle_head4 = DisentangledHead(64, embed_dim, num_bg_patterns)
        
    def forward(self, x: torch.Tensor, high: torch.Tensor, low: torch.Tensor,
                return_intermediate: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with disentangled outputs.
        """
        # Get base model features (need to modify base model to return intermediates)
        # For now, we'll use a hook-based approach or modify the base model
        
        # This is a simplified version - you'd need to modify the base model
        # to return intermediate features before the out_conv layers
        out1, out2, out3, out4 = self.base_model(x, high, low)
        
        return {
            'out1': out1,
            'out2': out2,
            'out3': out3,
            'out4': out4
        }


# Utility function to create pseudo-labels for background patterns
def create_bg_pseudo_labels(image: torch.Tensor, mask: torch.Tensor,
                            num_clusters: int = 8) -> torch.Tensor:
    """
    Create pseudo-labels for background patterns using simple clustering.
    Can be used for initialization or supervised pre-training.
    
    Args:
        image: [B, 3, H, W] input images
        mask: [B, 1, H, W] binary masks
        num_clusters: number of background patterns
        
    Returns:
        labels: [B, H, W] with 0=foreground, 1-K=background patterns
    """
    B, C, H, W = image.shape
    labels = torch.zeros(B, H, W, device=image.device, dtype=torch.long)
    
    for b in range(B):
        img = image[b].permute(1, 2, 0).reshape(-1, C)  # [HW, 3]
        m = mask[b, 0].reshape(-1)  # [HW]
        
        # Background pixels
        bg_mask = m < 0.5
        bg_pixels = img[bg_mask]
        
        if bg_pixels.shape[0] > num_clusters:
            # Simple k-means-like clustering based on color
            from torch.cluster import KMeans  # or use sklearn
            # Simplified: use quantization
            bg_normalized = (bg_pixels - bg_pixels.min()) / (bg_pixels.max() - bg_pixels.min() + 1e-8)
            cluster_ids = (bg_normalized.mean(dim=1) * num_clusters).long().clamp(0, num_clusters - 1)
            
            # Assign labels
            full_labels = torch.zeros(H * W, device=image.device, dtype=torch.long)
            full_labels[bg_mask] = cluster_ids + 1  # +1 because 0 is foreground
            labels[b] = full_labels.reshape(H, W)
    
    return labels


