import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class LaplacianPyramid(nn.Module):
    """
    Laplacian Pyramid decomposition module
    Creates multiple frequency bands at different scales
    """
    def __init__(self, num_levels=3):
        super(LaplacianPyramid, self).__init__()
        self.num_levels = num_levels
        
        # Gaussian kernel for smoothing (3x3 kernel for 3-level pyramid)
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
        return F.conv2d(x, self.gaussian_kernel, padding=1, groups=3)
    
    def _downsample(self, x):
        """Downsample by factor of 2"""
        return F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
    
    def _upsample(self, x, target_size):
        """Upsample to target size"""
        return F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
    
    def forward(self, x):
        """
        Create Laplacian pyramid from input image
        Returns: List of Laplacian levels [L0, L1, L2] for 3-level pyramid
        L0 is the finest level, L2 is the coarsest
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

class LDConv(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(LDConv, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),nn.BatchNorm2d(outc),nn.SiLU())  # the conv adds the BN and SiLU to compare original Conv in YOLOv5.
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)
        self.register_buffer("p_n", self._get_p_n(N=self.num_param))

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # resampling the features based on the modified coordinates.
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # bilinear
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)

        return out

    # generating the inital sampled shapes for the LDConv with different sizes.
    def _get_p_n(self, N):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x,p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0,base_int))
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number >  0:
            mod_p_n_x,mod_p_n_y = torch.meshgrid(
                torch.arange(row_number,row_number+1),
                torch.arange(0,mod_number))

            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x,p_n_y  = torch.cat((p_n_x,mod_p_n_x)),torch.cat((p_n_y,mod_p_n_y))
        p_n = torch.cat([p_n_x,p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1)
        return p_n

    # no zero-padding
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        # p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + self.p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h_q, w_q, _ = q.size()  # Spatial dimensions of q (post-stride)
        h_x = x.size(2)  # Height of input x
        w_x = x.size(3)  # Width of input x
        padded_w = w_x   # Use original width for index calculation
        
        # Extract and clamp coordinates to x's spatial dimensions
        q_x = torch.clamp(q[..., :N], 0, h_x - 1)  # Clamp x-coordinates
        q_y = torch.clamp(q[..., N:], 0, w_x - 1)  # Clamp y-coordinates
        
        # Compute flat indices
        index = q_x * padded_w + q_y  # Shape: (b, h_q, w_q, N)
        
        # Validate indices are within bounds
        max_index = h_x * w_x - 1
        if not (torch.all(index >= 0) and torch.all(index <= max_index)):
            invalid = torch.logical_or(index < 0, index > max_index)
            print(f"Invalid indices found: {invalid.sum()} entries")
            raise AssertionError(f"Index out of bounds. Min: {index.min()}, Max: {index.max()}, Allowed: [0, {max_index}]")
        
        # Reshape input x to (b, c, h_x * w_x)
        x_flat = x.contiguous().view(b, -1, h_x * w_x)
        c = x_flat.size(1)
        
        # Prepare index tensor for gathering
        index = index.view(b, h_q, w_q, N)  # Ensure correct shape
        index = index.unsqueeze(1).expand(-1, c, -1, -1, -1)  # Shape: (b, c, h_q, w_q, N)
        index = index.contiguous().view(b, c, -1)  # Flatten to (b, c, h_q * w_q * N)
        
        # Gather values from x_flat using index
        x_offset = x_flat.gather(dim=-1, index=index)  # Shape: (b, c, h_q * w_q * N)
        x_offset = x_offset.view(b, c, h_q, w_q, N)  # Reshape to (b, c, h_q, w_q, N)
        
        return x_offset

    
    #  Stacking resampled features in the row direction.
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        # using Conv3d
        # x_offset = x_offset.permute(0,1,4,2,3), then Conv3d(c,c_out, kernel_size =(num_param,1,1),stride=(num_param,1,1),bias= False)
        # using 1 × 1 Conv
        # x_offset = x_offset.permute(0,1,4,2,3), then, x_offset.view(b,c×num_param,h,w)  finally, Conv2d(c×num_param,c_out, kernel_size =1,stride=1,bias= False)
        # using the column conv as follow， then, Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias)
        
        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset
    
def replace_conv_with_ldconv(module):
    """
    Recursively replace all nn.Conv2d layers with kernel_size=3 
    by LDConv (keeping in/out channels, stride, bias).
    """
    for name, child in module.named_children():
        # If it's a Conv2d with kernel size 3
        if isinstance(child, nn.Conv2d) and child.kernel_size == (3, 3):
            # Build LDConv with matching params
            new_layer = LDConv(
                inc=child.in_channels,
                outc=child.out_channels,
                num_param=3,   
                stride=child.stride[0],
                bias=(child.bias is not None)
            )
            setattr(module, name, new_layer)

        else:
            # Recurse down into children
            replace_conv_with_ldconv(child)

    return module

class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sum(torch.stack(x, dim=0), dim=0)

class asf_channel_att(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(asf_channel_att, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
    
class asf_local_att(nn.Module):
    def __init__(self, channel, reduction=16):
        super(asf_local_att, self).__init__()
        
        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)
 
        self.relu   = nn.ReLU()
        self.bn     = nn.BatchNorm2d(channel//reduction)
 
        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
 
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
 
    def forward(self, x):
        _, _, h, w = x.size()
        
        x_h = torch.mean(x, dim = 3, keepdim = True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim = 2, keepdim = True)
 
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))
 
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)
 
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
 
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out
    
class asf_attention_model(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, ch=256):
        super().__init__()
        self.channel_att = asf_channel_att(ch)
        self.local_att = asf_local_att(ch)
    def forward(self, x):
        input1,input2 = x[0], x[1]
        input1 = self.channel_att(input1)
        x = input1 + input2
        x = self.local_att(x)
        return x
    
class ScalSeq(nn.Module):
    def __init__(self, inc, channel):
        super(ScalSeq, self).__init__()
        if channel != inc[0]:
            self.conv0 = nn.Conv2d(inc[0], channel,1)
        self.conv1 =  nn.Conv2d(inc[1], channel,1)
        self.conv2 =  nn.Conv2d(inc[2], channel,1)
        self.conv3d = nn.Conv3d(channel,channel,kernel_size=(1,1,1))
        self.bn = nn.BatchNorm3d(channel)
        self.act = nn.LeakyReLU(0.1)
        self.pool_3d = nn.MaxPool3d(kernel_size=(3,1,1))

    def forward(self, x):
        p3, p4, p5 = x[0],x[1],x[2]
        if hasattr(self, 'conv0'):
            p3 = self.conv0(p3)
        p4_2 = self.conv1(p4)
        p4_2 = F.interpolate(p4_2, p3.size()[2:], mode='nearest')
        p5_2 = self.conv2(p5)
        p5_2 = F.interpolate(p5_2, p3.size()[2:], mode='nearest')
        p3_3d = torch.unsqueeze(p3, -3)
        p4_3d = torch.unsqueeze(p4_2, -3)
        p5_3d = torch.unsqueeze(p5_2, -3)
        combine = torch.cat([p3_3d, p4_3d, p5_3d],dim = 2)
        conv_3d = self.conv3d(combine)
        bn = self.bn(conv_3d)
        act = self.act(bn)
        x = self.pool_3d(act)
        x = torch.squeeze(x, 2)
        return x
    
def structure_loss(logits, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    
    pred: logits without activation
    mask: binary mask {0, 1}
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(logits, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(logits)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def laplacian_pyramid_loss(pred_pyramid, mask_pyramid, weights=None):
    """
    Multi-scale loss using Laplacian pyramid levels
    
    Args:
        pred_pyramid: List of predicted masks at different scales
        mask_pyramid: List of ground truth masks at different scales  
        weights: Optional weights for different scales
    """
    if weights is None:
        weights = [1.0, 0.8, 0.6, 0.4]
    
    total_loss = 0
    for i, (pred, mask, weight) in enumerate(zip(pred_pyramid, mask_pyramid, weights)):
        # Resize mask to match prediction if needed
        if mask.shape[2:] != pred.shape[2:]:
            mask = F.interpolate(mask, size=pred.shape[2:], mode='bilinear', align_corners=False)
        
        loss = structure_loss(pred, mask)
        total_loss += weight * loss
    
    return total_loss


def create_mask_pyramid(mask, output_shapes):
    """
    Create a pyramid of ground truth masks matching the output shapes
    
    Args:
        mask: Original ground truth mask
        output_shapes: List of tuples containing (H, W) for each output level
    """
    mask_pyramid = []
    
    for shape in output_shapes:
        h, w = shape
        resized_mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
        mask_pyramid.append(resized_mask)
    
    return mask_pyramid


def lap_structure_loss(logits, mask, alpha=0.7, beta=0.3):
    """
    Enhanced loss function combining structure loss with edge-aware loss
    
    Args:
        logits: Predicted logits
        mask: Ground truth mask
        alpha: Weight for structure loss
        beta: Weight for edge loss
    """
    # Ensure mask and logits have the same spatial dimensions
    if mask.shape[2:] != logits.shape[2:]:
        mask = F.interpolate(mask, size=logits.shape[2:], mode='bilinear', align_corners=False)
    
    # Original structure loss
    struct_loss = structure_loss(logits, mask)
    
    # Edge-aware loss using Laplacian operator
    laplacian_kernel = torch.tensor([[[[-1, -1, -1],
                                      [-1,  8, -1],
                                      [-1, -1, -1]]]], dtype=torch.float32).to(logits.device)
    
    pred = torch.sigmoid(logits)
    
    # Apply Laplacian filter to get edges
    pred_edges = F.conv2d(pred, laplacian_kernel, padding=1)
    mask_edges = F.conv2d(mask, laplacian_kernel, padding=1)
    
    # Edge loss
    edge_loss = F.mse_loss(pred_edges, mask_edges)
    
    return alpha * struct_loss + beta * edge_loss

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
 
def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
   
class GOLDYOLO_Attention(torch.nn.Module):
    def __init__(self, dim, key_dim = 2, num_heads = 2, attn_ratio=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        
        self.to_q = Conv(dim, nh_kd, 1, act=False)
        self.to_k = Conv(dim, nh_kd, 1, act=False)
        self.to_v = Conv(dim, self.dh, 1, act=False)
        
        self.proj = torch.nn.Sequential(nn.ReLU6(), Conv(self.dh, dim, act=False))
    
    def forward(self, x):  # x (B,N,C)
        B, C, H, W = get_shape(x)
        
        qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)
        
        attn = torch.matmul(qq, kk)
        attn = attn.softmax(dim=-1)  # dim = k
        
        xx = torch.matmul(attn, vv)
        
        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        xx = self.proj(xx)
        return xx

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv(in_features, hidden_features, act=False)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = nn.ReLU6()
        self.fc2 = Conv(hidden_features, out_features, act=False)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
class top_Block(nn.Module):
    
    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        self.attn = GOLDYOLO_Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
    
    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1