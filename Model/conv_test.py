import torch.nn as nn

class DilatedConv2d(nn.Module):
    """
    A wrapper for nn.Conv2d that applies dilation.
    This allows it to be used as a drop-in replacement for standard Conv2d layers.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=2, bias=True):
        super(DilatedConv2d, self).__init__()
        
        # Note: To maintain the same spatial dimensions with dilation, padding might need adjustment.
        # The formula is: padding = (kernel_size - 1) * (dilation - 1) // 2 + original_padding
        # For kernel_size=3, dilation=2, and original_padding=1, the new padding should be 2.
        # We will adjust it automatically if the kernel is not 1x1.
        if kernel_size > 1:
            padding = (kernel_size - 1) * (dilation - 1) // 2 + padding

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias
        )

    def forward(self, x):
        return self.conv(x)
    
    