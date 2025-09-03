"""
Spatial Convolution Module for SCNN
Implements the core spatial message passing mechanism
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpatialConv(nn.Module):
    """
    SCNN spatial convolution module for lane detection
    Processes feature maps in four directions: Down, Up, Right, Left
    """
    def __init__(self, num_channels=128):
        super(SpatialConv, self).__init__()
        
        # Define 1D convolutions for each direction
        self.conv_d = nn.Conv2d(num_channels, num_channels, (1, 9), padding=(0, 4))
        self.conv_u = nn.Conv2d(num_channels, num_channels, (1, 9), padding=(0, 4))
        self.conv_r = nn.Conv2d(num_channels, num_channels, (9, 1), padding=(4, 0))
        self.conv_l = nn.Conv2d(num_channels, num_channels, (9, 1), padding=(4, 0))
        
        self._initialize_weights(num_channels)
    
    def _initialize_weights(self, num_channels):
        """Initialize weights as per SCNN paper"""
        bound = math.sqrt(2.0 / (num_channels * 9 * 5))
        nn.init.uniform_(self.conv_d.weight, -bound, bound)
        nn.init.uniform_(self.conv_u.weight, -bound, bound)
        nn.init.uniform_(self.conv_r.weight, -bound, bound)
        nn.init.uniform_(self.conv_l.weight, -bound, bound)
    
    def non_inplace_forward(self, x):
        """Non-inplace spatial message passing to preserve autograd history"""
        output = x
        vertical = [True, True, False, False]
        reverse = [False, True, False, True]
        convs = [self.conv_d, self.conv_u, self.conv_r, self.conv_l]

        for ver, rev, conv in zip(vertical, reverse, convs):
            slices, dim = self.build_slice(ver, rev, output.shape)
            new_slices = []
            for idx, s in enumerate(slices):
                if new_slices:
                    msg = conv(output[s] if idx == 0 else new_slices[-1])
                    new_slices.append(output[s] + F.relu(msg))
                else:
                    new_slices.append(output[s])
            if rev:
                new_slices = new_slices[::-1]
            output = torch.cat(new_slices, dim=dim)
            if not ver and rev:
                break
        return output

    def build_slice(self, vertical, reverse, shape):
        """Generate slice indices for each row or column"""
        slices = []
        if vertical:
            concat_dim = 2
            length = shape[2]
            for i in range(length):
                slices.append((slice(None), slice(None), slice(i, i+1), slice(None)))
        else:
            concat_dim = 3
            length = shape[3]
            for i in range(length):
                slices.append((slice(None), slice(None), slice(None), slice(i, i+1)))
        if reverse:
            slices = slices[::-1]
        return slices, concat_dim

    def forward(self, x):
        """
        Forward pass through spatial convolution
        Args:
            x: Input feature map [B, C, H, W]
        Returns:
            Spatially processed feature map
        """
        # Always use non-inplace to avoid modifying tensors needed for backward
        return self.non_inplace_forward(x)
