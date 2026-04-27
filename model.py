"""
Xu et al. (2016) CNN architecture for steganalysis.

Key design principles:
1. ABS layer: Enforces sign symmetry
2. TanH: Prevents overfitting to extreme values
3. 1x1 convolutions: Prevents spatial pattern learning
4. Global average pooling: Prevents location learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class KVFilter(nn.Module):
    # Fixed 5x5 KB high-pass filter for preprocessing (same kernel as Qian et al. [21])

    def __init__(self):
        super(KVFilter, self).__init__()
        # Define the 5x5 KB filter kernel
        kb_kernel = torch.tensor([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8,-12,  8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1]
        ], dtype=torch.float32) / 12.0

        # Register as a buffer since it's not a learnable parameter
        self.register_buffer('kernel', kb_kernel.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        # Apply fixed high-pass filter to extract noise residuals.
        # Pad input with reflection mode for edge handling
        x_padded = F.pad(x, (2, 2, 2, 2), mode='reflect')
        # Apply convolution with the fixed kernel
        output = F.conv2d(x_padded, self.kernel)
        return output


class XuNetSteganalysis(nn.Module):

    """
    CNN architecture for steganalysis based on Xu et al. (2016).
    
    Architecture:
    Input (batch, 1, 512, 512)
        ↓ HPF
    Groups 1-2: Conv + BN + TanH + Pool (early processing)
        ↓ (includes ABS layer in Group 1)
    Groups 3-5: Conv(1x1) + BN + ReLU + Pool (1x1 convs prevent spatial learning)
        ↓
    Global Average Pool
        ↓
    FC layers
        ↓
    Output: (batch, 2) logits [cover, stego]
    """

    def __init__(self):
        super(XuNetSteganalysis, self).__init__()

        # HPF layer (fixed, not learned)
        self.hpf = KVFilter()

        # Group 1: Conv + ABS + BN + TanH + AvgPool
        self.g1_conv = nn.Conv2d(1, 8, kernel_size=5, padding=2, bias=False)    # Bias=False for sign symmetry
        self.g1_bn = nn.BatchNorm2d(8)                                          # Batch normalisation
        self.g1_tanh = nn.Tanh()                                                # Bounded activation (prevents overfitting to extreme values)
        self.g1_pool = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)         # Spatial downsampling via average pooling

        # Group 2: Conv + BN + TanH + AvgPool
        self.g2_conv = nn.Conv2d(8, 16, kernel_size=5, padding=2, bias=False)   # Expand channels from 8 to 16
        self.g2_bn = nn.BatchNorm2d(16)                                         # Batch normalisation
        self.g2_tanh = nn.Tanh()                                                # Still use TanH for value bounding
        self.g2_pool = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)         # Spatial downsampling via average pooling

        # Group 3: Conv(1x1) + BN + ReLU + AvgPool
        self.g3_conv = nn.Conv2d(16, 32, kernel_size=1, padding=0, bias=False)  # 1x1 conv: prevents spatial pattern learning
        self.g3_bn = nn.BatchNorm2d(32)                                         # Batch normalisation
        self.g3_relu = nn.ReLU()                                                # Switch to ReLU in deeper layers
        self.g3_pool = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)         # Spatial downsampling via average pooling

        # Group 4: Conv(1x1) + BN + ReLU + AvgPool
        self.g4_conv = nn.Conv2d(32, 64, kernel_size=1, padding=0, bias=False)  # 1x1 conv: location invariance
        self.g4_bn = nn.BatchNorm2d(64)                                         # Batch normalisation
        self.g4_relu = nn.ReLU()                                                # Standard ReLU activation
        self.g4_pool = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)         # Spatial downsampling via average pooling

        # Group 5: Conv(1x1) + BN + ReLU + GlobalAvgPool
        self.g5_conv = nn.Conv2d(64, 128, kernel_size=1, padding=0, bias=False) # Final feature extraction
        self.g5_bn = nn.BatchNorm2d(128)                                        # Batch normalisation
        self.g5_relu = nn.ReLU()                                                # Standard ReLU activation
        self.g5_global_pool = nn.AdaptiveAvgPool2d(1)                           # Collapse spatial dimensions to prevent location learning

        # Classification module (single FC layer as per paper)
        self.fc = nn.Linear(128, 2) # Output layer (2 classes: cover, stego)

    def forward(self, x):
        """Forward pass through the network."""
        # HPF preprocessing
        x = self.hpf(x)

        # Group 1 with ABS layer
        x = self.g1_conv(x)         # Convolution
        x = torch.abs(x)            # ABS layer to enforce sign symmetry (discard sign information)
        x = self.g1_bn(x)           # Batch normalisation
        x = self.g1_tanh(x)         # Bounded activation
        x = self.g1_pool(x)         # Downsample

        # Group 2
        x = self.g2_conv(x)         # Convolution
        x = self.g2_bn(x)           # Batch normalisation
        x = self.g2_tanh(x)         # Bounded activation
        x = self.g2_pool(x)         # Downsample

        # Group 3 (1x1 convolutions start - prevents spatial pattern learning)
        x = self.g3_conv(x)         # 1x1 Convolution
        x = self.g3_bn(x)           # Batch normalisation
        x = self.g3_relu(x)         # ReLU activation (switch to ReLU in deeper layers)
        x = self.g3_pool(x)         # Downsample

        # Group 4
        x = self.g4_conv(x)         # 1x1 Convolution
        x = self.g4_bn(x)           # Batch normalisation
        x = self.g4_relu(x)         # ReLU activation
        x = self.g4_pool(x)         # Downsample

        # Group 5
        x = self.g5_conv(x)         # 1x1 Convolution
        x = self.g5_bn(x)           # Batch normalisation
        x = self.g5_relu(x)         # ReLU activation  
        x = self.g5_global_pool(x)  # Global average pooling

        # Flatten for FC layers
        x = x.view(x.size(0), -1) # Reshape from (batch, 128, 1, 1) to (batch, 128)

        # Classification module (single FC layer, no dropout per paper)
        x = self.fc(x)          # Output layer - 2 logits (cover, stego)

        return x
