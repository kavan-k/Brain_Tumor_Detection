"""
Model definition module.
Contains 2D U-Net and 3D U-Net architectures in PyTorch.
Contributor: Person C

NOTE: PyTorch uses channel-first format: (N, C, H, W) for 2D, (N, C, D, H, W) for 3D.
Input to 2D model: (N, 4, 128, 128)
Input to 3D model: (N, 4, D, H, W)
"""

import torch
import torch.nn as nn
from config import MODEL_TYPE, PATCH_SIZE


# Shared building blocks

class ConvBlock2D(nn.Module):
    """Two consecutive Conv2D → BN → ReLU layers."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvBlock3D(nn.Module):
    """Two consecutive Conv3D → BN → ReLU layers."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

# 2D U-Net

class UNet2D(nn.Module):
    """
    Standard 2D U-Net with BatchNorm.
    Input shape : (N, 4, 128, 128)   — 4 MRI modalities, channel-first
    Output shape: (N, 1, 128, 128)   — binary segmentation mask (sigmoid)
    """
    def __init__(self, in_channels=4):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock2D(in_channels, 64)
        self.enc2 = ConvBlock2D(64, 128)
        self.enc3 = ConvBlock2D(128, 256)
        self.enc4 = ConvBlock2D(256, 512)
        self.pool = nn.MaxPool2d(2)

        # Bridge
        self.bridge = ConvBlock2D(512, 1024)

        # Decoder
        self.up6  = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec6 = ConvBlock2D(1024, 512)   # 512 (up) + 512 (skip) = 1024

        self.up7  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec7 = ConvBlock2D(512, 256)

        self.up8  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec8 = ConvBlock2D(256, 128)

        self.up9  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec9 = ConvBlock2D(128, 64)

        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.enc1(x);  p1 = self.pool(c1)
        c2 = self.enc2(p1); p2 = self.pool(c2)
        c3 = self.enc3(p2); p3 = self.pool(c3)
        c4 = self.enc4(p3); p4 = self.pool(c4)

        # Bridge
        c5 = self.bridge(p4)

        # Decoder
        u6 = self.up6(c5);  u6 = torch.cat([u6, c4], dim=1); c6 = self.dec6(u6)
        u7 = self.up7(c6);  u7 = torch.cat([u7, c3], dim=1); c7 = self.dec7(u7)
        u8 = self.up8(c7);  u8 = torch.cat([u8, c2], dim=1); c8 = self.dec8(u8)
        u9 = self.up9(c8);  u9 = torch.cat([u9, c1], dim=1); c9 = self.dec9(u9)

        return torch.sigmoid(self.out_conv(c9))

# 3D U-Net

class UNet3D(nn.Module):
    """
    3D U-Net for volumetric segmentation.
    Input shape : (N, 4, D, H, W)
    Output shape: (N, 1, D, H, W)
    """
    def __init__(self, in_channels=4):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock3D(in_channels, 64)
        self.enc2 = ConvBlock3D(64, 128)
        self.enc3 = ConvBlock3D(128, 256)
        self.enc4 = ConvBlock3D(256, 512)
        self.pool = nn.MaxPool3d(2)

        # Bridge
        self.bridge = ConvBlock3D(512, 1024)

        # Decoder
        self.up6  = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.dec6 = ConvBlock3D(1024, 512)

        self.up7  = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec7 = ConvBlock3D(512, 256)

        self.up8  = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec8 = ConvBlock3D(256, 128)

        self.up9  = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec9 = ConvBlock3D(128, 64)

        self.out_conv = nn.Conv3d(64, 1, kernel_size=1)

    def forward(self, x):
        c1 = self.enc1(x);  p1 = self.pool(c1)
        c2 = self.enc2(p1); p2 = self.pool(c2)
        c3 = self.enc3(p2); p3 = self.pool(c3)
        c4 = self.enc4(p3); p4 = self.pool(c4)

        c5 = self.bridge(p4)

        u6 = self.up6(c5);  u6 = torch.cat([u6, c4], dim=1); c6 = self.dec6(u6)
        u7 = self.up7(c6);  u7 = torch.cat([u7, c3], dim=1); c7 = self.dec7(u7)
        u8 = self.up8(c7);  u8 = torch.cat([u8, c2], dim=1); c8 = self.dec8(u8)
        u9 = self.up9(c8);  u9 = torch.cat([u9, c1], dim=1); c9 = self.dec9(u9)

        return torch.sigmoid(self.out_conv(c9))

# Factory

def get_model(model_type='2d'):
    """Return the appropriate model and move it to the available device."""
    if model_type == '2d':
        return UNet2D(in_channels=4)
    elif model_type == '3d':
        return UNet3D(in_channels=4)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")