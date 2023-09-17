import torch
import torch.nn as nn
from torchvision import models

class UNetInConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.in_conv(x)
    
class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            UNetInConv(in_channels, out_channels)
        )
        
    def forward(self, x):
        return self.down_conv(x)
    
class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2),
            UNetInConv(in_channels, out_channels)
        )
        
    def forward(self, x):
        return self.up_conv(x)
    
class UNetOutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_conv = nn.Sequential(
            UNetInConv(in_channels, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )
        
    def forward(self, x):
        return self.out_conv(x)