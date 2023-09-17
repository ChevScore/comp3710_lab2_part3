from unet_model_components import *
import torch.nn as nn
import torch
import torchvision.transforms as transforms

class UNet(nn.Module):
    def __init__(self) -> None:
        super(UNet, self).__init__()
        
        self.in_conv = UNetInConv(3, 32)
        
        self.down1 = UNetDown(32, 64)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        
        self.up1 = UNetUp(512, 256)
        self.up2 = UNetUp(256, 128)
        self.up3 = UNetUp(128, 64)
        self.up4 = UNetUp(64, 32)
        
        self.out_conv = UNetOutConv(32, 1)
        
    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        
        x = torch.cat([x, x4], dim=1)
        x = self.up1(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up4(x)
        
        x = self.out_conv(x)
        return x