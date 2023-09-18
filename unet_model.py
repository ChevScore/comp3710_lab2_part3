from unet_model_components import *
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F


class UNet(nn.Module):
    """
    UNet model architecture as defined in:
    Reference: https://camo.githubusercontent.com/f3686ec6ba3a790633d1806ec5eb7ffcb4be041eeec8c508eb0200d140450810/68747470733a2f2f64726976652e676f6f676c652e636f6d2f75633f6578706f72743d766965772669643d316677615f4977733668306e6b647075527249484b3345616c38456e6d4a5f6238
    """
    def __init__(self) -> None:
        super(UNet, self).__init__()
        
        self.in_conv = UNetInConv(1, 32)
        
        self.down1 = UNetDown(32, 64)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        
        self.up1 = UNetUp(512, 256)
        self.up2 = UNetUp(256, 128)
        self.up3 = UNetUp(128, 64)
        
        self.out_conv = UNetOutConv(64, 4)
        
    def forward(self, x):
        """
        Forward pass of the UNet model as per above architecture reference
        """
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        
        # x = torch.cat([x, x4])
        x = self.up1(x)
        # x = torch.cat([x, x3])
        x = self.up2(x)
        # x = torch.cat([x, x2])
        x = self.up3(x)
        # x = torch.cat([x, x1])
        
        x = self.out_conv(x)
        x = F.softmax(x, dim=1)

        print(x.shape)
        return x