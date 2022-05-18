""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch
from torch import nn

from models.unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.sig = nn.Sigmoid()
        # self.outf = OutConv(n_channels, n_channels)


        # self.down23_8 = Down(128,256)
        # self.down23_6 = Down(128,128)
        # self.down34 = Down(256,256)
        # self.up43 = Up(512,128, bilinear)
        # self.up32 = Up(256,64, bilinear)
        # self.up21 = Up(128,64, bilinear)
        
    # ori
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # logits = self.sig(logits)
        # logits = self.outf(logits)
        
        return logits

    # # 6 layers
    # def forward(self, x):
    #     x1 = self.inc(x)
    #     x2 = self.down1(x1)
    #     x3 = self.down23_6(x2)
    #     x = self.up32(x3,x2)
    #     x = self.up21(x, x1)
    #     logits = self.outc(x)
    #     return logits
    
    # # 8 layers
    # def forward(self, x):
    #     x1 = self.inc(x)
    #     x2 = self.down1(x1)
    #     x3 = self.down23_8(x2)
    #     x4 = self.down34(x3)
    #     x = self.up43(x4,x3)
    #     x = self.up32(x,x2)
    #     x = self.up21(x, x1)
    #     logits = self.outc(x)
    #     return logits
    
