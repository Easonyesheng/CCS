import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np 

# from models.unet_parts_simplify import *
from models.unet_parts import *


class UNet_half_4_dc(nn.Module):
    def __init__(self, order, device, n_channels=1, n_classes=1, bilinear=True, img_size = 480, model='poly'):
        super(UNet_half_4_dc, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.device = device
        self.img_size = img_size
        self.order = order # the order of polynomial model
        self.outsize = (order+2)*(order+1) if model=='poly' else order

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)

        # for p in self.parameters():
        #     p.requires_grad=False
        #====output is Nx1x480x480, img_size = 480
        # Then perform self.mul_mesh_grid
        # After that get into the calculate part

        self.cal_part = nn.Sequential(
            nn.Conv2d(256, 1024, kernel_size=img_size // 16, padding=0),
            # nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1, padding=0),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1, padding=0),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=1, padding=0),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.outsize, kernel_size=1, padding=0)
        ) # batch x (order+2)*(order+1) x 1 x 1


    # ori
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
      
        calcul = self.cal_part(x5)

        out = {
            'parameters': calcul
        }

        return out