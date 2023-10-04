""" Full assembly of the parts to form the complete network """
import torch

from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, out_channels, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.out_channels = out_channels
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
        self.outc = OutConv(64, out_channels)

    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        pred_error = self.outc(x)
        pred_ref = pred_error + input
        return pred_ref, pred_error


if __name__ == '__main__':
    input = torch.randn((1, 3, 112, 112))
    model = UNet(n_channels=3, out_channels=3, bilinear=True)
    input = input.cuda()
    model = model.cuda()
    output = model(input)
    print(1)