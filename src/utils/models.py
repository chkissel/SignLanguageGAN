# -*- coding: utf-8 -*-
"""SignLanguageGAN models

This is a modified and extended version of
https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/pix2pix/models.py
"""
import torch.nn as nn
import torch.nn.functional as F
import torch

import sys


def weights_init_normal(m):
    """ Initialize network weights"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################

class UNetDown(nn.Module):
    """Returns down-sampling convolution"""
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """Returns up-sampling convolution"""
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    """
    U-Net Generator

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        
    Returns:
        Tensor with out_channels X 256 X 256 dimensions.
    """
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


##########################################
#           U-NET Double Encoder
##########################################

class DoubleUNetUp(nn.Module):
    """Returns up-sampling convolution with skip-connections from two encoders"""
    def __init__(self, in_size, out_size, dropout=0.0):
        super(DoubleUNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input, skip_input2):
        x = self.model(x)
        x = torch.cat((x, skip_input, skip_input2), 1)

        return x


class DoubleUNet4Up(nn.Module):
    """Returns up-sampling convolution and merges latent spaces from two encoders"""
    def __init__(self, in_size, out_size, dropout=0.0):
        super(DoubleUNet4Up, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, input, input2, skip_input, skip_input2):
        x = torch.cat((input, input2), 1)
        x = self.model(x)
        x = torch.cat((x, skip_input, skip_input2), 1)

        return x


class DoubleUNet(nn.Module):
    """
    Double U-Net Generator

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        
    Returns:
        Tensor with out_channels X 256 X 256 dimensions.
    """
    def __init__(self, in_channels=6, out_channels=3):
        super(DoubleUNet, self).__init__()
        
        self.down0 = UNetDown(in_channels, 64, normalize=False)
        #self.down0 = UNetDown(3, 64, normalize=False)

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)
        
        self.up1 = DoubleUNet4Up(1024, 512, dropout=0.5)
        self.up2 = DoubleUNetUp(1536, 512, dropout=0.5)
        self.up3 = DoubleUNetUp(1536, 512, dropout=0.5)
        self.up4 = DoubleUNetUp(1536, 512, dropout=0.5)
        self.up5 = DoubleUNetUp(1536, 256)
        self.up6 = DoubleUNetUp(768, 128)
        self.up7 = DoubleUNetUp(384, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(192, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, y, z):
        d1 = self.down0(z)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        d_2_1 = self.down1(y)
        d_2_2 = self.down2(d_2_1)
        d_2_3 = self.down3(d_2_2)
        d_2_4 = self.down4(d_2_3)
        d_2_5 = self.down5(d_2_4)
        d_2_6 = self.down6(d_2_5)
        d_2_7 = self.down7(d_2_6)
        d_2_8 = self.down8(d_2_7)

        u1 = self.up1(d8, d_2_8, d7, d_2_7)
        u2 = self.up2(u1, d6, d_2_6)     
        u3 = self.up3(u2, d5, d_2_5)
        u4 = self.up4(u3, d4, d_2_4)
        u5 = self.up5(u4, d3, d_2_3)
        u6 = self.up6(u5, d2, d_2_2)
        u7 = self.up7(u6, d1, d_2_1)

        return self.final(u7)


##############################
#        Discriminator
##############################

class Discriminator(nn.Module):
    """
    Discriminator

    Args:
        in_channels: Number of input channels.
        
    Returns:
        Tensor with soley one channel.
    """
    def __init__(self, in_channels=15):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            #*discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B, pose, pred_parsing, parsing):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B, pose, pred_parsing, parsing), 1)
        return self.model(img_input)