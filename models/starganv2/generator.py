# From https://github.com/yunjey/stargan-v2-demo/tree/master/core
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HighPass(nn.Module):
    def __init__(self, division_factor):
        super(HighPass, self).__init__()
        self.filter = torch.tensor([[-1, -1, -1],
                                    [-1, 8., -1],
                                    [-1, -1, -1]]).cuda() / division_factor
    
    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1)) # depth-wise conv


class FRN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(FRN, self).__init__()
        self.tau = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.eps = eps

    def forward(self, x):
        x = x * torch.rsqrt(torch.mean(x**2, dim=[2, 3], keepdim=True) + self.eps)
        return torch.max(self.gamma * x + self.beta, self.tau)


class AdaFRN(nn.Module):
    def __init__(self, style_dim, num_features, eps=1e-5):
        super(AdaFRN, self).__init__()
        self.tau = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.fc = nn.Linear(style_dim, num_features*2)
        self.eps = eps

    def forward(self, x, s):
        # filter response normalization
        x = x * torch.rsqrt(torch.mean(x**2, dim=[2, 3], keepdim=True) + self.eps)

        # adaptive gamma and beta prediction
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)

        out = (1 + gamma) * x# + beta
        return torch.max(out, self.tau)


class AdainResBlk(nn.Module):
    """Preactivation residual block with AdaFRN."""
    def __init__(self, dim_in, dim_out, style_dim=64, upsample=False):
        super(AdainResBlk, self).__init__()
        self.upsample = upsample
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaFRN(style_dim, dim_in)
        self.norm2 = AdaFRN(style_dim, dim_out)
        if self.upsample:
            self.conv1x1 = nn.ConvTranspose2d(dim_in, dim_out, 4, 2, 1)

    def _shortcut(self, x):
        if self.upsample:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, y):
        x = self.norm1(x, y)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, y)
        x = self.conv2(x)
        return x

    def forward(self, x, y):
        x = self._residual(x, y) #+ 0.1 * self._shortcut(x)
        #print(torch.mean(torch.var(x, dim=0, unbiased=False)))
        return x


class ResBlk(nn.Module):
    """Preactivation residual block with filter response norm."""
    def __init__(self, dim_in, dim_out, style_dim=64, downsample=False):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.downsample = downsample
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.norm1 = FRN(dim_in)
        self.norm2 = FRN(dim_in)
        if self.downsample:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 4, 2, 1)

    def _shortcut(self, x):
        if self.downsample:
            x = self.conv1x1(x)
        return x

    def _residual(self, x):
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        x = self.conv2(x)
        return x

    def forward(self, x):
        #x = self._residual(x)
        x = self._residual(x) + self._shortcut(x)
        return x


class Generator(nn.Module):
    """Generator: (image x, style s) -> (image out)."""
    def __init__(self, image_size=256, style_dim=64, division_factor=64, sigma_factor=3):
        super(Generator, self).__init__()
        
        conv_dim = 2**13 // image_size
        self.from_rgb = nn.Conv2d(3, conv_dim, 1, 1, 0)
        self.down1 = ResBlk(conv_dim, conv_dim*2, style_dim, downsample=True)
        self.down2 = ResBlk(conv_dim*2, conv_dim*4, style_dim, downsample=True)
        self.down3 = ResBlk(conv_dim*4, conv_dim*6, style_dim, downsample=True)
        self.down4 = ResBlk(conv_dim*6, conv_dim*8, style_dim, downsample=True)
        self.down5 = ResBlk(conv_dim*8, conv_dim*16, style_dim, downsample=True)

        self.block1 = ResBlk(conv_dim*16, conv_dim*16, style_dim)
        self.block2 = ResBlk(conv_dim*16, conv_dim*16, style_dim)
        self.block3 = AdainResBlk(conv_dim*16, conv_dim*16, style_dim)
        self.block4 = AdainResBlk(conv_dim*16, conv_dim*16, style_dim)

        self.up1 = AdainResBlk(conv_dim*16, conv_dim*8, style_dim, upsample=True)
        self.up2 = AdainResBlk(conv_dim*8, conv_dim*6, style_dim, upsample=True)
        self.up3 = AdainResBlk(conv_dim*6, conv_dim*4, style_dim, upsample=True)
        self.up4 = AdainResBlk(conv_dim*4, conv_dim*2, style_dim, upsample=True)
        self.up5 = AdainResBlk(conv_dim*2, conv_dim, style_dim, upsample=True)
        self.to_rgb = nn.Sequential(
            FRN(conv_dim),
            nn.Conv2d(conv_dim, 3, 1, 1, 0))

        self.hpf = HighPass(division_factor)

    def forward(self, x, s):
        """
        Inputs:
            - x: input images of shape (batch, num_channels, image_size, image_size).
            - s: style vectors of shape (batch, style_dim).
        Output:
            - out: generated images of shape (batch, num_channels, image_size, image_size).
        """

        # downsample
        h = self.from_rgb(x)
        h_128 = self.down1(h)
        h_64 = self.down2(h_128)
        h_32 = self.down3(h_64)
        h = self.down4(h_32)
        h = self.down5(h)

        # bottleneck
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h, s)
        h = self.block4(h, s)

        # upsample
        h = self.up1(h, s)
        h = self.up2(h, s)
        h = self.up3(h, s)
        h = self.up4(h, s)
        h = self.up5(h, s)
        out = self.to_rgb(h)
        return out