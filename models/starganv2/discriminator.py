# From https://github.com/yunjey/stargan-v2-demo/tree/master/core

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResBlk(nn.Module):
    """Preactivation residual block."""
    def __init__(self, dim_in, dim_out, downsample=True, is_first=False):
        super(ResBlk, self).__init__()
        if is_first:
            self.conv1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0)   # 1x1 conv for the first layer
        else:
            self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.actv = nn.LeakyReLU(0.2)
        self.downsample = downsample
        self.is_first = is_first
        self.learned_sc = (dim_in != dim_out)   # learnable shortcut
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if not self.is_first:
            x = self.actv(x)
        x = self.conv1(x)
        x = self.actv(x)
        x = self.conv2(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2) * 2
        return x
    
    def forward(self, x):
        return self._residual(x) # +self._shortcut(x)


class Discriminator(nn.Module):
    """Discriminator: (image x, domain y) -> (logit out)."""
    def __init__(self, image_size=256, num_domains=2, max_conv_dim=1024):
        super(Discriminator, self).__init__()
        
        dim_in = 2**13 // image_size
        blocks = []
        blocks += [ResBlk(3, dim_in, downsample=True, is_first=True)]
        
        repeat_num = int(np.log2(image_size)) - 2
        for _ in range(1, repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out
        
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        self.main1 = nn.Sequential(*blocks)
        
        blocks = []
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        """
        Inputs:
            - x: images of shape (batch, 3, image_size, image_size).
            - y: domain labels of shape (batch).
        Output:
            - out: logits of shape (batch).
        """
        out = self.main1(x)
        #print('D_out', torch.mean(torch.var(out, dim=0, unbiased=False)))
        out = self.main(out)
        
        out = out.view(out.size(0), -1)                          # (batch, num_domains)
        
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]                                         # (batch)
        return out