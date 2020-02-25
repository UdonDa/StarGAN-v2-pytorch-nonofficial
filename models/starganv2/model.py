import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out, downsample=True, is_first=False):
        super(ResidualBlock, self).__init__()
        layers = []
        self.downsample = downsample

        if not is_first:
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True))
        else:
            layers.append(nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=True))


        layers.append(nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=True))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=True))

        self.conv_up = None
        if dim_in != dim_out:
            self.conv_up = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        dx = self.main(x)
        if self.conv_up is not None:
            x = self.conv_up(x)
        out = x + dx

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        return out

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

class AdaINResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64):
        super(AdaINResidualBlock, self).__init__()
        self.upsample = (dim_in != dim_out)
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaFRN(style_dim, dim_in)
        self.norm2 = AdaFRN(style_dim, dim_out)
        self.conv1x1 = None
        if self.upsample:
            self.conv1x1 = nn.ConvTranspose2d(dim_in, dim_out, 4, 2, 1)


    def residual(self, x, y):
        x = self.norm1(x, y)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, y)
        x = self.conv2(x)
        return x

    def forward(self, x, y):
        dx = self.residual(x, y)
        # if self.conv1x1 is not None:
        #     x = self.conv1x1(x)
        return dx # + x

class GResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(GResidualBlock, self).__init__()
        layers = []
        layers.append(nn.Sequential(
            FRN(dim_in),
            nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1, padding=1),
            FRN(dim_in),
        ))
        
        self.conv1x1 = None
        if dim_in != dim_out:
            layers.append(nn.AvgPool2d(3, stride=2, padding=1))
            self.conv1x1 = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0),
                nn.AvgPool2d(3, stride=2, padding=1)
            )

        layers.append(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1))
        
        self.main = nn.Sequential(*layers)

    def forward(self, x):

        dx = self.main(x)
        if self.conv1x1 is not None:
            x = self.conv1x1(x)
        return dx + x



class Generator(nn.Module):
    def __init__(self, in_dim=3, image_size=256, style_dim=64):
        super(Generator, self).__init__()
        
        conv_dim = 2**13 // image_size
        self.down = nn.Sequential(
            nn.Conv2d(in_dim, conv_dim, 1, 1, 0),
            GResidualBlock(conv_dim, conv_dim*2),
            GResidualBlock(conv_dim*2, conv_dim*4),
            GResidualBlock(conv_dim*4, conv_dim*6),
            GResidualBlock(conv_dim*6, conv_dim*8),
            GResidualBlock(conv_dim*8, conv_dim*16),
            GResidualBlock(conv_dim*16, conv_dim*16),
            GResidualBlock(conv_dim*16, conv_dim*16)
        )

        self.adain_res_1 = AdaINResidualBlock(conv_dim*16, conv_dim*16, style_dim)
        self.adain_res_2 = AdaINResidualBlock(conv_dim*16, conv_dim*16, style_dim)

        self.up1 = AdaINResidualBlock(conv_dim*16, conv_dim*8, style_dim)
        self.up2 = AdaINResidualBlock(conv_dim*8, conv_dim*6, style_dim)
        self.up3 = AdaINResidualBlock(conv_dim*6, conv_dim*4, style_dim)
        self.up4 = AdaINResidualBlock(conv_dim*4, conv_dim*2, style_dim)
        self.up5 = AdaINResidualBlock(conv_dim*2, conv_dim, style_dim)
        self.conv_final = nn.Sequential(
            FRN(conv_dim),
            nn.Conv2d(conv_dim, 3, 1, 1, 0))

    def forward(self, image, style_code):
        """
        Arguments:
            x {torch.tensor} -- [bs, in_dim, image_size, image_size]
            s {torch.tensor} -- [bs, style_dim]
        
        Returns:
            fake {torch.tensor} -- [bs, in_dim, image_size, image_size]
        """
        # downsample
        f = self.down(image)

        # bottleneck
        f = self.adain_res_1(f, style_code)
        f = self.adain_res_2(f, style_code)

        # upsample
        f = self.up1(f, style_code)
        f = self.up2(f, style_code)
        f = self.up3(f, style_code)
        f = self.up4(f, style_code)
        f = self.up5(f, style_code)
        out = self.conv_final(f)
        return out
        # return torch.tanh(out)

class MappingNetwork(nn.Module):
    def __init__(self, in_dim=64, style_dim=64, hidden_dim=512, num_domain=3, num_layers=6, pixel_norm=False):
        super(MappingNetwork, self).__init__()
        self.num_domain = num_domain
        self.pixel_norm = pixel_norm
        
        layers = [nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )]
        for _ in range(num_layers):
            layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ))
        self.main = nn.Sequential(*layers)

        self.classifier = nn.ModuleList([nn.Linear(hidden_dim, style_dim) for _ in range(num_domain)])

    def forward(self, code, condition):
        """
        Arguments:
            code {torch.tensor} -- [bs, in_dim]
            condition {torch.tensor} -- [bs]
        
        Returns:
            style_code {torch.tensor} -- [bs, style_dim]
        """        
        if self.pixel_norm:
            z = z / torch.norm(z, p=2, dim=1, keepdim=True)
            z = z / (torch.sqrt(torch.mean(z**2, dim=1, keepdim=True)) + 1e-8)
        feature = self.main(code)

        out = [classifier(feature) for classifier in self.classifier]

        out = torch.stack(out, dim=1)
        index = torch.LongTensor(range(condition.size(0))).to(condition.device)
        style_code = out[index, condition]
        return style_code

class StyleEncoder(nn.Module):
    def __init__(self, in_channel=3, image_size=256, num_domain=3, D=64, max_dim=512):
        super(StyleEncoder, self).__init__()
        """conv_dim = 16, D = 64 => style encoder"""

        dim_in = 2**13 // image_size
        layers = []
        layers.append(ResidualBlock(in_channel, dim_in, downsample=True, is_first=True))

        num_layers = int(np.log2(image_size)) - 2

        for _ in range(1, num_layers):
            dim_out = min(max_dim, dim_in*2)
            layers.append(ResidualBlock(dim_in, dim_out, downsample=True, is_first=False))
            dim_in = dim_out

        layers.append(
            nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(dim_out, dim_out, kernel_size=4, stride=1, padding=0),
                nn.LeakyReLU(0.2, inplace=True),
            )
        )
        self.main = nn.Sequential(*layers)

        self.classifier = nn.ModuleList([
            nn.Linear(dim_out, D) for _ in range(num_domain)
        ])

    
    def forward(self, image, condition):
        """
        Arguments:
            image {torch.tensor} -- [bs, 3, image_size, image_size]
            condition {torch.tensor} -- [bs]
        
        Returns:
            style {torch.tensor} -- [bs, D]
        """
        feature = self.main(image)#; print(feature.size())
        feature = feature.view(feature.size(0), -1)
        pred_list = [linear(feature) for linear in self.classifier]

        pred = torch.stack(pred_list, dim=1)
        index = torch.LongTensor(range(condition.size(0))).to(condition.device)
        style = pred[index, condition]

        return style

class Discriminator(nn.Module):
    def __init__(self, in_channel=3, image_size=256, num_domain=3, D=1, max_dim=1024):
        super(Discriminator, self).__init__()
        """conv_dim = 16, D = 64 => style encoder"""

        dim_in = 2**13 // image_size
        layers = []
        layers.append(ResidualBlock(in_channel, dim_in, downsample=True, is_first=True))

        num_layers = int(np.log2(image_size)) - 2

        for _ in range(1, num_layers):
            dim_out = min(max_dim, dim_in*2)
            layers.append(ResidualBlock(dim_in, dim_out, downsample=True, is_first=False))
            dim_in = dim_out

        layers.append(
            nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Conv2d(dim_out, dim_out, kernel_size=4, stride=1, padding=0),
                nn.LeakyReLU(0.2),
            )
        )
        self.main = nn.Sequential(*layers)

        self.classifier = nn.Conv2d(dim_out, num_domain, 1, 1, 0)
    
    def forward(self, image, condition):
        """
        Arguments:
            image {torch.tensor} -- [bs, 3, image_size, image_size]
            condition {torch.tensor} -- [bs]
        
        Returns:
            out {torch.tensor} -- [bs]
        """
        feature = self.main(image)
        logits = self.classifier(feature)

        logits = logits.view(logits.size(0), -1) # [bs, num_domain]

        index = torch.LongTensor(range(condition.size(0))).to(condition.device)
        out = logits[index, condition]

        return out

if __name__ == "__main__":
    bs = 8
    H, W = 256, 256
    num_domain = 3

    x = torch.randn(bs, 3, H, W).cuda()
    y = torch.randperm(bs).cuda() % num_domain
    z = torch.randn(bs, 16)

    model = Discriminator(in_channel=3, image_size=H, num_domain=num_domain, D=1, max_dim=1024).cuda()
    pred = model(x, y)
    print("(D) pred:", pred.shape) # -> [domain]
    
    model = StyleEncoder(in_channel=3, image_size=H, num_domain=num_domain, D=64, max_dim=512).cuda()
    style_code = model(x, y)
    print("(Enc) pred:", style_code.shape)

    model = MappingNetwork(in_dim=64, style_dim=64, hidden_dim=512, num_domain=num_domain, num_layers=6, pixel_norm=False)
    pred = model(z, y)
    print("(MapNet) pred:", pred.size())

    model = Generator(in_dim=3, image_size=256, style_dim=64).cuda()
    pred = model(x, style_code)
    print("(G) pred:", pred.shape)