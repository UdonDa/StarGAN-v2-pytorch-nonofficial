import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out, affine=True, track_running_stats=True):
        super(ResidualBlock, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        layers = []
        layers.append(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True))
        layers.append(nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=True))
        layers.append(nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

        if dim_in != dim_out:
            self.conv_up = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, bias=False)
        
        self.main = nn.Sequential(*layers)


    def forward(self, x):
        if self.dim_in != self.dim_out:
            return self.conv_up(x) + self.main(x)
        else:
            return x + self.main(x)

def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            bs = adain_params.size(0)
            mean = torch.mean(adain_params, 1, keepdim=True) # -> [bs, 1]
            std = torch.std(adain_params, 1, keepdim=True) # -> [bs, 1]

            mean = mean.repeat(1, m.num_features)
            std = std.repeat(1, m.num_features)
            m.bias = mean
            m.weight = std

def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            num_adain_params += 2*m.num_features
    return num_adain_params

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x): # -> [1, 512, 16, 16])
        assert self.weight is not None and \
               self.bias is not None, "Please assign AdaIN weight first"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class Decoder(nn.Module):
    def __init__(self, in_channel=512, out_channel=3, in_feature=16, middle_feature=512, num_domain=3, num_layers=6):
        super(Decoder, self).__init__()

        layers = []

        for i in range(2):
            layers.append(
                nn.Sequential(
                    ResidualBlock(in_channel, in_channel, affine=True, track_running_stats=True),
                    AdaptiveInstanceNorm2d(in_channel)
                )
            )
        for i in range(4):
            layers.append(
                nn.Sequential(
                    ResidualBlock(in_channel, in_channel//2, affine=True, track_running_stats=True),
                    AdaptiveInstanceNorm2d(in_channel//2),
                    nn.Upsample(scale_factor=2)
                )
            )
            in_channel //= 2
        layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1))

        self.block = nn.Sequential(*layers)

        self.mapping_net = MappingNetwork(
            in_feature=in_feature, middle_feature=middle_feature,
            num_domain=num_domain, num_layers=num_layers
        )
        self.mapping_net.apply(weights_init_adain_linear)

    
    def forward(self, x, code=None):
        if code is not None:
            code = self.mapping_net(code)
        assign_adain_params(code, self.block)
        out = self.block(x)
        return out

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, in_channel=3, conv_dim=64, c_dim=5, repeat_num=4):
        super(Generator, self).__init__()
        encoder = []
        encoder.append(nn.Conv2d(in_channel, conv_dim // 2, kernel_size=1, stride=1, bias=True))

        for i in range(repeat_num):
            encoder.append(
                nn.Sequential(
                    ResidualBlock(conv_dim//2, conv_dim, affine=True, track_running_stats=True),
                    nn.AvgPool2d(3, stride=2, padding=1)
                )
            )
            if i != (repeat_num-1):
                conv_dim *= 2
        encoder.append(
            nn.Sequential(
                ResidualBlock(conv_dim, conv_dim, affine=True, track_running_stats=True),
                ResidualBlock(conv_dim, conv_dim, affine=True, track_running_stats=True)
            )
        )

        self.encoder = nn.Sequential(*encoder)

        self.decoder = Decoder(in_channel=conv_dim, out_channel=in_channel, in_feature=16, middle_feature=512, num_domain=3, num_layers=6)

        self.encoder.apply(weights_init_func)
        self.decoder.block.apply(weights_init_func)

    def forward(self, x, code=None):
        ### Down sampling
        f = self.encoder(x)#; print("f1:", f.size()) # -> [1, 512, 16, 16]
        out = self.decoder(f, code)
        return out


class MappingNetwork(nn.Module):
    def __init__(self, in_feature=16, middle_feature=512, num_domain=3, num_layers=6):
        super(MappingNetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(in_feature, middle_feature))
        for _ in range(num_layers):
            layers.append(
                nn.Sequential(
                    nn.Linear(middle_feature, middle_feature),
                    nn.ReLU()
                )
            )
        layers.append(nn.Linear(middle_feature, 64 * num_domain))
        self.main = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.main(z)
                
        
class Discriminator(nn.Module):
    def __init__(self, in_channel=3, conv_dim=16, num_domain=3, num_layers=6, D=64):
        super(Discriminator, self).__init__()
        """
        conv_dim = 16, D = 64 => style encoder
        conv_dim = 32, D = 1  => style encoder
        """
        layers = []
        layers.append(nn.Conv2d(in_channel, conv_dim, kernel_size=1, stride=1, bias=True))
        for i in range(num_layers):
            if i < (num_layers-1):
                layers.append(
                    nn.Sequential(
                        ResidualBlock(conv_dim, conv_dim*2, affine=True, track_running_stats=True),
                        nn.AvgPool2d(3, stride=2, padding=1)
                    )
                )
                conv_dim *= 2
            else:
                layers.append(
                    nn.Sequential(
                        ResidualBlock(conv_dim, conv_dim, affine=True, track_running_stats=True),
                        nn.AvgPool2d(3, stride=2, padding=1)
                    )
                )

        layers.append(
            nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(conv_dim, conv_dim, kernel_size=4, stride=4, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )
        )
        self.main = nn.Sequential(*layers)
        self.classifier = nn.Linear(conv_dim, num_domain * D)

        self.main.apply(weights_init_func)
        self.classifier.apply(weights_init_func)

    
    def forward(self, z):
        feature = self.main(z)#; print(feature.size())
        feature = feature.view(feature.size(0), -1)
        pred = self.classifier(feature)#; print(pred.size())
        return pred



def weights_init_func(m, init_type='kaiming'):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv')!=-1 or classname.find('Linear')!=-1):
        if init_type == 'normal':
            nn.init.normal_(m.weight.data, 0.0, gain)
        elif init_type == 'xavier':
            nn.init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(m.weight.data, gain=gain)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def weights_init_adain_linear(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and classname.find('Linear')!=-1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.bias.data.fill_(1.)

if __name__ == "__main__":
    z = torch.randn(4,3,256,256).cuda()
    G = Generator().cuda()
    styleEncoder = Discriminator(in_channel=3, conv_dim=16, num_domain=3, num_layers=6, D=64).cuda()
    discriminator = Discriminator(in_channel=3, conv_dim=32, num_domain=3, num_layers=6, D=1).cuda()
    mappingNetwork = MappingNetwork().cuda()

    code = torch.randn(4, 16).cuda()
    # y = mappingNetwork(style_code).cuda()
    # print("mappingNetwork:", y.size())

    # y = styleEncoder(z)
    # print("styleEncoder:", y.size())

    # y = discriminator(z)
    # print("discriminator:", y.size())
    # for m in G.modules():
    #     if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
    #         print(m.__class__.__name__)

    # y = G(z, code)
    # print(y.size())

    
    
    
    # y = G(z)
