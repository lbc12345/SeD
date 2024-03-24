import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import functools
import math
import clip
from clip.model import ModifiedResNet
from .module_attention import ModifiedSpatialTransformer
    

class CLIP_Semantic_extractor(ModifiedResNet):
    def __init__(self, layers=(3, 4, 6, 3), pretrained=True, path=None, output_dim=1024, heads=32):
        super(CLIP_Semantic_extractor, self).__init__(layers=layers, output_dim=output_dim, heads=heads)

        ckpt = 'RN50' if path is None else path

        if pretrained:
            model, _ = clip.load(ckpt, device='cpu')
        
        self.load_state_dict(model.visual.state_dict())
        self.register_buffer(
            'mean',
            torch.Tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.Tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        )
        self.requires_grad_(False)

        del model

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x
        
        x = (x - self.mean) / self.std
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


class SeD_P(nn.Module):
    def __init__(self, input_nc, ndf=64, semantic_dim=1024, semantic_size=16, use_bias=True, nheads=1, dhead=64):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()

        kw = 4
        padw = 1        
        # ss = [128, 64, 32, 31, 30]  # PatchGAN's spatial size
        # cs = [64, 128, 256, 512, 1]  # PatchGAN's channel size

        norm = spectral_norm

        self.lrelu = nn.LeakyReLU(0.2, True)
        self.conv_first = nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)

        self.conv1 = norm(nn.Conv2d(ndf * 1, ndf * 2, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        upscale = math.ceil(64 / semantic_size)
        self.att1 = ModifiedSpatialTransformer(in_channels=semantic_dim, n_heads=nheads, d_head=dhead, context_dim=128, up_factor=upscale)
        
        ex_ndf = int(semantic_dim / upscale**2)
        self.conv11 = norm(nn.Conv2d(ndf * 2 + ex_ndf, ndf * 2, kernel_size=3, stride=1, padding=padw, bias=use_bias))
        
        self.conv2 = norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        upscale = math.ceil(32 / semantic_size)
        self.att2 = ModifiedSpatialTransformer(in_channels=semantic_dim, n_heads=nheads, d_head=dhead, context_dim=256, up_factor=upscale)
        
        ex_ndf = int(semantic_dim / upscale**2)
        self.conv21 = norm(nn.Conv2d(ndf * 4 + ex_ndf, ndf * 4, kernel_size=3, stride=1, padding=padw, bias=use_bias))
        
        self.conv3 = norm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=kw, stride=1, padding=padw, bias=use_bias))
        upscale = math.ceil(31 / semantic_size)
        self.att3 = ModifiedSpatialTransformer(in_channels=semantic_dim, n_heads=nheads, d_head=dhead, context_dim=512, up_factor=upscale, is_last=True)
        
        ex_ndf = int(semantic_dim / upscale**2)
        self.conv31 = norm(nn.Conv2d(ndf * 8 + ex_ndf, ndf * 8, kernel_size=3, stride=1, padding=padw, bias=use_bias))
        
        self.conv_last = nn.Conv2d(ndf * 8, 1, kernel_size=kw, stride=1, padding=padw)
        
        init_weights(self, init_type='normal')

    def forward(self, input, semantic):
        """Standard forward."""
        input = self.conv_first(input)
        input = self.lrelu(input)
        
        input = self.conv1(input)
        se = self.att1(semantic, input)
        input = self.lrelu(self.conv11(torch.cat([input, se], dim=1)))
        
        input = self.conv2(input)
        se = self.att2(semantic, input)
        input = self.lrelu(self.conv21(torch.cat([input, se], dim=1)))
        
        input = self.conv3(input)
        se = self.att3(semantic, input)
        input = self.lrelu(self.conv31(torch.cat([input, se], dim=1)))
            
        input = self.conv_last(input)
        return input


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
