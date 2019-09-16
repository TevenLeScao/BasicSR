import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
from anode.anode.odeblock import make_odeblock
from anode.models.sr_trunk import SRTrunk
from torchdiffeq._impl.conv import ODEBlock, ODEfunc


class MSRResNet(nn.Module):
    ''' modified SRResNet'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4, differential=False):
        super(MSRResNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        if differential == "checkpointed":
            self.conv_trunk = SRTrunk(nf, nb, make_odeblock(2, 'RK4'))
            mutil.initialize_weights(self.conv_trunk.odefunc.convs)
        elif differential == "standard":
            self.conv_trunk = ODEBlock(ODEfunc(nf, nb=nb, normalization=False))
            mutil.initialize_weights(self.conv_trunk.odefunc.convs)
        elif differential is None:
            basic_block = functools.partial(mutil.ResidualBlock_noBN, nf=nf)
            self.conv_trunk = mutil.make_layer(basic_block, nb)
        else:
            raise NotImplementedError("unrecognized differential system passed")

        # upsampling
        if self.upscale == 2:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2d(nf, nf * 9, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

        # initialization
        mutil.initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last], 0.1)
        if self.upscale == 4:
            mutil.initialize_weights(self.upconv2, 0.1)

    def forward(self, x):
        convd = self.conv_first(x)
        fea = self.lrelu(convd)
        out = self.conv_trunk(fea)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.HRconv(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out = base + out
        return out
