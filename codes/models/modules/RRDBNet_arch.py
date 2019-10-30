import functools
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import models.modules.module_util as mutil
from anode.anode.odeblock import make_odeblock
from anode.models.sr_trunk import SRTrunk
from torchdiffeq._impl.conv import ODEBlock, ODEfunc, ConcatConv2d
from torchdiffeq._impl.augmented_conv import ODEBlock as AugBlock, ConvODEFunc as AugFunc


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, sb=5, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.convs = torch.nn.ModuleList([nn.Conv2d(nf+i*gc, gc, 3, 1, 1, bias=bias) for i in range(sb-1)])
        self.final_conv = nn.Conv2d(nf + (sb-1) * gc, nf, 3, 1, 1, bias=bias, padding_mode="reflect")
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        mutil.initialize_weights([*self.convs, self.final_conv], 0.1)

    def forward(self, x):
        x_original = x
        for conv in self.convs:
            x = torch.cat((x, self.lrelu(conv(x))), 1)
        x = self.final_conv(x)
        return x * 0.2 + x_original


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32, sb=5):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc, sb)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc, sb)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc, sb)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class TimeResidualDenseBlock5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(TimeResidualDenseBlock5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = ConcatConv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = ConcatConv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = ConcatConv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = ConcatConv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = ConcatConv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, t, x):
        x1 = self.lrelu(self.conv1(t, x))
        x2 = self.lrelu(self.conv2(t, torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(t, torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(t, torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(t, torch.cat((x, x1, x2, x3, x4), 1))
        return x5



class RRDBODEfunc(nn.Module):

    def __init__(self, nf=64, gc=32, nb=3, bias=True, normalization=False, time_dependent=False):
        super(RRDBODEfunc, self).__init__()
        self.normalization = normalization
        self.nb = nb
        self.time_dependent = time_dependent
        assert nb > 0
        if time_dependent:
            self.convs = nn.ModuleList([TimeResidualDenseBlock5C(nf, gc, bias=bias) for _ in range(nb)])
        else:
            self.convs = nn.ModuleList([ResidualDenseBlock_5C(nf, gc, bias=bias) for _ in range(nb)])
        if self.normalization:
            self.norms = nn.ModuleList([nn.GroupNorm(nf, nf) for _ in range(nb)])
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = x
        for i in range(self.nb):
            if self.time_dependent:
                out = self.convs[i](t, out)
            else:
                out = self.convs[i](out)
            if self.normalization:
                out = self.norms[i](out)
        return out


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, differential=None, time_dependent=False, adjoint=False, sb=5):
        super(RRDBNet, self).__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if differential == "checkpointed":
            self.conv_trunk = SRTrunk(nf, nb, make_odeblock(5, 'RK4'))
            mutil.initialize_weights(self.conv_trunk.odefunc.convs)
        elif differential == "standard":
            self.conv_trunk = ODEBlock(ODEfunc(nf, nb=nb, normalization=False, time_dependent=time_dependent), adjoint=adjoint)
            mutil.initialize_weights(self.conv_trunk.odefunc.convs)
        elif differential == "sequential":
            self.conv_trunk = nn.Sequential(*[ODEBlock(ODEfunc(nf, nb=sb, normalization=False, time_dependent=time_dependent), adjoint=adjoint) for _ in range(nb)])
            for block in self.conv_trunk:
                mutil.initialize_weights(block.odefunc.convs)
        elif differential == "augmented":
            augment_dim = nf//4
            self.conv_trunk = AugBlock(AugFunc(nf=nf, nb=nb, augment_dim=augment_dim, time_dependent=time_dependent), adjoint=adjoint, is_conv=True)
            self.trunk_conv = nn.Conv2d(nf+augment_dim, nf, 3, 1, 1, bias=True)
            mutil.initialize_weights(self.conv_trunk.odefunc.convs)
        elif differential is None or differential == "nodiff":
            RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc, sb=sb)
            self.conv_trunk = mutil.make_layer(RRDB_block_f, nb)
        else:
            raise NotImplementedError("unrecognized differential system passed")
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, interpolation_start=True):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.conv_trunk(fea))
        # trunk = fea + trunk
        trunk = self.lrelu(self.upconv1(F.interpolate(trunk, scale_factor=2, mode='bicubic')))
        trunk = self.lrelu(self.upconv2(F.interpolate(trunk, scale_factor=2, mode='bicubic')))
        out = self.conv_last(self.lrelu(self.HRconv(trunk)))

        if interpolation_start:
            interpolated = F.interpolate(x, scale_factor=4, mode='bicubic')
            out = out + interpolated

        return out