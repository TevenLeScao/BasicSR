import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
from anode.anode.odeblock import make_odeblock
from anode.models.sr_trunk import SRTrunk
from torchdiffeq._impl.reaction_diffusion import ReactionDiffusion
from torchdiffeq._impl.conv import ODEBlock


class RDNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, differential=None, time_dependent=False):
        super(RDNet, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        if differential == "checkpointed":
            self.conv_trunk = SRTrunk(nf, nb, make_odeblock(5, 'RK4'))
            mutil.initialize_weights(self.conv_trunk.odefunc.convs)
        elif differential == "standard":
            self.conv_trunk = ODEBlock(ODEfunc(nf, nb=nb, normalization=False, time_dependent=time_dependent))
            mutil.initialize_weights(self.conv_trunk.odefunc.convs)
        elif differential == "sequential":
            self.conv_trunk = nn.Sequential(*[ODEBlock(ODEfunc(nf, nb=1, normalization=False, time_dependent=time_dependent)) for _ in range(nb)])
            for block in self.conv_trunk:
                mutil.initialize_weights(block.odefunc.convs)
        elif differential is None:
            RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
            self.conv_trunk = mutil.make_layer(RRDB_block_f, nb)
        else:
            raise NotImplementedError("unrecognized differential system passed")
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, interpolation_start=True):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.conv_trunk(fea))
        fea = fea + trunk
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='bicubic')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='bicubic')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        if interpolation_start:
            interpolated = F.interpolate(x, scale_factor=4, mode='bicubic')
            out = out + interpolated

        return out
