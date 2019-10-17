import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
from anode.anode.odeblock import make_odeblock
from anode.models.sr_trunk import SRTrunk
from torchdiffeq._impl.reaction_diffusion import ReactionDiffusion
from torchdiffeq._impl.conv import ODEBlock


class ReacDiff(nn.Module):
    def __init__(self, in_nc, out_nc, nf):
        super(ReacDiff, self).__init__()

        self.odefunc = ReactionDiffusion(in_nc, nf)
        self.conv_trunk = ODEBlock(self.odefunc)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        self.odefunc.set_lr_input(x)
        fea = F.interpolate(x, scale_factor=4, mode='bicubic')
        out = self.conv_trunk(fea)
        return out
