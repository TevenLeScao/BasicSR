import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchdiffeq import odeint_adjoint as odeint
# from torchdiffeq import odeint


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc(nn.Module):

    def __init__(self, dim, nb, normalization=True):
        super(ODEfunc, self).__init__()
        self.normalization = normalization
        self.relu = nn.ReLU(inplace=False)
        self.nb = nb
        for _ in range(nb):
            self.convs = nn.ModuleList([ConcatConv2d(dim, dim, 3, 1, 1) for _ in range(nb)])
        if self.normalization:
            self.norms = nn.ModuleList([norm(dim) for _ in range(nb + 1)])
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        if self.normalization:
            out = self.norms[-1](x)
            out = self.relu(out)
        else:
            out = self.relu(x)
        for i in range(self.nb):
            out = self.convs[i](t, out)
            if self.normalization:
                out = self.norms[i](out)
        return out


class StaticODEfunc(nn.Module):

    def __init__(self, dim, nb, normalization=True):
        super(StaticODEfunc, self).__init__()
        self.normalization = normalization
        self.relu = nn.ReLU(inplace=False)
        self.nb = nb
        self.convs = nn.ModuleList([nn.Conv2d(dim, dim, 3, 1, 1) for _ in range(nb)])
        if self.normalization:
            self.norms = nn.ModuleList([norm(dim) for _ in range(nb + 1)])
        self.nfe = 0

    def forward(self, x):
        self.nfe += 1
        if self.normalization:
            out = self.norms[-1](x)
            out = self.relu(out)
        else:
            out = self.relu(x)
        for i in range(self.nb):
            out = self.convs[i](out)
            if self.normalization:
                out = self.norms[i](out)
        return out


class DenseODEfunc(nn.Module):

    def __init__(self, dim=64, growth=32, nb=5, bias=True, normalization=True):
        super(DenseODEfunc, self).__init__()
        self.normalization = normalization
        self.nb = nb
        assert nb > 0
        self.convs = nn.ModuleList([nn.Conv2d(dim + growth*conv_index, dim + growth*(conv_index+1), 3, 1, 1, bias=bias) for conv_index in range(nb-1)])
        self.final_conv = nn.Conv2d(dim + growth*(nb-1), dim, 3, 1, 1, bias=bias)
        if self.normalization:
            self.norms = nn.ModuleList([norm(dim + growth*conv_index) for conv_index in range(nb + 1)])
        self.nfe = 0
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x):
        self.nfe += 1
        if self.normalization:
            out = self.norms[-1](x)
            out = self.lrelu(out)
        else:
            out = self.lrelu(x)
        for i in range(self.nb-1):
            out = self.convs[i](out)
            if self.normalization:
                out = self.norms[i](out)
            out = self.lrelu(out)
        out = self.final_conv(out)
        out = self.lrelu(out)
        return out * 0.2 + x


class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
