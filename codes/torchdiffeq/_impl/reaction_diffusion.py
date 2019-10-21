import torch
from torch import nn
import torch.nn.functional as F
import math
from utils.downsampling import cross_scale_4_loss_gradient
from multipledispatch import dispatch


class ReactionDiffusion(nn.Module):

    def __init__(self, space_channels, hidden_channels, kernel_size=3, stride=1, base_size=8, normalization=10, scale_factor=4, rbf=True, residual=False):
        super(ReactionDiffusion, self).__init__()

        self.kernel_size = kernel_size
        self.filters = nn.Parameter(torch.randn(hidden_channels, space_channels, kernel_size, kernel_size) / math.sqrt(kernel_size**2 * space_channels),
                                          requires_grad=True)
        self.stride = stride
        self.rbf = rbf
        if rbf:
            self.normalization = normalization
            self.base_size = base_size
            self.base_centers, self.std = build_radial_base(base_size, normalization)
            self.base_weights = nn.Parameter(torch.ones(hidden_channels, base_size), requires_grad=True)
        else:
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.data_weight_exponent = nn.Parameter(torch.Tensor([math.log(1/2)]), requires_grad=True)   # lambda=exp(data_weight_exponent)
        self.lr_input = None
        self.scale_factor = scale_factor
        self.nfe = 0
        self.residual = residual

    def set_lr_input(self, x):
        self.lr_input = x

    @dispatch(object, object)
    def forward(self, t, x):

        self.nfe += 1

        diffusion_term = F.conv2d(F.pad(x, [2, 2, 2, 2], mode='reflect'),
                                  self.filters, bias=None, stride=self.stride, padding=0)
        if self.rbf:
            diffusion_term = apply_rbf(diffusion_term, self.base_weights, self.base_centers, self.std)
        else:
            diffusion_term = self.activation(diffusion_term)
        diffusion_term = F.conv2d(diffusion_term, torch.flip(self.filters.transpose(0, 1), dims=[2, 3]), bias=None, stride=self.stride, padding=0)

        if self.lr_input is not None:
            reaction_term = cross_scale_4_loss_gradient(x, self.lr_input)
            diffusion_term = diffusion_term + torch.exp(self.data_weight_exponent) * reaction_term
            # diffusion_term = diffusion_term + reaction_term

        if self.residual:
            diffusion_term = diffusion_term + x

        return diffusion_term

    @dispatch(object)
    def forward(self, x):

        self.nfe += 1

        diffusion_term = F.conv2d(F.pad(x, [2, 2, 2, 2], mode='reflect'),
                                  self.filters, bias=None, stride=self.stride, padding=0)
        if self.rbf:
            diffusion_term = apply_rbf(diffusion_term, self.base_weights, self.base_centers, self.std)
        else:
            diffusion_term = self.activation(diffusion_term)
        diffusion_term = F.conv2d(diffusion_term, torch.flip(self.filters.transpose(0, 1), dims=[2, 3]), bias=None, stride=self.stride, padding=0)

        if self.lr_input is not None:
            reaction_term = cross_scale_4_loss_gradient(x, self.lr_input)
            diffusion_term = diffusion_term + torch.exp(self.data_weight_exponent) * reaction_term
            # diffusion_term = diffusion_term + reaction_term

        if self.residual:
            diffusion_term = diffusion_term + x

        return diffusion_term


def apply_rbf(inp, base_weights, base_centers, std):
    # B = batch size, H = height, W = width, K = kernel number, R = radial basis size
    base_size = base_weights.shape[1]
    inp = inp.unsqueeze(dim=len(inp.shape)).expand(*inp.shape, base_size)           # add end dim and expand tensor to (B, H, W, K, R)
    expanded_centers = base_centers[(None,) * (len(inp.shape) - 1)]                       # multiple unsqueeze to (1, 1, 1, 1, R)
    expanded_centers = expanded_centers.expand(*inp.shape)                          # expand to (B, H, W, K, R)
    gaussians = torch.exp(-torch.pow(inp - expanded_centers, 2) / (2 * std**2))
    gaussians = gaussians.permute(0, 2, 3, 1, 4)                                    # permute to put in (B, H, W, K, R) form for batched_dot_product
    result = batched_dot_product(gaussians, base_weights).permute(0, 3, 1, 2)       # permute back to (B, K, H, W) form
    return result


def batched_dot_product(inp, weights):
    input_shape = inp.shape                       # B, H, W, K, R
    weights_shape = weights.shape                 # K, R
    inp = inp.reshape(-1, *weights_shape)            # B * H * W, K, R
    result = (inp * weights).sum(dim=-1, keepdim=False)          # B * H * W, K
    result = result.view(*input_shape[:-1])       # B, H, W, K
    return result


def build_radial_base(base_size, normalization):
    std = 2 * normalization / (base_size - 1)
    centers = torch.Tensor([-normalization + std * i for i in range(base_size)]).cuda()
    return centers, std
