import torch
from torch import nn
import torch.nn.functional as F


class ReactionDiffusion(nn.Module):

    def __init__(self, space_channels, hidden_channels, kernel_size=3, stride=1, base_size=64, normalization=10):
        super(ReactionDiffusion, self).__init__()

        self.filters = torch.nn.Parameter(torch.zeros(hidden_channels, space_channels, kernel_size, kernel_size),
                                          requires_grad=True)
        self.stride = stride
        self.normalization = normalization
        self.base_size = base_size
        self.base_centers, self.std = build_radial_base(base_size, normalization)
        self.base_weights = torch.nn.Parameter(torch.ones(hidden_channels, base_size), requires_grad=True)
        self.data_weight_exponent = torch.nn.Parameter(torch.ones(1), requires_grad=True)   # lambda=exp(data_weight_exponent)
        self.lr_input = None
        self.nfe = 0

    def forward(self, t, x):

        diffusion_term = F.conv_transpose2d(
            rbf(F.conv2d(x, self.filters, bias=None, stride=self.stride, padding=0),
                self.base_weights, self.base_centers, self.std),
            self.filters, bias=None, stride=self.stride, padding=0)

        if self.lr_input is not None:
            x_copy = x.clone().detach().requires_grad_(True)
            lr_input_copy = self.lr_input.clone().detach()
            data_term = torch.norm(F.interpolate(x_copy, scale_factor=2, mode='bicubic', align_corners=False)
                                   - lr_input_copy, p=2)  # data term ||Au - f||^2
            data_term.backward()
            reaction_term = torch.exp(self.data_weight_exponent) * x_copy.grad
            diffusion_term = diffusion_term - reaction_term

        return diffusion_term


def rbf(inp, base_weights, base_centers, std):
    # B = batch size, H = height, W = width, K = kernel number, R = radial basis size
    base_size = base_weights.shape[0]
    inp = inp.unsqueeze(dim=len(inp.shape)).expand(*inp.shape, base_size)           # add end dim and expand tensor to (B, H, W, K, R)
    base_centers = base_centers[(None,) * len(inp.shape)]                           # multiple unsqueeze to (1, 1, 1, 1, R)
    base_centers = base_centers.expand(*inp.shape)                                  # expand to (B, H, W, K, R)
    gaussians = torch.exp(-torch.pow(inp - base_centers, 2) / (2 * std**2))
    gaussians = gaussians.permute(0, 2, 3, 1, 4)                                    # permute to put in (B, H, W, K, R) form for batched_dot_product
    result = batched_dot_product(gaussians, base_weights).permute(0, 3, 1, 2)       # permute back to (B, K, H, W) form
    return result


def batched_dot_product(inp, weights):
    input_shape = inp.shape                       # B, H, W, K, R
    weights_shape = weights.shape                 # K, R
    inp = inp.view(-1, *weights_shape)            # B * H * W, K, R
    result = (inp * weights).sum(sum=-1)          # B * H * W, K
    result = result.view(*input_shape[:-1])       # B, H, W, K
    return result


def build_radial_base(base_size, normalization):
    std = 2 * normalization / (base_size - 1)
    centers = [-normalization + std * i for i in range(base_size)]
    return centers, std
