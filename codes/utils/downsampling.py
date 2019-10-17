import torch
from torch.nn.functional import conv2d, pad, conv_transpose2d
import matplotlib.pyplot as plt


def display_tensor(x):
    plt.imshow(x[0].permute(1, 2, 0) / 255)


def cubic_kernel_function(x, b=1 / 3, c=1 / 3):
    if -1 < x < 1:
        return (12 - 9 * b - 6 * c) * abs(x) ** 3 + (-18 + 12 * b + 6 * c) * abs(x) ** 2 + (6 - 2 * b)
    if -2 < x < 2:
        return (-b - 6 * c) * abs(x) ** 3 + (6 * b + 30 * c) * abs(x) ** 2 + (-12 * b - 48 * c) * abs(x) + (
                8 * b + 24 * c)
    return 0


CATMULL_ROM_2X = tuple(cubic_kernel_function(x / 2 + 1 / 4, 0, 0.5) for x in range(-4, 4))
MITCHELL_2X = tuple(cubic_kernel_function(x / 2 + 1 / 4) for x in range(-4, 4))
CATMULL_ROM_4X = tuple(cubic_kernel_function(x / 4 + 1 / 8, 0, 0.5) for x in range(-8, 8))
MITCHELL_4X = tuple(cubic_kernel_function(x / 4 + 1 / 8) for x in range(-8, 8))


def filter_2x(b, c):
    return [cubic_kernel_function(x / 2 + 1 / 4, b, c) for x in range(-4, 4)]


def filter_4x(b, c):
    return [cubic_kernel_function(x / 4 + 1 / 8, b, c) for x in range(-8, 8)]


def filter_to_square_kweight(sampling_filter, cuda=True):
    normalization = sum(sampling_filter)
    tensor_filter = torch.Tensor(sampling_filter) / normalization
    tensor_filter = torch.einsum('i, j->ij', [tensor_filter, tensor_filter])
    channel_matrix = torch.eye(3)
    kweight = torch.einsum('ij, kl->ijkl', [channel_matrix, tensor_filter])
    if cuda:
        kweight = kweight.cuda()
    return kweight


def downsample_2_sq(x, sampling_filter=CATMULL_ROM_2X, cuda=True):
    x = pad(x, [3, 3, 3, 3], 'reflect')
    x = conv2d(x, filter_to_square_kweight(sampling_filter, cuda), stride=2)
    return x


def downsample_4_sq(x, sampling_filter=CATMULL_ROM_4X, cuda=True):
    x = pad(x, [6, 6, 6, 6], 'reflect')
    x = conv2d(x, filter_to_square_kweight(sampling_filter, cuda), stride=4)
    return x


def upsample_2_sq(x, sampling_filter=CATMULL_ROM_2X, cuda=True):
    x = conv_transpose2d(x, filter_to_square_kweight(sampling_filter, cuda), stride=2)
    return x[:, :, 3:-3, 3:-3]


def upsample_4_sq(x, sampling_filter=CATMULL_ROM_4X, cuda=True):
    x = conv_transpose2d(x, filter_to_square_kweight(sampling_filter, cuda), stride=4)
    return x[:, :, 6:-6, 6:-6]


def cross_scale_2_loss_gradient(hr_input, lr_ref, sampling_filter=CATMULL_ROM_2X, cuda=True):
    return -upsample_2_sq(downsample_2_sq(hr_input, sampling_filter=sampling_filter, cuda=cuda) - lr_ref,
                          sampling_filter=sampling_filter, cuda=cuda)


def cross_scale_4_loss_gradient(hr_input, lr_ref, sampling_filter=CATMULL_ROM_4X, cuda=True):
    return -upsample_4_sq(downsample_4_sq(hr_input, sampling_filter=sampling_filter, cuda=cuda) - lr_ref,
                          sampling_filter=sampling_filter, cuda=cuda)
