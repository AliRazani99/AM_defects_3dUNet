from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

def conv3d(in_channels, out_channels, kernel_size, bias, padding):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)

def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding):
    """
    Create a list of modules that together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int or tuple): size of the convolving kernel
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input

    Returns:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(negative_slope=0.1, inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            bias = not ('g' in order or 'b' in order)
            modules.append(('conv', conv3d(in_channels, out_channels, kernel_size, bias, padding=padding)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            num_channels = in_channels if is_before_conv else out_channels
            if num_channels < num_groups:
                num_groups = 1
            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        elif char == 'd':
            modules.append(('dropout', nn.Dropout3d(p=0.1)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c', 'd']")

    return modules

class SingleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, order='gcr', num_groups=8, padding=1):
        super(SingleConv, self).__init__()
        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=padding):
            self.add_module(name, module)

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='gcr', num_groups=8, padding=1):
        super(DoubleConv, self).__init__()
        if encoder:
            conv1_in_channels = in_channels
            conv1_out_channels = max(out_channels // 2, in_channels)
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        self.add_module('SingleConv1', SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups, padding=padding))
        self.add_module('SingleConv2', SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups, padding=padding))

class ExtResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, order='cge', num_groups=8):
        super(ExtResNetBlock, self).__init__()
        self.conv1 = SingleConv(in_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups)
        self.conv2 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups)
        n_order = order.translate({ord(c): None for c in 'rel'})
        self.conv3 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=n_order, num_groups=num_groups)
        self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True) if 'l' in order else nn.ELU(inplace=True) if 'e' in order else nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        residual = out
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        return self.non_linearity(out)

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True, pool_kernel_size=2, pool_type='max', basic_module=DoubleConv, conv_layer_order='gcr', num_groups=8, padding=1):
        super(Encoder, self).__init__()
        assert pool_type in ['max', 'avg']
        self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size) if apply_pooling and pool_type == 'max' else nn.AvgPool3d(kernel_size=pool_kernel_size) if apply_pooling else None
        self.basic_module = basic_module(in_channels, out_channels, encoder=True, kernel_size=conv_kernel_size, order=conv_layer_order, num_groups=num_groups, padding=padding)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        return self.basic_module(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, scale_factor=(2, 2, 2), basic_module=DoubleConv, conv_layer_order='gcr', num_groups=8, mode='nearest', padding=1):
        super(Decoder, self).__init__()
        self.upsampling = Upsampling(transposed_conv=False, in_channels=in_channels, out_channels=out_channels, kernel_size=conv_kernel_size, scale_factor=scale_factor, mode=mode)
        self.joining = partial(self._joining, concat=True) if basic_module == DoubleConv else partial(self._joining, concat=False)
        self.basic_module = basic_module(out_channels if basic_module != DoubleConv else in_channels, out_channels, encoder=False, kernel_size=conv_kernel_size, order=conv_layer_order, num_groups=num_groups, padding=padding)

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features=encoder_features, x=x)
        x = self.joining(encoder_features, x)
        return self.basic_module(x)

    @staticmethod
    def _joining(encoder_features, x, concat):
        return torch.cat((encoder_features, x), dim=1) if concat else encoder_features + x

class Upsampling(nn.Module):
    def __init__(self, transposed_conv, in_channels=None, out_channels=None, kernel_size=3, scale_factor=(2, 2, 2), mode='nearest'):
        super(Upsampling, self).__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=scale_factor, padding=1) if transposed_conv else partial(self._interpolate, mode=mode)

    def forward(self, encoder_features, x):
        output_size = encoder_features.size()[2:]
        return self.upsample(x, output_size)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)
