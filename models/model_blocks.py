import torch
import math
from torch import nn
from collections import OrderedDict
import sys


"""
Reference : https://github.com/alterzero/DBPN-Pytorch/blob/master/base_networks.py
"""

class DenseBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation='relu', norm='batch'):
        super(DenseBlock, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm1d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm1d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.fc(x))
        else:
            out = self.fc(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm='batch'):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(num_filter)
        elif norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(num_filter)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()


    def forward(self, x):
        residual = x
        if self.norm is not None:
            out = self.bn(self.conv1(x))
        else:
            out = self.conv1(x)

        if self.activation is not None:
            out = self.act(out)

        if self.norm is not None:
            out = self.bn(self.conv2(out))
        else:
            out = self.conv2(out)

        out = torch.add(out, residual)
        return out

class UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(UpBlock, self).__init__()
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
    	h0 = self.up_conv1(x)
    	l0 = self.up_conv2(h0)
    	h1 = self.up_conv3(l0 - x)
    	return h1 + h0

class UpBlockPix(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, scale=4, bias=True, activation='prelu', norm=None):
        super(UpBlockPix, self).__init__()
        self.up_conv1 = Upsampler(scale,num_filter)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = Upsampler(scale,num_filter)

    def forward(self, x):
    	h0 = self.up_conv1(x)
    	l0 = self.up_conv2(h0)
    	h1 = self.up_conv3(l0 - x)
    	return h1 + h0

class D_UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu', norm=None):
        super(D_UpBlock, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
    	x = self.conv(x)
    	h0 = self.up_conv1(x)
    	l0 = self.up_conv2(h0)
    	h1 = self.up_conv3(l0 - x)
    	return h1 + h0

class D_UpBlockPix(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, scale=4, bias=True, activation='prelu', norm=None):
        super(D_UpBlockPix, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.up_conv1 = Upsampler(scale,num_filter)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = Upsampler(scale,num_filter)

    def forward(self, x):
    	x = self.conv(x)
    	h0 = self.up_conv1(x)
    	l0 = self.up_conv2(h0)
    	h1 = self.up_conv3(l0 - x)
    	return h1 + h0

class DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(DownBlock, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
    	l0 = self.down_conv1(x)
    	h0 = self.down_conv2(l0)
    	l1 = self.down_conv3(h0 - x)
    	return l1 + l0

class DownBlockPix(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, scale=4,bias=True, activation='prelu', norm=None):
        super(DownBlockPix, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = Upsampler(scale,num_filter)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
    	l0 = self.down_conv1(x)
    	h0 = self.down_conv2(l0)
    	l1 = self.down_conv3(h0 - x)
    	return l1 + l0

class D_DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu', norm=None):
        super(D_DownBlock, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
    	x = self.conv(x)
    	l0 = self.down_conv1(x)
    	h0 = self.down_conv2(l0)
    	l1 = self.down_conv3(h0 - x)
    	return l1 + l0

class D_DownBlockPix(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, scale=4, bias=True, activation='prelu', norm=None):
        super(D_DownBlockPix, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = Upsampler(scale,num_filter)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        x = self.conv(x)
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0

class PSBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, scale_factor, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm='batch'):
        super(PSBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size * scale_factor**2, kernel_size, stride, padding, bias=bias)
        self.ps = torch.nn.PixelShuffle(scale_factor)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.ps(self.conv(x)))
        else:
            out = self.ps(self.conv(x))

        if self.activation is not None:
            out = self.act(out)
        return out


class Upsampler(torch.nn.Module):
    def __init__(self, scale, n_feat, bn=False, act='prelu', bias=True):
        super(Upsampler, self).__init__()
        modules = []
        for _ in range(int(math.log(scale, 2))):
            modules.append(ConvBlock(n_feat, 4 * n_feat, 3, 1, 1, bias, activation=None, norm=None))
            modules.append(torch.nn.PixelShuffle(2))
            if bn: modules.append(torch.nn.BatchNorm2d(n_feat))
            #modules.append(torch.nn.PReLU())
        self.up = torch.nn.Sequential(*modules)

        self.activation = act
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.up(x)
        if self.activation is not None:
            out = self.act(out)
        return out


class Upsample2xBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True, upsample='deconv', activation='relu', norm='batch'):
        super(Upsample2xBlock, self).__init__()
        scale_factor = 2
        # 1. Deconvolution (Transposed convolution)
        if upsample == 'deconv':
            self.upsample = DeconvBlock(input_size, output_size,
                                        kernel_size=4, stride=2, padding=1,
                                        bias=bias, activation=activation, norm=norm)

        # 2. Sub-pixel convolution (Pixel shuffler)
        elif upsample == 'ps':
            self.upsample = PSBlock(input_size, output_size, scale_factor=scale_factor,
                                    bias=bias, activation=activation, norm=norm)

        # 3. Resize and Convolution
        elif upsample == 'rnc':
            self.upsample = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=scale_factor, mode='nearest'),
                ConvBlock(input_size, output_size,
                          kernel_size=3, stride=1, padding=1,
                          bias=bias, activation=activation, norm=norm)
            )

    def forward(self, x):
        out = self.upsample(x)
        return out




################
# Basic blocks
################

def activation(act_type='relu', inplace=True, slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    layer = None
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=slope)
    else:
        raise NotImplementedError('[ERROR] Activation layer [%s] is not implemented!'%act_type)
    return layer


def norm(n_feature, norm_type='bn'):
    norm_type = norm_type.lower()
    layer = None
    if norm_type =='bn':
        layer = nn.BatchNorm2d(n_feature)
    else:
        raise NotImplementedError('[ERROR] Normalization layer [%s] is not implemented!'%norm_type)
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None

    layer = None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('[ERROR] Padding layer [%s] is not implemented!'%pad_type)
    return layer


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('[ERROR] %s.sequential() does not support OrderedDict'%sys.modules[__name__])
        else:
            return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module:
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def ConvBlock_v2(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, valid_padding=True, padding=0,\
              act_type='relu', norm_type='bn', pad_type='zero', mode='CNA'):
    assert (mode in ['CNA', 'NAC']), '[ERROR] Wrong mode in [%s]!'%sys.modules[__name__]

    if valid_padding:
        padding = get_valid_padding(kernel_size, dilation)
    else:
        pass
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    if mode == 'CNA':
        act = activation(act_type) if act_type else None
        n = norm(out_channels, norm_type) if norm_type else None
        return sequential(p, conv, n, act)
    elif mode == 'NAC':
        act = activation(act_type, inplace=False) if act_type else None
        n = norm(in_channels, norm_type) if norm_type else None
        return sequential(n, act, p, conv)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * 255. * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

################
# Advanced blocks
################

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channle, mid_channel, kernel_size, stride=1, valid_padding=True, padding=0, dilation=1, bias=True, \
                 pad_type='zero', norm_type='bn', act_type='relu', mode='CNA', res_scale=1):
        super(ResBlock, self).__init__()
        conv0 = ConvBlock(in_channel, mid_channel, kernel_size, stride, dilation, bias, valid_padding, padding, act_type, norm_type, pad_type, mode)
        act_type = None
        norm_type = None
        conv1 = ConvBlock(mid_channel, out_channle, kernel_size, stride, dilation, bias, valid_padding, padding, act_type, norm_type, pad_type, mode)
        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res

class UpprojBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, valid_padding=False, padding=0, bias=True, \
                 pad_type='zero', norm_type=None, act_type='prelu'):
        super(UpprojBlock, self).__init__()

        self.deconv_1 = DeconvBlock(in_channel, out_channel, kernel_size, stride=stride, \
                                    padding=padding, norm_type=norm_type, act_type=act_type)

        self.conv_1 = ConvBlock(out_channel, out_channel, kernel_size, stride=stride, padding=padding, \
                                valid_padding=valid_padding, norm_type=norm_type, act_type=act_type)

        self.deconv_2 = DeconvBlock(out_channel, out_channel, kernel_size, stride=stride, \
                                    padding=padding, norm_type=norm_type, act_type=act_type)

    def forward(self, x):
        H_0_t = self.deconv_1(x)
        L_0_t = self.conv_1(H_0_t)
        H_1_t = self.deconv_2(L_0_t-x)

        return H_0_t + H_1_t

class D_UpprojBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, valid_padding=False, padding=0, bias=True, \
                         pad_type='zero', norm_type=None, act_type='prelu'):
        super(D_UpprojBlock, self).__init__()
        self.conv_1 = ConvBlock(in_channel, out_channel, kernel_size=1, norm_type=norm_type, act_type=act_type)
        self.deconv_1 = DeconvBlock(out_channel, out_channel, kernel_size, stride=stride, \
                                    padding=padding, norm_type=norm_type, act_type=act_type)
        self.conv_2 = ConvBlock(out_channel, out_channel, kernel_size, stride=stride, padding=padding, \
                                valid_padding=valid_padding, norm_type=norm_type, act_type=act_type)
        self.deconv_2 = DeconvBlock(out_channel, out_channel, kernel_size, stride=stride, \
                                    padding=padding, norm_type=norm_type, act_type=act_type)

    def forward(self, x):
        x = self.conv_1(x)
        H_0_t = self.deconv_1(x)
        L_0_t = self.conv_2(H_0_t)
        H_1_t = self.deconv_2(L_0_t - x)

        return H_1_t + H_0_t

class DownprojBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, valid_padding=True,
                         padding=0, dilation=1, bias=True, \
                         pad_type='zero', norm_type=None, act_type='prelu', mode='CNA', res_scale=1):
        super(DownprojBlock, self).__init__()

        self.conv_1 = ConvBlock(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                valid_padding=valid_padding, norm_type=norm_type, act_type=act_type)

        self.deconv_1 = DeconvBlock(out_channel, out_channel, kernel_size, stride=stride,
                                    padding=padding, norm_type=norm_type, act_type=act_type)

        self.conv_2 = ConvBlock(out_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                valid_padding=valid_padding, norm_type=norm_type, act_type=act_type)

    def forward(self, x):
        L_0_t = self.conv_1(x)
        H_0_t = self.deconv_1(L_0_t)
        L_1_t = self.conv_2(H_0_t - x)

        return L_0_t + L_1_t

class D_DownprojBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, valid_padding=False, padding=0, bias=True, \
                         pad_type='zero', norm_type=None, act_type='prelu'):
        super(D_DownprojBlock, self).__init__()
        self.conv_1 = ConvBlock(in_channel, out_channel, kernel_size=1, norm_type=norm_type, act_type=act_type)

        self.conv_2 = ConvBlock(out_channel, out_channel, kernel_size, stride=stride, padding=padding, \
                                valid_padding=valid_padding, norm_type=norm_type, act_type=act_type)
        self.deconv_1 = DeconvBlock(out_channel, out_channel, kernel_size, stride=stride, \
                                    padding=padding, norm_type=norm_type, act_type=act_type)
        self.conv_3 = ConvBlock(out_channel, out_channel, kernel_size, stride=stride, padding=padding, \
                                valid_padding=valid_padding, norm_type=norm_type, act_type=act_type)

    def forward(self, x):
        x = self.conv_1(x)
        L_0_t = self.conv_2(x)
        H_0_t = self.deconv_1(L_0_t)
        L_1_t = self.conv_3(H_0_t - x)

        return L_1_t + L_0_t

class DensebackprojBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, bp_stages, stride=1, valid_padding=True,
                         padding=0, dilation=1, bias=True, \
                         pad_type='zero', norm_type=None, act_type='prelu', mode='CNA', res_scale=1):
        super(DensebackprojBlock, self).__init__()

        # This is an example that I have to create nn.ModuleList() to append a sequence of models instead of list()
        self.upproj = nn.ModuleList()
        self.downproj = nn.ModuleList()
        self.bp_stages = bp_stages
        self.upproj.append(UpprojBlock(in_channel, out_channel, kernel_size, stride=stride, valid_padding=False,
                                             padding=padding, norm_type=norm_type, act_type=act_type))

        for index in range(self.bp_stages - 1):
            if index < 1:
                self.upproj.append(UpprojBlock(out_channel, out_channel, kernel_size, stride=stride, valid_padding=False,
                                      padding=padding, norm_type=norm_type, act_type=act_type))
            else:
                uc = ConvBlock(out_channel*(index+1), out_channel, kernel_size=1, norm_type=norm_type, act_type=act_type)
                u = UpprojBlock(out_channel, out_channel, kernel_size, stride=stride, valid_padding=False,
                                padding=padding, norm_type=norm_type, act_type=act_type)
                self.upproj.append(sequential(uc, u))

            if index < 1:
                self.downproj.append(DownprojBlock(out_channel, out_channel, kernel_size, stride=stride, valid_padding=False,
                                                  padding=padding, norm_type=norm_type, act_type=act_type))
            else:
                dc = ConvBlock(out_channel*(index+1), out_channel, kernel_size=1, norm_type=norm_type, act_type=act_type)
                d = DownprojBlock(out_channel, out_channel, kernel_size, stride=stride, valid_padding=False,
                                padding=padding, norm_type=norm_type, act_type=act_type)
                self.downproj.append(sequential(dc, d))

    def forward(self, x):
        low_features = []
        high_features = []

        H = self.upproj[0](x)
        high_features.append(H)

        for index in range(self.bp_stages - 1):
            if index < 1:
                L = self.downproj[index](H)
                low_features.append(L)
                H = self.upproj[index+1](L)
                high_features.append(H)
            else:
                H_concat = torch.cat(tuple(high_features), 1)
                L = self.downproj[index](H_concat)
                low_features.append(L)
                L_concat = torch.cat(tuple(low_features), 1)
                H = self.upproj[index+1](L_concat)
                high_features.append(H)

        output = torch.cat(tuple(high_features), 1)
        return output


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

class ConcatBlock(nn.Module):
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), 1)
        return output

################
# Upsampler
################
def UpsampleConvBlock(upscale_factor, in_channels, out_channels, kernel_size, stride, valid_padding=True, padding=0, bias=True,\
                 pad_type='zero', act_type='relu', norm_type=None, mode='nearest'):
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = ConvBlock(in_channels, out_channels, kernel_size, stride, bias=bias, valid_padding=valid_padding, padding=padding, \
                     pad_type=pad_type, act_type=act_type, norm_type=norm_type)
    return sequential(upsample, conv)


def PixelShuffleBlock():
    pass


def DeconvBlock_v2(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, padding=0, \
                act_type='relu', norm_type='bn', pad_type='zero', mode='CNA'):
    assert (mode in ['CNA', 'NAC']), '[ERROR] Wrong mode in [%s]!'%sys.modules[__name__]

    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias)

    if mode == 'CNA':
        act = activation(act_type) if act_type else None
        n = norm(out_channels, norm_type) if norm_type else None
        return sequential(p, deconv, n, act)
    elif mode == 'NAC':
        act = activation(act_type, inplace=False) if act_type else None
        n = norm(in_channels, norm_type) if norm_type else None
        return sequential(n, act, p, deconv)


################
# helper funcs
################

def get_valid_padding(kernel_size, dilation):
    """
    Padding value to remain feature size.
    """
    kernel_size = kernel_size + (kernel_size-1)*(dilation-1)
    padding = (kernel_size-1) // 2
    return padding



class FeedbackBlock(nn.Module):
    def __init__(self, num_features, num_groups, upscale_factor, act_type, norm_type):
        super(FeedbackBlock, self).__init__()
        if upscale_factor == 2:
            stride = 2
            padding = 2
            kernel_size = 6
        elif upscale_factor == 3:
            stride = 3
            padding = 2
            kernel_size = 7
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            kernel_size = 8
        elif upscale_factor == 8:
            stride = 8
            padding = 2
            kernel_size = 12

        self.num_groups = num_groups

        self.compress_in = ConvBlock_v2(2*num_features, num_features,
                                     kernel_size=1,
                                     act_type=act_type, norm_type=norm_type)

        self.upBlocks = nn.ModuleList()
        self.downBlocks = nn.ModuleList()
        self.uptranBlocks = nn.ModuleList()
        self.downtranBlocks = nn.ModuleList()

        for idx in range(self.num_groups):
            self.upBlocks.append(DeconvBlock_v2(num_features, num_features,
                                             kernel_size=kernel_size, stride=stride, padding=padding,
                                             act_type=act_type, norm_type=norm_type))
            self.downBlocks.append(ConvBlock_v2(num_features, num_features,
                                             kernel_size=kernel_size, stride=stride, padding=padding,
                                             act_type=act_type, norm_type=norm_type, valid_padding=False))
            if idx > 0:
                self.uptranBlocks.append(ConvBlock_v2(num_features*(idx+1), num_features,
                                                   kernel_size=1, stride=1,
                                                   act_type=act_type, norm_type=norm_type))
                self.downtranBlocks.append(ConvBlock_v2(num_features*(idx+1), num_features,
                                                     kernel_size=1, stride=1,
                                                     act_type=act_type, norm_type=norm_type))

        self.compress_out = ConvBlock_v2(num_groups*num_features, num_features,
                                      kernel_size=1,
                                      act_type=act_type, norm_type=norm_type)

        self.should_reset = True
        self.last_hidden = None

    def forward(self, x):
        if self.should_reset:
            self.last_hidden = torch.zeros(x.size()).cuda()
            self.last_hidden.copy_(x)
            self.should_reset = False

        x = torch.cat((x, self.last_hidden), dim=1)
        x = self.compress_in(x)

        lr_features = []
        hr_features = []
        lr_features.append(x)

        for idx in range(self.num_groups):
            LD_L = torch.cat(tuple(lr_features), 1)    # when idx == 0, lr_features == [x]
            if idx > 0:
                LD_L = self.uptranBlocks[idx-1](LD_L)
            LD_H = self.upBlocks[idx](LD_L)

            hr_features.append(LD_H)

            LD_H = torch.cat(tuple(hr_features), 1)
            if idx > 0:
                LD_H = self.downtranBlocks[idx-1](LD_H)
            LD_L = self.downBlocks[idx](LD_H)

            lr_features.append(LD_L)

        del hr_features
        output = torch.cat(tuple(lr_features[1:]), 1)   # leave out input x, i.e. lr_features[0]
        output = self.compress_out(output)

        self.last_hidden = output

        return output

    def reset_state(self):
        self.should_reset = True


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    