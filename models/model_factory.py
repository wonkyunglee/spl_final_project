import torch
import torch.nn as nn
import math
from .model_blocks import UpBlock, DownBlock, D_UpBlock, D_DownBlock, ConvBlock
from .model_blocks import FeedbackBlock, ConvBlock_v2, DeconvBlock_v2, MeanShift, Flatten
import numpy as np
import sys, os
sys.path.append('../')
from utils.imresize import imresize


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DBPN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, base_filter=64, feat=256,
                 num_stages=7):
        super(DBPN, self).__init__()

        if scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2

        #Initial Feature Extraction
        self.feat0 = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu', norm=None)
        self.feat1 = ConvBlock(feat, base_filter, 1, 1, 0, activation='prelu', norm=None)
        #Back-projection stages
        self.up1 = UpBlock(base_filter, kernel, stride, padding)
        self.down1 = DownBlock(base_filter, kernel, stride, padding)
        self.up2 = UpBlock(base_filter, kernel, stride, padding)
        self.down2 = D_DownBlock(base_filter, kernel, stride, padding, 2)
        self.up3 = D_UpBlock(base_filter, kernel, stride, padding, 2)
        self.down3 = D_DownBlock(base_filter, kernel, stride, padding, 3)
        self.up4 = D_UpBlock(base_filter, kernel, stride, padding, 3)
        self.down4 = D_DownBlock(base_filter, kernel, stride, padding, 4)
        self.up5 = D_UpBlock(base_filter, kernel, stride, padding, 4)
        self.down5 = D_DownBlock(base_filter, kernel, stride, padding, 5)
        self.up6 = D_UpBlock(base_filter, kernel, stride, padding, 5)
        self.down6 = D_DownBlock(base_filter, kernel, stride, padding, 6)
        self.up7 = D_UpBlock(base_filter, kernel, stride, padding, 6)
        #Reconstruction
        self.output_conv = ConvBlock(num_stages*base_filter, num_channels, 3, 1, 1, activation=None, norm=None)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        original = x
        x = self.feat0(x)
        x = self.feat1(x)

        h1 = self.up1(x)
        l1 = self.down1(h1)
        h2 = self.up2(l1)

        concat_h = torch.cat((h2, h1),1)
        l = self.down2(concat_h)

        concat_l = torch.cat((l, l1),1)
        h = self.up3(concat_l)

        concat_h = torch.cat((h, concat_h),1)
        l = self.down3(concat_h)

        concat_l = torch.cat((l, concat_l),1)
        h = self.up4(concat_l)

        concat_h = torch.cat((h, concat_h),1)
        l = self.down4(concat_h)

        concat_l = torch.cat((l, concat_l),1)
        h = self.up5(concat_l)

        concat_h = torch.cat((h, concat_h),1)
        l = self.down5(concat_h)

        concat_l = torch.cat((l, concat_l),1)
        h = self.up6(concat_l)

        concat_h = torch.cat((h, concat_h),1)
        l = self.down6(concat_h)

        concat_l = torch.cat((l, concat_l),1)
        h = self.up7(concat_l)

        concat_h = torch.cat((h, concat_h),1)
        x = self.output_conv(concat_h)
        return x

class DRRN(nn.Module):
    """
    This implementation is from https://github.com/jt827859032/DRRN-pytorch
    """
    def __init__(self, scale_factor=None):
        super(DRRN, self).__init__()
        self.input = nn.Conv2d(in_channels=1, out_channels=128,
                               kernel_size=3,stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=128, out_channels=1,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))


    def forward(self, x):
        residual = x
        inputs = self.input(self.relu(x))
        out = inputs
        for _ in range(25):
            out = self.conv2(self.relu(self.conv1(self.relu(out))))
            out = torch.add(out, inputs)

        out = self.output(self.relu(out))
        out = torch.add(out, residual)
        return out



class FSRCNN(nn.Module):

    def __init__(self, scale_factor=4):
        super(FSRCNN, self).__init__()
        self.scale_factor = scale_factor

        # Feature extraction
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(1, 56, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
        )
        # shrinking
        self.shrinking = nn.Sequential(
            nn.Conv2d(56, 12, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
        )
        # non-linear mapping
        self.non_lin_mapping = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        # expanding
        self.expanding = nn.Sequential(
            nn.Conv2d(12, 56, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
        )
        # deconv
        self.deconvolution = nn.ConvTranspose2d(56, 1, kernel_size=5+self.scale_factor,
                                                stride=self.scale_factor, padding=3,
                                                output_padding=1)
        self.weight_init()


    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
                # nn.init.kaiming_normal_(m.weight.data, a=0.25) # for SGD
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, LR_patch):
        output = self.feature_extraction(LR_patch)
        output = self.shrinking(output)
        output = self.non_lin_mapping(output)
        output = self.expanding(output)
        # assert output.shape[-1] == 16, output.shape
        output = self.deconvolution(output)
        # assert output.shape[-1] == 64, output.shape
        return output


'''
Reference : https://github.com/Paper99/SRFBN_CVPR19/blob/master/networks/srfbn_arch.py
'''
class SRFBN(nn.Module):
    def __init__(self, scale_factor, in_channels=1, out_channels=1,
                 num_features=64, num_steps=4, num_groups=6,
                 act_type='prelu', norm_type=None):
        super(SRFBN, self).__init__()

        if scale_factor == 2:
            stride = 2
            padding = 2
            kernel_size = 6
        elif scale_factor == 3:
            stride = 3
            padding = 2
            kernel_size = 7
        elif scale_factor == 4:
            stride = 4
            padding = 2
            kernel_size = 8
        elif scale_factor == 8:
            stride = 8
            padding = 2
            kernel_size = 12

        self.num_steps = num_steps
        self.num_features = num_features
        self.scale_factor = scale_factor

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_mean, rgb_std)

        # LR feature extraction block
        self.conv_in = ConvBlock_v2(in_channels, 4*num_features,
                                 kernel_size=3,
                                 act_type=act_type, norm_type=norm_type)
        self.feat_in = ConvBlock_v2(4*num_features, num_features,
                                 kernel_size=1,
                                 act_type=act_type, norm_type=norm_type)

        # basic block
        self.block = FeedbackBlock(num_features, num_groups, scale_factor, act_type, norm_type)

        # reconstruction block
        self.out = DeconvBlock_v2(num_features, num_features,
                               kernel_size=kernel_size, stride=stride, padding=padding,
                               act_type='prelu', norm_type=norm_type)
        self.conv_out = ConvBlock_v2(num_features, out_channels,
                                  kernel_size=3,
                                  act_type=None, norm_type=norm_type)

        self.add_mean = MeanShift(rgb_mean, rgb_std, 1)

    def forward(self, x):
        self._reset_state()

        # x = self.sub_mean(x)

        inter_res = nn.functional.interpolate(x, scale_factor=self.scale_factor,
                                              mode='bilinear', align_corners=False)

        x = self.conv_in(x)
        x = self.feat_in(x)

        outs = []
        for _ in range(self.num_steps):
            h = self.block(x)

            h = torch.add(inter_res, self.conv_out(self.out(h)))
            # h = self.add_mean(h)
            outs.append(h)

        return outs # return output of every timesteps

    def _reset_state(self):
        self.block.reset_state()


class PSRN(nn.Module):
    def __init__(self, scale_factor=4, residual=False, repeat=10):
        super(PSRN, self).__init__()

        self.scale_factor = scale_factor
        self.repeat = repeat
        self.residual = residual

        self.layers = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(48, 24, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(24, 12, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(12, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):

        inter_res = nn.functional.interpolate(x, scale_factor=self.scale_factor,
                                      mode='bilinear', align_corners=False)

        x = nn.functional.interpolate(x, scale_factor=self.scale_factor,
                                      mode='bilinear', align_corners=False)


        out = []
        for i in range(self.repeat):
            x = self.layers(x)
            if self.residual:
                x = torch.add(inter_res, x)
                inter_res = x.detach()
            out.append(x)
        return out



class PSRN_S(nn.Module):
    def __init__(self, scale_factor=4, residual=False, repeat=10):
        super(PSRN_S, self).__init__()

        self.scale_factor = scale_factor
        self.repeat = repeat
        self.residual = residual

        self.layers = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(12, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):

        inter_res = nn.functional.interpolate(x, scale_factor=self.scale_factor,
                                      mode='bilinear', align_corners=False)

        x = nn.functional.interpolate(x, scale_factor=self.scale_factor,
                                      mode='bilinear', align_corners=False)


        out = []
        for i in range(self.repeat):
            x = self.layers(x)
            if self.residual:
                x = torch.add(inter_res, x)
                inter_res = x.detach()
            out.append(x)
        return out


class PSRN_L(nn.Module):
    def __init__(self, scale_factor=4, residual=False, repeat=10):
        super(PSRN_L, self).__init__()

        self.scale_factor = scale_factor
        self.repeat = repeat
        self.residual = residual

        self.layers = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(192, 96, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(96, 48, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(48, 24, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(24, 12, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(12, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):

        inter_res = nn.functional.interpolate(x, scale_factor=self.scale_factor,
                                      mode='bilinear', align_corners=False)

        x = nn.functional.interpolate(x, scale_factor=self.scale_factor,
                                      mode='bilinear', align_corners=False)


        out = []
        for i in range(self.repeat):
            x = self.layers(x)
            if self.residual:
                x = torch.add(inter_res, x)
                inter_res = x.detach()
            out.append(x)
        return out



class PSRN_D(nn.Module):
    def __init__(self, scale_factor=4, residual=True, repeat=10):
        super(PSRN_D, self).__init__()

        self.scale_factor = scale_factor
        self.repeat = repeat
        self.residual = residual

        self.layers = nn.Sequential(
            ConvBlock_v2(1, 12, kernel_size=3, stride=1, padding=1, act_type='prelu', norm_type=None),

            ConvBlock_v2(12, 12, kernel_size=3, stride=2, padding=1, act_type='prelu', norm_type=None),
            DeconvBlock_v2(12, 12,
                               kernel_size=6, stride=2, padding=2,
                               act_type='prelu', norm_type=None),

            ConvBlock_v2(12, 12, kernel_size=3, stride=2, padding=1, act_type='prelu', norm_type=None),
            DeconvBlock_v2(12, 12,
                               kernel_size=6, stride=2, padding=2,
                               act_type='prelu', norm_type=None),

            ConvBlock_v2(12, 12, kernel_size=3, stride=2, padding=1, act_type='prelu', norm_type=None),
            DeconvBlock_v2(12, 12,
                               kernel_size=6, stride=2, padding=2,
                               act_type='prelu', norm_type=None),

            nn.Conv2d(12, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):

        inter_res = nn.functional.interpolate(x, scale_factor=self.scale_factor,
                                      mode='bilinear', align_corners=False)

        x = nn.functional.interpolate(x, scale_factor=self.scale_factor,
                                      mode='bilinear', align_corners=False)


        out = []
        for i in range(self.repeat):
            x = self.layers(x)
            if self.residual:
                x = torch.add(inter_res, x)
                inter_res = x.detach()
            out.append(x)
        return out



class PSRN_DL(nn.Module):
    def __init__(self, scale_factor=4, residual=True, repeat=10):
        super(PSRN_DL, self).__init__()

        self.scale_factor = scale_factor
        self.repeat = repeat
        self.residual = residual

        self.layers = nn.Sequential(
            ConvBlock_v2(1, 64, kernel_size=3, stride=1, padding=1, act_type='prelu', norm_type=None),

            ConvBlock_v2(64, 64, kernel_size=3, stride=2, padding=1, act_type='prelu', norm_type=None),
            DeconvBlock_v2(64, 64,
                               kernel_size=6, stride=2, padding=2,
                               act_type='prelu', norm_type=None),

            ConvBlock_v2(64, 64, kernel_size=3, stride=2, padding=1, act_type='prelu', norm_type=None),
            DeconvBlock_v2(64, 64,
                               kernel_size=6, stride=2, padding=2,
                               act_type='prelu', norm_type=None),

            ConvBlock_v2(64, 64, kernel_size=3, stride=2, padding=1, act_type='prelu', norm_type=None),
            DeconvBlock_v2(64, 64,
                               kernel_size=6, stride=2, padding=2,
                               act_type='prelu', norm_type=None),

            ConvBlock_v2(64, 64, kernel_size=3, stride=2, padding=1, act_type='prelu', norm_type=None),
            DeconvBlock_v2(64, 64,
                               kernel_size=6, stride=2, padding=2,
                               act_type='prelu', norm_type=None),

            ConvBlock_v2(64, 64, kernel_size=3, stride=2, padding=1, act_type='prelu', norm_type=None),
            DeconvBlock_v2(64, 64,
                               kernel_size=6, stride=2, padding=2,
                               act_type='prelu', norm_type=None),

            ConvBlock_v2(64, 64, kernel_size=3, stride=2, padding=1, act_type='prelu', norm_type=None),
            DeconvBlock_v2(64, 64,
                               kernel_size=6, stride=2, padding=2,
                               act_type='prelu', norm_type=None),

            ConvBlock_v2(64, 64, kernel_size=3, stride=2, padding=1, act_type='prelu', norm_type=None),
            DeconvBlock_v2(64, 64,
                               kernel_size=6, stride=2, padding=2,
                               act_type='prelu', norm_type=None),

            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):

        inter_res = nn.functional.interpolate(x, scale_factor=self.scale_factor,
                                      mode='bilinear', align_corners=False)

        x = nn.functional.interpolate(x, scale_factor=self.scale_factor,
                                      mode='bilinear', align_corners=False)


        out = []
        for i in range(self.repeat):
            x = self.layers(x)
            if self.residual:
                x = torch.add(inter_res, x)
                inter_res = x.detach()
            out.append(x)
        return out


class RAPN(nn.Module):
    def __init__(self, scale_factor=4, repeat=10):
        super(RAPN, self).__init__()
        self.repeat = repeat
        self.scale_factor = scale_factor

        self.reconstructor = nn.Sequential(
            ConvBlock_v2(1, 32, kernel_size=3, stride=1, padding=1, act_type='prelu', norm_type=None),

            ConvBlock_v2(32, 32, kernel_size=3, stride=2, padding=1, act_type='prelu', norm_type=None),
            DeconvBlock_v2(32, 32,
                               kernel_size=6, stride=2, padding=2,
                               act_type='prelu', norm_type=None),

            ConvBlock_v2(32, 32, kernel_size=3, stride=2, padding=1, act_type='prelu', norm_type=None),
            DeconvBlock_v2(32, 32,
                               kernel_size=6, stride=2, padding=2,
                               act_type='prelu', norm_type=None),

            ConvBlock_v2(32, 32, kernel_size=3, stride=2, padding=1, act_type='prelu', norm_type=None),
            DeconvBlock_v2(32, 32,
                               kernel_size=6, stride=2, padding=2,
                               act_type='prelu', norm_type=None),

            ConvBlock_v2(32, 1, kernel_size=3, stride=1, padding=1, act_type=None, norm_type=None),
        )

        self.scale_regressor = nn.Sequential(
            ConvBlock_v2(repeat, 64, kernel_size=3, stride=1, padding=1, act_type='prelu', norm_type='bn'),
            nn.MaxPool2d(2),
            ConvBlock_v2(64, 64, kernel_size=3, stride=2, padding=1, act_type='prelu', norm_type='bn'),
            nn.MaxPool2d(2),
            ConvBlock_v2(64, 64, kernel_size=3, stride=2, padding=1, act_type='prelu', norm_type='bn'),
            nn.MaxPool2d(2),
            #ConvBlock_v2(64, 25, kernel_size=3, stride=2, padding=1, act_type=None, norm_type=None),
            ConvBlock_v2(64, 64, kernel_size=3, stride=2, padding=1, act_type='prelu', norm_type=None),
            nn.AdaptiveAvgPool2d(1),
            ConvBlock_v2(64, 64, kernel_size=1, stride=1, padding=0, act_type='prelu', norm_type=None),
            ConvBlock_v2(64, 1, kernel_size=1, stride=1, padding=0, act_type=None, norm_type=None),
            #nn.Softmax()
        )

        #self.scale_regressor = nn.Sequential(
        #    Flatten(),
        #    nn.Linear(64*64*3, 64),
        #    nn.PReLU(),
        #    nn.Linear(64, 64),
        #    nn.PReLU(),
        #    nn.Linear(64, 1),
        #)

    def forward(self, input_x, target_scales):
        target_scales = 1 / target_scales

        lrs = []
        for x, target_scale in zip(input_x, target_scales):
            x = x.unsqueeze(0)
            lr = nn.Upsample(mode='bilinear', scale_factor=target_scale)(x)
            lr = nn.Upsample(mode='bilinear', size=(
                input_x.shape[-2], input_x.shape[-1]))(lr)
            lrs.append(lr)

        lr = torch.cat(lrs, dim=0)

        #cx = int(input_x.shape[-2]/2)
        #cy = int(input_x.shape[-1]/2)

        #hr_patch = input_x[:,:, cx-32:cx+32, cy-32:cy+32]
        #hr_patch = input_x[:,:, :64,:64]
        #lr_patch = nn.UpsamplingBilinear2d(scale_factor=target_scale)(hr_patch)
        #lr_patch = nn.UpsamplingBilinear2d(size=(hr_patch.shape[-2], hr_patch.shape[-1]))(lr_patch)
        #lr_patch_fft = torch.rfft(lr_patch, 2, normalized=True, onesided=False).squeeze(1).transpose(2, 3).transpose(1,2)
        #lr_patch = torch.cat([lr_patch, lr_patch_fft], dim=1)

        inter_res = lr

        out = []
        for i in range(self.repeat):
            lr = self.reconstructor(lr)
            lr = torch.add(inter_res, lr)
            inter_res = lr #.detach()
            out.append(lr)

        lr = torch.cat(out, dim=1).detach()
        #pred_scale = self.scale_regressor(lr).squeeze().view(len(input_x), -1) # +2
        pred_scale = self.scale_regressor(lr).squeeze().view(len(input_x), -1) + 2.5

        return out, pred_scale


    def evaluate(self, input_x, target_scales):

        target_scales = 1 / target_scales

        lr = []
        for x, target_scale in zip(input_x, target_scales):

            lr = nn.UpsamplingBilinear2d(scale_factor=target_scale)(x)
            lr = nn.UpsamplingBilinear2d(size=
                (input_x.shape[-2], input_x.shape[-1]))(lr)
            lr.append(lr)

        lr = torch.cat(lr, dim=0)

        #hr_patch = input_x[:,:, :64, :64]
        #lr_patch = nn.UpsamplingBilinear2d(scale_factor=target_scale)(hr_patch)
        #lr_patch = nn.UpsamplingBilinear2d(size=hr_patch.shape[-1])(lr_patch)

        inter_res = lr

        out = []
        for i in range(self.repeat):
            lr = self.reconstructor(lr)
            lr = torch.add(inter_res, lr)
            inter_res = lr.detach()
            out.append(lr)

        ratio = np.power(self.scale_factor, 1/self.repeat)
        scales = np.array([1 / np.power(ratio, self.repeat - i - 1) for i in range(self.repeat)])

        pred_scales = self.scale_regressor(lr_patch)
        indexes = np.argmin(scales - target_scale)
        new_out = []
        for i, idx in enumerate(indexes):
            new_out.append(out[i][idx])
        new_out = torch.cat(new_out)
        new_out = new_out.view(*lr.shape)

        return new_out, pred_scale


class CRAPN(nn.Module):
    def __init__(self, scale_factor=4, repeat=10):
        super(CRAPN, self).__init__()
        self.repeat = repeat
        self.scale_factor = scale_factor

        self.reconstructor = nn.Sequential(
            ConvBlock_v2(3, 32, kernel_size=3, stride=1, padding=1, act_type='prelu', norm_type=None),

            ConvBlock_v2(32, 32, kernel_size=3, stride=2, padding=1, act_type='prelu', norm_type=None),
            DeconvBlock_v2(32, 32,
                               kernel_size=6, stride=2, padding=2,
                               act_type='prelu', norm_type=None),

            ConvBlock_v2(32, 32, kernel_size=3, stride=2, padding=1, act_type='prelu', norm_type=None),
            DeconvBlock_v2(32, 32,
                               kernel_size=6, stride=2, padding=2,
                               act_type='prelu', norm_type=None),

            ConvBlock_v2(32, 32, kernel_size=3, stride=2, padding=1, act_type='prelu', norm_type=None),
            DeconvBlock_v2(32, 32,
                               kernel_size=6, stride=2, padding=2,
                               act_type='prelu', norm_type=None),

            ConvBlock_v2(32, 1, kernel_size=3, stride=1, padding=1, act_type=None, norm_type=None),
        )

        self.scale_regressor = nn.Sequential(
            ConvBlock_v2(1, 64, kernel_size=3, stride=1, padding=1, act_type='prelu', norm_type='bn'),
            nn.MaxPool2d(2),
            ConvBlock_v2(64, 64, kernel_size=3, stride=2, padding=1, act_type='prelu', norm_type='bn'),
            nn.MaxPool2d(2),
            ConvBlock_v2(64, 64, kernel_size=3, stride=2, padding=1, act_type='prelu', norm_type='bn'),
            nn.MaxPool2d(2),
            #ConvBlock_v2(64, 25, kernel_size=3, stride=2, padding=1, act_type=None, norm_type=None),
            ConvBlock_v2(64, 64, kernel_size=3, stride=2, padding=1, act_type='prelu', norm_type=None),
            nn.AdaptiveAvgPool2d(1),
            ConvBlock_v2(64, 64, kernel_size=1, stride=1, padding=0, act_type='prelu', norm_type=None),
            ConvBlock_v2(64, 1, kernel_size=1, stride=1, padding=0, act_type=None, norm_type=None),
            #nn.Softmax()
        )

        #self.scale_regressor = nn.Sequential(
        #    Flatten(),
        #    nn.Linear(64*64*3, 64),
        #    nn.PReLU(),
        #    nn.Linear(64, 64),
        #    nn.PReLU(),
        #    nn.Linear(64, 1),
        #)

    def forward(self, input_x, target_scales):
        target_scales = 1 / target_scales
        batch_size = len(input_x)
        w = input_x.shape[-2]
        h = input_x.shape[-1]
        lrs = []
        for x, target_scale in zip(input_x, target_scales):
            x = x.unsqueeze(0)
            lr = nn.Upsample(mode='bilinear', scale_factor=target_scale)(x)
            lr = nn.Upsample(mode='bilinear', size=(w, h))(lr)
            lrs.append(lr)

        lr = torch.cat(lrs, dim=0)

        pred_scale = self.scale_regressor(lr).squeeze().view(batch_size, -1)

        upscaled_pred_scale = pred_scale.view((batch_size,1,1)).repeat((1,w,h)).reshape((batch_size, 1, w, h)).detach()


        out = []
        for i in range(self.repeat):
            normalized_stage = (i+1) / self.repeat
            upscaled_stage = torch.Tensor([normalized_stage]).repeat((batch_size, 1, w, h)).cuda()
            conditioned_input = torch.cat([lr, upscaled_pred_scale, upscaled_stage], dim=1) # conditioning scale and stage

            output = self.reconstructor(conditioned_input)
            lr = torch.add(lr, output)
            out.append(lr)


        return out, pred_scale + 2.5



class CRAPN_S(nn.Module):
    def __init__(self, scale_factor=4, repeat=10):
        super(CRAPN_S, self).__init__()
        self.repeat = repeat
        self.scale_factor = scale_factor

        self.reconstructor = nn.Sequential(
            ConvBlock_v2(1+self.repeat+16, 128, kernel_size=3, stride=1, padding=1, act_type='prelu', norm_type=None),
            ConvBlock_v2(128, 32, kernel_size=3, stride=1, padding=1, act_type='prelu', norm_type=None),

            ConvBlock_v2(32, 32, kernel_size=3, stride=2, padding=1, act_type='prelu', norm_type=None),
            DeconvBlock_v2(32, 32,
                               kernel_size=6, stride=2, padding=2,
                               act_type='prelu', norm_type=None),

            ConvBlock_v2(32, 32, kernel_size=3, stride=2, padding=1, act_type='prelu', norm_type=None),
            DeconvBlock_v2(32, 32,
                               kernel_size=6, stride=2, padding=2,
                               act_type='prelu', norm_type=None),

            ConvBlock_v2(32, 32, kernel_size=3, stride=2, padding=1, act_type='prelu', norm_type=None),
            DeconvBlock_v2(32, 32,
                               kernel_size=6, stride=2, padding=2,
                               act_type='prelu', norm_type=None),

            ConvBlock_v2(32, 1, kernel_size=3, stride=1, padding=1, act_type=None, norm_type=None),
        )

        self.scale_regressor = nn.Sequential(
            ConvBlock_v2(1, 16, kernel_size=3, stride=1, padding=1, act_type='relu', norm_type='bn'),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            ConvBlock_v2(16, 32, kernel_size=3, stride=2, padding=1, act_type='relu', norm_type='bn'),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            ConvBlock_v2(32, 64, kernel_size=3, stride=2, padding=1, act_type='relu', norm_type='bn'),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            #ConvBlock_v2(64, 25, kernel_size=3, stride=2, padding=1, act_type=None, norm_type=None),
            ConvBlock_v2(64, 128, kernel_size=3, stride=2, padding=1, act_type='relu', norm_type='bn'),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(0.25),
            ConvBlock_v2(128, 64, kernel_size=1, stride=1, padding=0, act_type='relu', norm_type=None),
            nn.Dropout(0.25),
            ConvBlock_v2(64, 16, kernel_size=1, stride=1, padding=0, act_type='relu', norm_type=None),
            nn.Dropout(0.25),
            ConvBlock_v2(16, 16, kernel_size=1, stride=1, padding=0, act_type=None, norm_type=None),
        )

        #self.scale_regressor = nn.Sequential(
        #    Flatten(),
        #    nn.Linear(64*64*3, 64),
        #    nn.PReLU(),
        #    nn.Linear(64, 64),
        #    nn.PReLU(),
        #    nn.Linear(64, 1),
        #)

    def down_and_up_sample(self, img_tensor, scales):
        ret = []
        batch_size = img_tensor.shape[0]
        w = img_tensor.shape[-2]
        h = img_tensor.shape[-1]
        for img, scale in zip(img_tensor, scales):
            img = img.view(1, 1, w, h)
            if scale < 0.1:
                scale = 0.25

            x = imresize(img.squeeze().cpu().numpy(), scalar_scale=scale)
            x = imresize(x, output_shape=(w, h))
            x = torch.Tensor(x).cuda()
            x = x.view(1,1,w,h)
            # x = nn.UpsamplingBilinear2d(scale_factor=scale)(img)
            # x = nn.UpsamplingBilinear2d(size=(w, h))(x)

            ret.append(x)
        ret = torch.cat(ret, dim=0)
        return ret


    def forward(self, input_x, target_scales):
        target_scales = 1 / target_scales
        batch_size = len(input_x)
        w = input_x.shape[-2]
        h = input_x.shape[-1]
        lrs = []
        lrs_for_scale_pred = []

        #print('x', target_scales[0])
        lr = self.down_and_up_sample(input_x, target_scales)
        pred_scale_raw = self.scale_regressor(lr).squeeze().view(batch_size, -1)
        pred_scale = nn.Softmax()(pred_scale_raw)
        T = 1/10 # temperature
        pred_scale_t = nn.Softmax()(pred_scale_raw / T)

        #upscaled_pred_scale = pred_scale.view((
        #    batch_size,1,1)).repeat((1,w,h)).reshape((batch_size, 1, w, h)).detach()

        #upscaled_pred_scale = pred_scale_t.view((
        #    batch_size,16,1,1)).repeat((1,1,w,h)).reshape((batch_size, 16, w, h)).detach()

        tt = torch.Tensor(1/target_scales.cpu() * 5 - 5).long()
        one_hot_target_scales = nn.functional.one_hot(tt, 16).float().cuda()
        upscaled_pred_scale = one_hot_target_scales.view((batch_size, -1)).view((
            batch_size,16,1,1)).repeat((1,1,w,h)).reshape((batch_size, 16, w, h)).detach()

        out = []
        #upscaled_stage_base = torch.zeros((batch_size, 1, w, h)).cuda()
        upscaled_stage_base = torch.zeros((batch_size, self.repeat, w, h)).cuda()
        for i in range(self.repeat):
            # normalized_stage = (i+1) / self.repeat
            # upscaled_stage = upscaled_stage_base + normalized_stage
            upscaled_stage_base[:,i] = 1
            upscaled_stage_base[:, np.arange(self.repeat) != i] = 0
            upscaled_stage = upscaled_stage_base
            conditioned_input = torch.cat([lr, upscaled_pred_scale, upscaled_stage], dim=1) # conditioning scale and stage

            #a = self.reconstructor[:2](conditioned_input)
            #a = self.reconstructor[2:4](a) + a
            #a = self.reconstructor[4:6](a) + a
            #a = self.reconstructor[6:8](a) + a
            #output = self.reconstructor[8](a)

            output = self.reconstructor(conditioned_input)
            lr = output + lr.detach()
            out.append(lr)


        #return out, pred_scale * 10 + self.repeat/2 + 0.5
        return out, pred_scale


    def inference(self, input_x):
        batch_size = len(input_x)
        T = 1/10
        #for i in range(40):
        #    x = input_x.shape[-2]
        #    y = input_x.shape[-1]
        #    x = np.random.randint(0, x-63)
        #    y = np.random.randint(0, y-63)
        #    if i == 0:
        #        pred_scale = self.scale_regressor(input_x[:,:,x:x+64,y:y+64]).squeeze().view(batch_size, -1)
        #        pred_scale = nn.Softmax()(pred_scale/T)
        #    else:
        #        pred_scale += nn.Softmax()(
        #            self.scale_regressor(input_x[:,:,x:x+64,y:y+64]).squeeze().view(batch_size, -1)/T
        #        )
        #pred_scale = pred_scale / 40

        pred_scale_raw = self.scale_regressor(input_x).squeeze().view(batch_size, -1)
        pred_scale = nn.Softmax()(pred_scale_raw)
        pred_scale = nn.Softmax()(pred_scale_raw / T)

        #pred_scale = self.scale_regressor(input_x).squeeze().view(batch_size, -1)

        repeat_list = []
        ratio = np.power(self.scale_factor, 1/self.repeat)
        for ps in pred_scale:
            ps = torch.argmax(ps).cpu().numpy() / 5 + 1
            repeat = 1
            for i in range(self.repeat):
                if ps / np.power(ratio, i+1) > 1:
                    repeat += 1
            repeat_list.append(repeat)

        out = []

        for x, repeat, ps in zip(input_x, repeat_list, pred_scale):

            w = x.shape[-2]
            h = x.shape[-1]
            upscaled_pred_scale = ps.view((
                1,16,1,1)).repeat((1,1,w,h)).reshape((1, 16, w, h)).detach()
            upscaled_stage_base = torch.zeros((1, self.repeat, w, h)).cuda()

            x = x.unsqueeze(0)

            for i in range(repeat):
                upscaled_stage_base[:,i] = 1
                upscaled_stage_base[:,np.arange(repeat) != i] = 0
                upscaled_stage = upscaled_stage_base
                conditioned_input = torch.cat([x,
                                               upscaled_pred_scale, upscaled_stage], dim=1) # conditioning scale and stage

                #a = self.reconstructor[:2](conditioned_input)
                #a = self.reconstructor[2:4](a) + a
                #a = self.reconstructor[4:6](a) + a
                #a = self.reconstructor[6:8](a) + a
                #output = self.reconstructor[8](a)
                output = self.reconstructor(conditioned_input)
                # x = x + output
                x = x.detach() + output
            out.append(x)

        out = torch.cat(out, dim=0)


        #return out, pred_scale * 10 + self.repeat/2 + 0.5
        return out, pred_scale


class USRN(nn.Module):
    def __init__(self, scale_factor=4, repeat=10):
        super(USRN, self).__init__()
        self.repeat = repeat
        self.scale_factor = scale_factor

        self.reconstructor = nn.Sequential(
            ConvBlock_v2(1+self.repeat+16, 128, kernel_size=3, stride=1, padding=1, act_type='prelu', norm_type=None),
            ConvBlock_v2(128, 32, kernel_size=3, stride=1, padding=1, act_type='prelu', norm_type=None),

            ConvBlock_v2(32, 32, kernel_size=3, stride=2, padding=1, act_type='prelu', norm_type=None),
            DeconvBlock_v2(32, 32,
                               kernel_size=6, stride=2, padding=2,
                               act_type='prelu', norm_type=None),

            ConvBlock_v2(32, 32, kernel_size=3, stride=2, padding=1, act_type='prelu', norm_type=None),
            DeconvBlock_v2(32, 32,
                               kernel_size=6, stride=2, padding=2,
                               act_type='prelu', norm_type=None),

            ConvBlock_v2(32, 32, kernel_size=3, stride=2, padding=1, act_type='prelu', norm_type=None),
            DeconvBlock_v2(32, 32,
                               kernel_size=6, stride=2, padding=2,
                               act_type='prelu', norm_type=None),

            ConvBlock_v2(32, 1, kernel_size=3, stride=1, padding=1, act_type=None, norm_type=None),
        )

        self.scale_regressor = nn.Sequential(
            ConvBlock_v2(1, 16, kernel_size=3, stride=1, padding=1, act_type='relu', norm_type='bn'),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            ConvBlock_v2(16, 32, kernel_size=3, stride=2, padding=1, act_type='relu', norm_type='bn'),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            ConvBlock_v2(32, 64, kernel_size=3, stride=2, padding=1, act_type='relu', norm_type='bn'),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            #ConvBlock_v2(64, 25, kernel_size=3, stride=2, padding=1, act_type=None, norm_type=None),
            ConvBlock_v2(64, 128, kernel_size=3, stride=2, padding=1, act_type='relu', norm_type='bn'),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(0.25),
            ConvBlock_v2(128, 64, kernel_size=1, stride=1, padding=0, act_type='relu', norm_type=None),
            nn.Dropout(0.25),
            ConvBlock_v2(64, 16, kernel_size=1, stride=1, padding=0, act_type='relu', norm_type=None),
            nn.Dropout(0.25),
            ConvBlock_v2(16, 16, kernel_size=1, stride=1, padding=0, act_type=None, norm_type=None),
        )

        #self.scale_regressor = nn.Sequential(
        #    Flatten(),
        #    nn.Linear(64*64*3, 64),
        #    nn.PReLU(),
        #    nn.Linear(64, 64),
        #    nn.PReLU(),
        #    nn.Linear(64, 1),
        #)

    def down_and_up_sample(self, img_tensor, scales):
        ret = []
        batch_size = img_tensor.shape[0]
        w = img_tensor.shape[-2]
        h = img_tensor.shape[-1]
        for img, scale in zip(img_tensor, scales):
            img = img.view(1, 1, w, h)
            if scale < 0.1:
                scale = 0.1

            x = imresize(img.squeeze().cpu().numpy(), scalar_scale=scale)
            x = imresize(x, output_shape=(w, h))
            x = torch.Tensor(x).cuda()
            x = x.view(1,1,w,h)
            # x = nn.UpsamplingBilinear2d(scale_factor=scale)(img)
            # x = nn.UpsamplingBilinear2d(size=(w, h))(x)

            ret.append(x)
        ret = torch.cat(ret, dim=0)
        return ret


    def forward(self, input_x, target_scales):
        target_scales = 1 / target_scales
        batch_size = len(input_x)
        w = input_x.shape[-2]
        h = input_x.shape[-1]
        lrs = []
        lrs_for_scale_pred = []

        #print('x', target_scales[0])
        lr = self.down_and_up_sample(input_x, target_scales)
        pred_scale_raw = self.scale_regressor(lr).squeeze().view(batch_size, -1)
        pred_scale = nn.Softmax()(pred_scale_raw)
        T = 1/10 # temperature
        pred_scale_t = nn.Softmax()(pred_scale_raw / T)

        #upscaled_pred_scale = pred_scale.view((
        #    batch_size,1,1)).repeat((1,w,h)).reshape((batch_size, 1, w, h)).detach()

        #upscaled_pred_scale = pred_scale_t.view((
        #    batch_size,16,1,1)).repeat((1,1,w,h)).reshape((batch_size, 16, w, h)).detach()

        tt = torch.Tensor(1/target_scales.cpu() * 5 - 5).long()
        one_hot_target_scales = nn.functional.one_hot(tt, 16).float().cuda()
        upscaled_pred_scale = one_hot_target_scales.view((batch_size, -1)).view((
            batch_size,16,1,1)).repeat((1,1,w,h)).reshape((batch_size, 16, w, h)).detach()

        out = []
        #upscaled_stage_base = torch.zeros((batch_size, 1, w, h)).cuda()
        upscaled_stage_base = torch.zeros((batch_size, self.repeat, w, h)).cuda()
        for i in range(self.repeat):
            # normalized_stage = (i+1) / self.repeat
            # upscaled_stage = upscaled_stage_base + normalized_stage
            upscaled_stage_base[:,i] = 1
            for j in range(self.repeat):
                if j == i:
                    continue
                upscaled_stage_base[:, j] = 0
            conditioned_input = torch.cat([lr, upscaled_pred_scale, upscaled_stage_base], dim=1) # conditioning scale and stage

            #a = self.reconstructor[:2](conditioned_input)
            #a = self.reconstructor[2:4](a) + a
            #a = self.reconstructor[4:6](a) + a
            #a = self.reconstructor[6:8](a) + a
            #output = self.reconstructor[8](a)

            output = self.reconstructor(conditioned_input)
            lr = output + lr.detach()
            out.append(lr)


        #return out, pred_scale * 10 + self.repeat/2 + 0.5
        return out, pred_scale


    def inference(self, input_x):
        batch_size = len(input_x)
        T = 1/10
        #for i in range(40):
        #    x = input_x.shape[-2]
        #    y = input_x.shape[-1]
        #    x = np.random.randint(0, x-63)
        #    y = np.random.randint(0, y-63)
        #    if i == 0:
        #        pred_scale = self.scale_regressor(input_x[:,:,x:x+64,y:y+64]).squeeze().view(batch_size, -1)
        #        pred_scale = nn.Softmax()(pred_scale/T)
        #    else:
        #        pred_scale += nn.Softmax()(
        #            self.scale_regressor(input_x[:,:,x:x+64,y:y+64]).squeeze().view(batch_size, -1)/T
        #        )
        #pred_scale = pred_scale / 40

        pred_scale_raw = self.scale_regressor(input_x).squeeze().view(batch_size, -1)
        #pred_scale = nn.Softmax()(pred_scale_raw)
        pred_scale = nn.Softmax()(pred_scale_raw / T)

        #pred_scale = self.scale_regressor(input_x).squeeze().view(batch_size, -1)

        #repeat_list = []
        #ratio = np.power(self.scale_factor, 1/self.repeat)
        #for ps in pred_scale:
        #    ps = torch.argmax(ps).cpu().numpy() / 5 + 1
        #    repeat = 1
        #    for i in range(self.repeat):
        #        if ps / np.power(ratio, i+1) > 1:
        #            repeat += 1
        #    repeat_list.append(repeat)
        out = []

        for x, ps in zip(input_x, pred_scale):

            w = x.shape[-2]
            h = x.shape[-1]
            upscaled_pred_scale = ps.view((
                1,16,1,1)).repeat((1,1,w,h)).reshape((1, 16, w, h)).detach()
            upscaled_stage_base = torch.zeros((1, self.repeat, w, h)).cuda()

            x = x.unsqueeze(0)

            for i in range(self.repeat):
                upscaled_stage_base[:,i] = 1
                for j in range(self.repeat):
                    if j == i:
                        continue
                    upscaled_stage_base[:, j] = 0
                conditioned_input = torch.cat([x,
                                               upscaled_pred_scale, upscaled_stage_base], dim=1) # conditioning scale and stage

                output = self.reconstructor(conditioned_input)
                # x = x + output
                x = x.detach() + output
            out.append(x)

        out = torch.cat(out, dim=0)


        #return out, pred_scale * 10 + self.repeat/2 + 0.5
        return out, pred_scale


def get_fsrcnn(scale_factor=4, **kwargs):
    return FSRCNN(scale_factor=scale_factor, **kwargs)


def get_drrn(scale_factor=4, **kwargs):
    return DRRN(scale_factor=scale_factor, **kwargs)


def get_dbpn(scale_factor=4, **kwargs):
    return DBPN(scale_factor=scale_factor, **kwargs)


def get_srfbn(scale_factor=4, **kwargs):
    return SRFBN(scale_factor=scale_factor, **kwargs)


def get_psrn(scale_factor=4, **kwargs):
    return PSRN(scale_factor=scale_factor, **kwargs)


def get_psrn_s(scale_factor=4, **kwargs):
    return PSRN_S(scale_factor=scale_factor, **kwargs)


def get_psrn_l(scale_factor=4, **kwargs):
    return PSRN_L(scale_factor=scale_factor, **kwargs)


def get_psrn_d(scale_factor=4, **kwargs):
    return PSRN_D(scale_factor=scale_factor, **kwargs)


def get_psrn_dl(scale_factor=4, **kwargs):
    return PSRN_DL(scale_factor=scale_factor, **kwargs)


def get_rapn(scale_factor=4, **kwargs):
    return RAPN(scale_factor=scale_factor, **kwargs)


def get_crapn(scale_factor=4, **kwargs):
    return CRAPN(scale_factor=scale_factor, **kwargs)

def get_crapn_s(scale_factor=4, **kwargs):
    return CRAPN_S(scale_factor=scale_factor, **kwargs)


def get_usrn(scale_factor=4, **kwargs):
    return USRN(scale_factor=scale_factor, **kwargs)



def get_model(config):
    print('model name:', config.model.name)
    f = globals().get('get_' + config.model.name)
    if config.model.params is None:
        return f()
    else:
        return f(**config.model.params)


from scipy import fftpack

def get_circle_bandpass_filter_mask(shape, ratio=0.5, offset=0.2):
    default_value = 0.1

    cx = shape[0]//2
    cy = shape[1]//2

    d1 = (shape[0] * (ratio + offset)) / 2
    d2 = (shape[0] * offset) / 2
    # d1 = np.sqrt(ratio * np.power(shape[0],2)/4 + np.power(d2, 2))
    mask = np.zeros(shape=shape) + default_value
    for i in range(shape[0]):
        for j in range(shape[1]):
            if np.power(d2, 2) <= np.power((i-cy), 2) + np.power((j-cx), 2) <= np.power(d1,2):
                mask[i, j] = 1
    return mask


def apply_filter(im, mask):
    im_fft = fftpack.fft2(im)
    im_fft2 = im_fft.copy()

    im_fft2 = fftpack.fftshift(im_fft2)
    im_fft2 = im_fft2 * mask
    im_fft2 = fftpack.ifftshift(im_fft2)
    im_new = fftpack.ifft2(im_fft2).real

    return im_new, im_fft2


def highpass_transform_fn(im):
    ratio = 0.9
    offset = 0.2
    mask = get_circle_bandpass_filter_mask(im.shape, ratio, offset)
    mask = 1 - mask
    im_new, im_fft2 = apply_filter(im, mask)

    return im_new


