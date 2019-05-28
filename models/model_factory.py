import torch
import torch.nn as nn
import math

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


def get_fsrcnn(scale_factor=4, **kwargs):
    return FSRCNN(scale_factor, **kwargs)


def get_drrn(scale_factor=4, **kwargs):
    return DRRN(scale_factor, **kwargs)


def get_model(config):
    print('model name:', config.model.name)
    f = globals().get('get_' + config.model.name)
    if config.model.params is None:
        return f()
    else:
        return f(**config.model.params)








