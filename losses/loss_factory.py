from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


def mse_loss(**_):
    return torch.nn.MSELoss()


def get_loss(config):
    f = globals().get(config.loss.name)
    return f(**config.loss.params)

