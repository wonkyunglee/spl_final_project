from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os

import numpy as np
from .bandpass_transform import bandpass_transform


def get_transform(config):

    if config.transform.name is None:
        return None
    else:
        f = globals().get(config.transform.name)
        return f(scale_fatcor=config.data.scale_factor,
                 **config.transform.params)

