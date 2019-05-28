
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
from easydict import EasyDict as edict


def _get_default_config():
    c = edict()

    # dataset
    c.data = edict()
    c.data.train = edict()
    c.data.train.name = 'T91'
    c.data.train.data_dir: './preprocessed/T91_HR_64_EEE4423.npy'
    c.data.train.params = edict()
    c.data.valid = edict()
    c.data.valid.params = edict()

    # model
    c.model = edict()
    c.model.params = edict()

    # train
    c.train = edict()
    c.train.batch_size = 64
    c.train.num_epochs = 50
    c.train.num_grad_acc = None
    c.train.params = edict()


    # evaluation
    c.eval = edict()
    c.eval.batch_size = 1
    c.eval.params = edict()

    # optimizer
    c.optimizer = edict()
    c.optimizer.name = 'adam'
    c.optimizer.params = edict()

    # scheduler
    c.scheduler = edict()
    c.scheduler.name = 'none'
    c.scheduler.params = edict()

    # losses
    c.loss = edict()
    c.loss.name = None
    c.loss.params = edict()

    return c


def _merge_config(src, dst):
    if not isinstance(src, edict):
        return

    for k, v in src.items():
        if isinstance(v, edict):
            print('*'*25)
            print(k, v)
            _merge_config(src[k], dst[k])
        else:
            dst[k] = v


def load(config_path):
    with open(config_path, 'r') as fid:
        yaml_config = edict(yaml.load(fid))

    config = _get_default_config()
    _merge_config(yaml_config, config)

    return config
