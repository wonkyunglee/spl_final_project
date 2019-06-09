from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os

import numpy as np
import torch.utils.data
import torch.utils.data.sampler
from torch.utils.data import DataLoader

from .dataset import T91, Set5, Set14, Manga109, BSDS100, Urban100


def get_train_dataset(config, transform):
    name = config.data.train.name
    f = globals().get(name)
    return f(data_dir=config.data.train.data_dir,
             scale_factor=config.data.scale_factor,
             upsample_LR_patch=config.data.upsample_LR_patch,
             transform=transform,
             **config.data.train.params)


def get_valid_dataset(config, idx):
    name = config.data.valid.name[idx]
    f = globals().get(name)
    data_dir = os.path.join(config.data.valid.base_dir, name)
    return f(data_dir=data_dir,
             scale_factor=config.data.scale_factor,
             upsample_LR_patch=config.data.upsample_LR_patch,
             **config.data.valid.params)


def get_train_dataloader(config, transform):
    dataset = get_train_dataset(config, transform)
    batch_size = config.train.batch_size
    dataloader = DataLoader(dataset,
                            shuffle=True,
                            batch_size=batch_size,
                            drop_last=True)
    return dataloader


def get_valid_dataloaders(config):
    batch_size = config.eval.batch_size
    dataloaders = []
    for idx, name in enumerate(config.data.valid.name):
        dataset = get_valid_dataset(config, idx)
        dataloader = DataLoader(dataset,
                                shuffle=False,
                                batch_size=batch_size,
                                drop_last=False)
        dataloaders.append(dataloader)
    return dataloaders


# T91_directory = "../../dataset/taskwise/super_resolution/T91/"
# T91_npy_path = './T91_HR_64_EEE4423.npy'

# Set5_directory = '../../dataset/taskwise/super_resolution/Set5/'
# Set14_directory = '../../dataset/taskwise/super_resolution/Set14/'
# Manga109_directory = '../../dataset/taskwise/super_resolution/Manga109/'
# BSDS100_directory = '../../dataset/taskwise/super_resolution/BSDS100/'
# Urban100_directory = '../../dataset/taskwise/super_resolution/Urban100/'



# define dataset loader
# scale_factor = 4 # 1/4 down scaling

# # Train set
# # T91
# train_dataset = T91_images(data_dir=T91_npy_path, HR_patch_size = 64,
#                            scale_factor=scale_factor,
#                            upsample_LR_patch=upsample_LR_patch)

# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=64,
#                                            shuffle=True,
#                                            num_workers = 4)
# # Valid set
# # Set5
# test_Set5_dataset = Set5(data_dir=Set5_directory,
#                          scale_factor=scale_factor,
#                          upsample_LR_patch=upsample_LR_patch)
# test_Set5_loader = torch.utils.data.DataLoader(dataset=test_Set5_dataset,
#                                                batch_size=1,
#                                                shuffle=False)
# # Set14
# test_Set14_dataset = Set14(data_dir=Set14_directory,
#                          scale_factor=scale_factor,
#                          upsample_LR_patch=upsample_LR_patch)
# test_Set14_loader = torch.utils.data.DataLoader(dataset=test_Set14_dataset,
#                                                batch_size=1,
#                                                shuffle=False)
# # Manga109
# test_Manga109_dataset = Set5(data_dir=Manga109_directory,
#                          scale_factor=scale_factor,
#                          upsample_LR_patch=upsample_LR_patch)
# test_Manga109_loader = torch.utils.data.DataLoader(dataset=test_Manga109_dataset,
#                                                batch_size=1,
#                                                shuffle=False)
# # BSDS100
# test_BSDS100_dataset = Set5(data_dir=BSDS100_directory,
#                          scale_factor=scale_factor,
#                          upsample_LR_patch=upsample_LR_patch)
# test_BSDS100_loader = torch.utils.data.DataLoader(dataset=test_BSDS100_dataset,
#                                                batch_size=1,
#                                                shuffle=False)
# # Urban100
# test_Urban100_dataset = Set5(data_dir=Urban100_directory,
#                          scale_factor=scale_factor,
#                          upsample_LR_patch=upsample_LR_patch)
# test_Urban100_loader = torch.utils.data.DataLoader(dataset=test_Urban100_dataset,
#                                                batch_size=1,
#                                                shuffle=False)
