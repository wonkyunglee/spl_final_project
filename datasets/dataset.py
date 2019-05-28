import os, sys
sys.path.append('../')

from skimage.io import imread
import skimage
import numpy as np
from utils.imresize import imresize
import torch
from torch.utils.data.dataset import Dataset
from .preprocessor import preprocess


class TrainSet(Dataset):


    def __init__(self, data_dir, original_data_dir,
                 HR_patch_size=64, scale_factor=4,
                 upsample_LR_patch=True, patch_size=64, stride=32):
        super(TrainSet, self).__init__()

        if not os.path.exists(data_dir):
            preprocess(original_data_dir, data_dir, patch_size, stride)

        self.HR_patches_np = np.load(data_dir) # pre-processed patches
        self.HR_patch_size = HR_patch_size
        self.scale_factor = scale_factor
        self.upsample_LR_patch = upsample_LR_patch


    def __getitem__(self, idx):
        HR_patch_np = self.HR_patches_np[idx] # high resolution patch
        LR_patch_np = imresize(HR_patch_np, scalar_scale = 1.0 / self.scale_factor) # low resolution patch
        BC_patch_np = imresize(LR_patch_np, output_shape=HR_patch_np.shape[-2:]) # bicubic upsampled patch

        if self.upsample_LR_patch:
            LR_patch_np = BC_patch_np

        HR_patch = torch.from_numpy(HR_patch_np).type(torch.FloatTensor)
        LR_patch = torch.from_numpy(LR_patch_np).type(torch.FloatTensor)
        BC_patch = torch.from_numpy(BC_patch_np).type(torch.FloatTensor)


        HR_patch = HR_patch.unsqueeze(0) # size : 1(c) x 64(h) x 64(w)
        LR_patch = LR_patch.unsqueeze(0) # size : 1(c) x 16(h) x 16(w)
        BC_patch = BC_patch.unsqueeze(0) # size : 1(c) x 64(h) x 64(w)

        return HR_patch, LR_patch, BC_patch # Y-channel patches


    def __len__(self):
        return len(self.HR_patches_np)


class ValidSet(Dataset):
    def __init__(self, data_dir, scale_factor=4, upsample_LR_patch=False):
        super(ValidSet, self).__init__()
        self.image_filenames = [os.path.join(data_dir, x) for x in sorted(os.listdir(data_dir))]
        self.scale_factor = scale_factor
        self.upsample_LR_patch = upsample_LR_patch


    def load_img(self, filepath):
        img = skimage.io.imread(filepath)
        ch = img.ndim
        if ch == 2:
            img = skimage.color.gray2rgb(img)
        img = skimage.color.rgb2ycbcr(img)

        return img / 255.

    def calculate_valid_crop_size(self, crop_size, scale_factor):
        return crop_size - (crop_size % scale_factor)

    def __getitem__(self, idx):
        # load image
        img_np = self.load_img(self.image_filenames[idx])

        # original HR image size
        h, w, _ = img_np.shape

        # determine valid HR image size with scale factor
        HR_img_w = self.calculate_valid_crop_size(w, self.scale_factor)
        HR_img_h = self.calculate_valid_crop_size(h, self.scale_factor)

        # determine lr_img LR image size
        LR_img_w = HR_img_w // self.scale_factor
        LR_img_h = HR_img_h // self.scale_factor

        HR_img_np = img_np[:HR_img_h,:HR_img_w,:] # high resolution image
        LR_img_np = imresize(HR_img_np, scalar_scale = 1.0/self.scale_factor) # low resolution image
        BC_img_np = imresize(LR_img_np, scalar_scale = self.scale_factor) # bicubic upsampled image
        if self.upsample_LR_patch:
            LR_img_np = BC_img_np


        HR_img = torch.from_numpy(HR_img_np).type(torch.FloatTensor).permute(2,0,1) # size : 3(c) x h x w
        LR_img = torch.from_numpy(LR_img_np).type(torch.FloatTensor).permute(2,0,1) # size : 3(c) x (h/scale_factor) x (w/scale_factor)
        BC_img = torch.from_numpy(BC_img_np).type(torch.FloatTensor).permute(2,0,1) # size : 3(c) x h x w

        return HR_img, LR_img, BC_img # YCbCr images


    def __len__(self):
        return len(self.image_filenames)


class T91(TrainSet):
    def __init__(self, data_dir, original_data_dir,
                 HR_patch_size=64, scale_factor=4,
                 upsample_LR_patch=True, **_):
        super(T91, self).__init__(data_dir, original_data_dir,
                                  HR_patch_size, scale_factor,
                                  upsample_LR_patch, **_)


class Set5(ValidSet):
    def __init__(self, data_dir,
                 scale_factor=4, upsample_LR_patch=False):
        super(Set5, self).__init__(data_dir, scale_factor, upsample_LR_patch)


class Set14(ValidSet):
    def __init__(self, data_dir,
                 scale_factor=4, upsample_LR_patch=False):
        super(Set14, self).__init__(data_dir, scale_factor, upsample_LR_patch)


class Manga109(ValidSet):
    def __init__(self, data_dir,
                 scale_factor=4, upsample_LR_patch=False):
        super(Manga109, self).__init__(data_dir, scale_factor, upsample_LR_patch)


class BSDS100(ValidSet):
    def __init__(self, data_dir,
                 scale_factor=4, upsample_LR_patch=False):
        super(BSDS100, self).__init__(data_dir, scale_factor, upsample_LR_patch)


class Urban100(ValidSet):
    def __init__(self, data_dir,
                 scale_factor=4, upsample_LR_patch=False):
        super(Urban100, self).__init__(data_dir, scale_factor, upsample_LR_patch)

