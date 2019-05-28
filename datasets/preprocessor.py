import sys
sys.path.append('../')

import glob
from skimage.io import imread
import skimage
from utils.imresize import imresize
import numpy as np


# Augmentation setting

def load_img(filepath, is_gray):
    img = imread(filepath)
    ch = img.ndim
    if ch == 2:
        img = skimage.color.gray2rgb(img)
    if is_gray:
        img = skimage.color.rgb2ycbcr(img)
    return img / 255.


def preprocess(directory, save_npy_path, patch_size=64, stride=64,
               scale=[1, 0.9, 0.8, 0.7, 0.6, 2], rotation=[0, 1, 2, 3],
               flip=[True, False], is_gray=True):
    HR_SET = []
    images = sorted(glob.glob(directory + "*.png"))
    print("The number of training images : ",len(images))

    for idx in range(len(images)):
        print("\r Processing ", idx+1," / ",len(images), end = '')
        image_directory = images[idx]
        for f in flip:
            for r in rotation:
                for s in scale:
                     # load image
                    image = load_img(image_directory, is_gray) # is_gary : YCbCr or RGB
                     # flipping
                    if f:
                        image = np.fliplr(image)
                     # rotation
                    image = np.rot90(image, k=r, axes=(0,1))
                     # scaling
                    image = imresize(image, scalar_scale = s)
                    image = image.clip(0,1)
                    # generate HR patch
                    h,w,_ = image.shape
                    for i in range(0, h-patch_size, stride):
                        for j in range(0, w-patch_size, stride):
                            hr_patch = image[i:i+patch_size, j:j+patch_size, :]
                            if is_gray:
                                hr_patch = hr_patch[:,:,0]
                            HR_SET.append(hr_patch)

    print("\nThe number of training patches : ",len(HR_SET))
    np.save(save_npy_path, HR_SET)
    print("Training patches are successfully saved")
