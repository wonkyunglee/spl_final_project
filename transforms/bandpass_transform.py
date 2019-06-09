from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import fftpack
import numpy as np

def bandpass_transform(scale_factor=4, transform_ratio=0.1, **_):

    if scale_factor == 2:
        ratio = 0.45
    elif scale_factor == 3:
        ratio = 0.26
    elif scale_factor == 4:
        ratio = 0.21

    offset_candidates = [0.0, 0.05, 0.1]
    transform_ratio = transform_ratio

    def transform(im):

        if np.random.rand() < transform_ratio :
            offset = np.random.choice(offset_candidates)
            im = get_circle_bandpass_filtered_img(im, ratio=ratio, offset=offset)
        return im

    return transform


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


def get_circle_bandpass_filtered_img(img, ratio=0.5, offset=0.2):
    im_new, im_fft2 = apply_filter(img, get_circle_bandpass_filter_mask(img.shape, ratio, offset))
    return im_new

