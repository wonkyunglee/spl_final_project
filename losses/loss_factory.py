from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import UpsamplingBilinear2d
import sys
sys.path.append('../')
from utils.imresize import imresize

import numpy as np


def multi_mse_loss(reduction='sum', **_):
    def mse_loss_fn(pred, target):
        if type(pred) == list:
            loss = 0
            for p in pred:
                loss += torch.nn.MSELoss(reduction=reduction)(p, target)
        else:
            loss = torch.nn.MSELoss(reduction=reduction)(pred, target)
        return loss
    return mse_loss_fn


def mse_loss(reduction='sum', **_):
    return torch.nn.MSELoss(reduction=reduction)


def get_loss(config):
    f = globals().get(config.loss.name)
    return f(**config.loss.params)


def psrn_loss(reduction='sum', scale_factor=4, **_):
    def loss_fn(pred, target):
        if type(pred) == list:
            loss = 0
            length = len(pred)
            ratio = np.power(scale_factor, 1/length)
            for i, p in enumerate(pred):
                scale = 1 / np.power(ratio, length - i - 1)
                t = UpsamplingBilinear2d(scale_factor=scale)(target)
                t = UpsamplingBilinear2d(size=target.shape[-1])(t)

                loss += (length - i) * torch.nn.MSELoss(reduction=reduction)(p, t)
        else:
            loss = torch.nn.MSELoss(reduction=reduction)(pred, target)
        return loss
    return loss_fn


def rapn_loss(reduction='mean', scale_factor=4, **_):

    def loss_fn(pred_imgs, pred_scale, target_imgs, target_scales):

        batch_size = len(target_scales)
        # pred_imgs : [repeat, batch_size, 1, w, h]
        # target_scales : [batch_size]
        if type(pred_imgs) == list:
            reconstruction_loss = 0
            length = len(pred_imgs)
            ratio = np.power(scale_factor, 1/length)
            scale_v = [np.power(ratio, length - i - 1) for i in range(length)]

            t_s_v = [[] for i in range(batch_size)]
            for i, ts in enumerate(target_scales):
                for j  in range(length):
                    tmp_ts = ts / np.power(ratio, j+1)
                    t_s_v[i].append(tmp_ts)
                    if tmp_ts < 1:
                        break

            for i, pred in enumerate(pred_imgs):
                for j, p in enumerate(pred):

                    if i >= len(t_s_v[j]):
                        break
                    scale = 1/t_s_v[j][i]
                    target_img = target_imgs[j].unsqueeze(0)
                    p = p.unsqueeze(0)

                    #t = UpsamplingBilinear2d(scale_factor=scale)(target_img)
                    #t = UpsamplingBilinear2d(size=(
                    #    target_img.shape[-2], target_img.shape[-1]))(t)

                    target_img = target_img.squeeze(0)
                    #if j == 0:
                    #    print(1/target_scales[j] , scale)
                    t = imresize(target_img.cpu().numpy(), scalar_scale=scale)
                    t = imresize(t, output_shape=(target_img.shape[-2], target_img.shape[-1]))
                    t = torch.Tensor(t).cuda().unsqueeze(0).unsqueeze(0)

                    reconstruction_loss += \
                        np.sqrt(length - i) * torch.nn.MSELoss(reduction=reduction)(p, t)
        else:
            reconstruction_loss = \
                torch.nn.MSELoss(reduction=reduction)(pred_imgs, target_imgs)

        target_scales = target_scales.view(batch_size, -1)

        #scale_loss = torch.nn.MSELoss(reduction='mean')(pred_scale, target_scales)
        one_hot_target_scales = nn.functional.one_hot((target_scales * 5 - 5).long(), 16).float()
        scale_loss = torch.nn.BCELoss(reduction='mean')(pred_scale, one_hot_target_scales.cuda())
        total_loss = reconstruction_loss + 10 * scale_loss
        return total_loss

    return loss_fn


def usrn_loss(reduction='mean', scale_factor=4, **_):

    def loss_fn(pred_imgs, pred_scale, target_imgs, target_scales):

        batch_size = len(target_scales)
        # pred_imgs : [repeat, batch_size, 1, w, h]
        # target_scales : [batch_size]
        if type(pred_imgs) == list:
            reconstruction_loss = 0
            length = len(pred_imgs)
            ratio = np.power(scale_factor, 1/length)
            scale_v = [np.power(ratio, length - i - 1) for i in range(length)]

            #t_s_v = [[] for i in range(batch_size)]
            #for i, ts in enumerate(target_scales):
            #    for j  in range(length):
            #        tmp_ts = ts / np.power(ratio, j+1)
            #        t_s_v[i].append(tmp_ts)
            #        if tmp_ts < 1:
            #            break

            for i, pred in enumerate(pred_imgs):
                for j, p in enumerate(pred):

                    #if i >= len(t_s_v[j]):
                    #    break
                    #scale = 1/t_s_v[j][i]
                    target_img = target_imgs[j].unsqueeze(0)
                    p = p.unsqueeze(0)

                    #t = UpsamplingBilinear2d(scale_factor=scale)(target_img)
                    #t = UpsamplingBilinear2d(size=(
                    #    target_img.shape[-2], target_img.shape[-1]))(t)

                    #target_img = target_img.squeeze(0)
                    #if j == 0:
                    #    print(1/target_scales[j] , scale)
                    #t = imresize(target_img.cpu().numpy(), scalar_scale=scale)
                    #t = imresize(t, output_shape=(target_img.shape[-2], target_img.shape[-1]))
                    #t = torch.Tensor(t).cuda().unsqueeze(0).unsqueeze(0)

                    t = target_img
                    reconstruction_loss += \
                        np.sqrt(i+1) * torch.nn.MSELoss(reduction=reduction)(p, t)
        else:
            reconstruction_loss = \
                torch.nn.MSELoss(reduction=reduction)(pred_imgs, target_imgs)

        target_scales = target_scales.view(batch_size, -1)

        #scale_loss = torch.nn.MSELoss(reduction='mean')(pred_scale, target_scales)
        one_hot_target_scales = nn.functional.one_hot((target_scales * 5 - 5).long(), 16).float()
        scale_loss = torch.nn.BCELoss(reduction='mean')(pred_scale, one_hot_target_scales.cuda())
        total_loss = reconstruction_loss + 10 * scale_loss
        return total_loss

    return loss_fn


