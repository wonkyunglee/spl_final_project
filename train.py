import os
import tqdm
import argparse
import pprint

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import numpy as np
import skimage
import os
import glob
from skimage.io import imread
import skimage
import math
import time
from utils.imresize import imresize
from datasets import get_train_dataloader, get_valid_dataloaders
from models import get_model
from losses import get_loss
from optimizers import get_optimizer
from schedulers import get_scheduler
from transforms import get_transform

from tensorboardX import SummaryWriter

import utils.config
import utils.checkpoint
from utils.metrics import PSNR

device = None

def adjust_learning_rate(config, epoch):
    lr = config.optimizer.params.lr * (0.5 ** (epoch // config.scheduler.params.step_size))
    return lr

def train_single_epoch(config, model, dataloader, criterion,
                       optimizer, epoch, writer, postfix_dict):
    model.train()
    batch_size = config.train.batch_size
    total_size = len(dataloader.dataset)
    total_step = math.ceil(total_size / batch_size)

    log_dict = {}

    tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
    for i, (HR_patch, LR_patch, BC_patch) in tbar:
        HR_patch = HR_patch.to(device)
        LR_patch = LR_patch.to(device)

        optimizer.zero_grad()
        output = model.forward(LR_patch)
        # loss = 0
        # if type(output) == list:
        #    for o in output:
        #        loss += criterion(o, HR_patch)
        #else:
        #    loss = criterion(output, HR_patch)
        loss = criterion(output, HR_patch)
        log_dict['loss'] = loss.item()

        loss.backward()
        if 'gradient_clip' in config.optimizer:
            lr = adjust_learning_rate(config, epoch)
            clip = config.optimizer.gradient_clip / lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        f_epoch = epoch + i / total_step

        log_dict['lr'] = optimizer.param_groups[0]['lr']
        for key, value in log_dict.items():
            postfix_dict['train/{}'.format(key)] = value

        desc = '{:5s}'.format('train')
        desc += ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
        tbar.set_description(desc)
        tbar.set_postfix(**postfix_dict)

        if i % 100 == 0:
            log_step = int(f_epoch * 10000)
            if writer is not None:
                for key, value in log_dict.items():
                    writer.add_scalar('train/{}'.format(key), value, log_step)


def evaluate_single_epoch(config, model, dataloader, criterion,
                          epoch, writer, postfix_dict):
    model.eval()
    with torch.no_grad():
        batch_size = config.eval.batch_size
        total_size = len(dataloader.dataset)
        total_step = math.ceil(total_size / batch_size)

        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)

        total_psnr = 0
        total_psnr_bic = 0
        total_loss = 0
        for i, (HR_img, LR_img, BC_img) in tbar:
            HR_img = HR_img[:,:1].to(device)
            LR_img = LR_img[:,:1].to(device)
            BC_img = BC_img[:,:1].to(device)

            pred = model.forward(LR_img)
            if type(pred) == list:
                pred = pred[-1]

            total_loss += criterion(pred, HR_img).item()

            total_psnr += PSNR(pred.cpu(), HR_img.cpu(),
                               s=config.data.scale_factor)

            f_epoch = epoch + i / total_step
            desc = '{:5s}'.format('val')
            desc += ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
            tbar.set_description(desc)
            tbar.set_postfix(**postfix_dict)

        log_dict = {}
        avg_loss = total_loss / (i+1)
        avg_psnr = total_psnr / (i+1)
        log_dict['loss'] = avg_loss
        log_dict['psnr'] = avg_psnr

        for key, value in log_dict.items():
            if writer is not None:
                writer.add_scalar('val/{}'.format(key), value, epoch)
            postfix_dict['val/{}'.format(key)] = value

        return avg_psnr


def train(config, model, dataloaders, criterion,
          optimizer, scheduler, writer, start_epoch):
    num_epochs = config.train.num_epochs
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.cuda()
    postfix_dict = {'train/lr': 0.0,
                    'train/loss': 0.0,
                    'val/psnr': 0.0,
                    'val/loss': 0.0}
    psnr_list = []
    best_psnr = 0.0
    best_psnr_mavg = 0.0
    for epoch in range(start_epoch, num_epochs):

        # val phase
        psnr = evaluate_single_epoch(config, model, dataloaders['val'],
                                   criterion, epoch, writer, postfix_dict)
        if config.scheduler.name == 'reduce_lr_on_plateau':
            scheduler.step(psnr)
        elif config.scheduler.name != 'reduce_lr_on_plateau':
            scheduler.step()

        utils.checkpoint.save_checkpoint(config, model, optimizer, epoch, 0)
        psnr_list.append(psnr)
        psnr_list = psnr_list[-10:]
        psnr_mavg = sum(psnr_list) / len(psnr_list)

        if psnr > best_psnr:
            best_psnr = psnr
        if psnr_mavg > best_psnr_mavg:
            best_psnr_mavg = psnr_mavg

        # train phase
        train_single_epoch(config, model, dataloaders['train'],
                           criterion, optimizer, epoch, writer, postfix_dict)


    return {'psnr': best_psnr, 'psnr_mavg': best_psnr_mavg}


def run(config):
    train_dir = config.train.dir

    model = get_model(config).cuda()
    criterion = get_loss(config)
    optimizer = get_optimizer(config, model.parameters())

    checkpoint = utils.checkpoint.get_initial_checkpoint(config)
    if checkpoint is not None:
        last_epoch, step = utils.checkpoint.load_checkpoint(model, optimizer, checkpoint)
    else:
        last_epoch, step = -1, -1

    print('from checkpoint: {} last epoch:{}'.format(checkpoint, last_epoch))
    scheduler = get_scheduler(config, optimizer, last_epoch)

#     dataloaders = {split:get_dataloader(config, split, get_transform(config, split))
#                    for split in ['train', 'val']}

    print(config.data)
    dataloaders = {'train':get_train_dataloader(config, get_transform(config)),
                   'val':get_valid_dataloaders(config)[0]}
    writer = SummaryWriter(train_dir)
    train(config, model, dataloaders, criterion, optimizer, scheduler,
          writer, last_epoch+1)


def parse_args():
    parser = argparse.ArgumentParser(description='Super Resolution')
    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default=None, type=str)
    return parser.parse_args()


def main():
    global device
    import warnings
    warnings.filterwarnings("ignore")

    print('train super resolution')
    args = parse_args()
    if args.config_file is None:
        raise Exception('no configuration file')

    config = utils.config.load(args.config_file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # os.environ["CUDA_VISIBLE_DEVICES"]= str(config.gpu)

    pprint.PrettyPrinter(indent=2).pprint(config)
    utils.prepare_train_directories(config)
    run(config)

    print('success!')


if __name__ == '__main__':
    main()



# #model = FSRCNN(scale_factor=scale_factor)
# model = DRRN()

# num_total_params = sum(p.numel() for p in model.parameters())
# print("The number of parameters : ", num_total_params)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.to(device)

# num_epochs = 51
# lr = 1e-3
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# loss_func = nn.MSELoss()

# if not os.path.exists("./weights/"):
#     os.mkdir("./weights/")


# for epoch in range(num_epochs):
#     # training stage
#     model.train()
#     log_dict = dict()
#     for i, (HR_patch, LR_patch, BC_patch) in enumerate(train_loader):
#         #############
#         # CODE HERE #
#         #############
#         HR_patch = HR_patch.to(device)
#         LR_patch = LR_patch.to(device)

#         optimizer.zero_grad()
#         output = model.forward(LR_patch)

#         loss = loss_func(output, HR_patch)
#         loss.backward()

#         optimizer.step()

#         f_epoch = epoch + i / total_step


#     # test stage
#     end = time.time()
#     model.eval()
#     # Calculate PSNR
#     total_psnr = 0
#     total_psnr_bic = 0
#     # Iterate through test dataset
#     with torch.no_grad():
#         for j, (HR_img, LR_img, BC_img) in enumerate(test_Set5_loader):
#             #############
#             # CODE HERE #
#             #############
#             HR_img = HR_img[:,:1].to(device)
#             LR_img = LR_img[:,:1].to(device)
#             BC_img = BC_img[:,:1].to(device)

#             pred = model.forward(LR_img)

#             total_loss += loss_func(pred, HR_img).item()

#             total_psnr += PSNR(pred.cpu(), HR_img.cpu(), s=scale_factor)
#             total_psnr_bic += PSNR(BC_img.cpu(), HR_img.cpu(), s=scale_factor)


#     print('Epochs: {}. Loss: {:.6f}. PSNR: {:.3f} (bicubic)\t {:.3f} (Model) Elapsed time: {:.3f} sec'.format(
#             epoch, total_loss/(j+1), total_psnr_bic/(j+1), total_psnr/(j+1), end-start))

#     # save weights
#     if epoch % 5 == 0 and epoch != 0:
#         torch.save({'state_dict':model.state_dict()},
#                    './weights/checkpoint_%03d.pkl'%(epoch))












