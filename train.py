import argparse
from collections import OrderedDict
import os

import numpy as np
import cv2
import yaml

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils import data
from torch import distributed as dist
import torch.optim as optim

from models import model_rrdb, model_swinir, sed
from datasets import srdata

import logging
from utils import utils_logger, util_calculate_psnr_ssim, losses


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--opt', type=str, help='path to option file', required=True)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="path to the checkpoints for pretrained model",
    )
    parser.add_argument(
        '--distributed',
        action='store_true'
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument('--data_root', type=str, default='./DF2K')
    parser.add_argument('--data_test_root', type=str, default='./Evaluation')
    parser.add_argument('--out_root', type=str, default='./checkpoint')

    args = parser.parse_args()

    return args

def data_sampler(dataset, shuffle=True, distributed=True):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def main():
    args = parse_args()

    # Initialization
    with open(args.opt, 'r') as f:
        opt = yaml.safe_load(f)
    opt['name'] = opt['name'].replace('RRDB', opt['model_type'])
    print(opt)

    ckpt_path = os.path.join(args.out_root, opt['name'])
    if not os.path.exists(ckpt_path):
        if torch.cuda.current_device() == 0:
            os.makedirs(ckpt_path, exist_ok=True)

    logger_name = opt['stage']
    utils_logger.logger_info(logger_name, os.path.join(ckpt_path, logger_name+'.log'), mode='w')
    logger = logging.getLogger(logger_name)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl")

    if opt['manual_seed']:
        torch.manual_seed(opt['manual_seed'])
        torch.cuda.manual_seed(opt['manual_seed'])
        torch.cuda.manual_seed_all(opt['manual_seed'])

    loss_weight = opt['loss_weights']

    # Models
    if opt['model_type'].lower() == 'rrdb':
        model = model_rrdb.RRDBNet(**opt['model']['rrdb']).to('cuda')
    elif opt['model_type'].lower() == 'swinir':
        model = model_swinir.SwinIR(**opt['model']['swinir']).to('cuda')
    else:
        raise ValueError(f"Model {opt['model_type']} is currently unsupported!")
    
    model_ex = sed.CLIP_Semantic_extractor(**opt['model_ex']).to('cuda')

    if opt['name'].split('_')[-1] == 'P+SeD':
        model_d = sed.SeD_P(**opt['model_d']).to('cuda')
    elif opt['name'].split('_')[-1] == 'U+SeD':
        model_d = sed.SeD_U(**opt['model_d']).to('cuda')

    if args.resume is not None:
        if torch.cuda.current_device() == 0:
            logger.info(f"load pretrained model: {args.resume}")
        if not os.path.isfile(args.resume):
            raise ValueError
        ckpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt)
        if torch.cuda.current_device() == 0:
            logger.info("model checkpoint load!")

    # Optimizers
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], **opt['optimizer'])
    optimizer_d = optim.Adam([p for p in model_d.parameters() if p.requires_grad], **opt['optimizer_d'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **opt['scheduler'])
    scheduler_d = optim.lr_scheduler.MultiStepLR(optimizer_d, **opt['scheduler_d'])

    # Loss
    # Loss settings are hard-coded for now!
    loss_pix = torch.nn.L1Loss()
    loss_pix = loss_pix.to('cuda')
    loss_g = losses.GANLoss(gan_type='vanilla', loss_weight=loss_weight['loss_g']).to('cuda')
    loss_dict_per = {'2': 0.1, '7': 0.1, '16': 1.0, '25': 1.0, '34': 1.0}
    loss_p = losses.PerceptualLoss(layer_weights=loss_dict_per, perceptual_weight=loss_weight['loss_p'], criterion='l1').to('cuda')

    # Datasets
    use_eval = opt['datasets']['test']['use_test']
    trainset = srdata.Train(**opt['datasets']['train'], data_root=args.data_root)
    data_loader = data.DataLoader(
        trainset, 
        **opt['dataloader']['train'],
        sampler=data_sampler(trainset, shuffle=True, distributed=args.distributed),
    )

    if use_eval:
        testset = srdata.Test(data_root=args.data_test_root)
        data_loader_test = data.DataLoader(
            testset, 
            **opt['dataloader']['test'],
            sampler=data_sampler(testset, shuffle=False, distributed=False),
        )

    if args.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=True,
        )
        model_d = DistributedDataParallel(
            model_d,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=True,
        )

    # Training settings
    total_epochs = 10000000
    current_step = opt['train']['current_step']
    endflag = False

    # Training starts
    for epoch in range(total_epochs):
         
        for lr, hr, filename in data_loader:

            current_step += 1

            if current_step > opt['train']['total_step']:
                endflag = True
                break

            learning_r = optimizer.param_groups[0]['lr']

            filename = filename[0].split('/')[-1]
            lr = lr.to('cuda')
            hr = hr.to('cuda')
            hr_semantic = model_ex(hr)

            loss_dict = OrderedDict()
            
            sr = model(lr)

            for p in model_d.parameters():
                p.requires_grad = False

            optimizer.zero_grad()

            l_g_total = 0

            # pixel loss
            loss_pixel = loss_pix(sr, hr)
            l_g_total += loss_pixel * loss_weight['loss_pix']
            loss_dict['loss_pix'] = loss_pixel.item()
            # perceptual loss
            loss_percep = loss_p(sr, hr)
            l_g_total += loss_percep
            loss_dict['loss_p'] = loss_percep.item()
            # gan loss
            fake_g_pred = model_d(sr, hr_semantic)
            loss_gan = loss_g(fake_g_pred, True, is_disc=False)
            l_g_total += loss_gan
            loss_dict['loss_g'] = loss_gan.item()
            l_g_total.backward()
            optimizer.step()
            scheduler.step()

            # optimize net_d
            for p in model_d.parameters():
                p.requires_grad = True

            optimizer_d.zero_grad()
            # real
            real_d_pred = model_d(hr, hr_semantic)
            l_d_real = loss_g(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real.item()
            l_d_real.backward()
            # fake
            fake_d_pred = model_d(sr.detach().clone(), hr_semantic)  # clone for pt1.9
            l_d_fake = loss_g(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake.item()
            l_d_fake.backward()
            optimizer_d.step()
            scheduler_d.step()

            if current_step % opt['train']['log_every'] == 0 and torch.cuda.current_device() == 0:
                logger.info('LR: {} | Step: {} | loss_pix: {:.3f} | loss_per: {:.3f} | loss_gan: {:.5f} | loss_d_real: {:.3f} | loss_d_fake: {:.3f}'.format(
                    learning_r,
                    current_step,
                    loss_dict['loss_pix'],
                    loss_dict['loss_p'],
                    loss_dict['loss_g'],
                    loss_dict['l_d_real'],
                    loss_dict['l_d_fake']))

            if current_step % opt['train']['save_every'] == 0 and torch.cuda.current_device() == 0:
                m = model.module if args.distributed else model
                model_dict = m.state_dict()
                if torch.cuda.current_device() == 0:
                    torch.save(
                        model_dict,
                        os.path.join(ckpt_path, 'model_{}.pt'.format(current_step))
                    )

            if use_eval and current_step % opt['train']['test_every'] == 0 and torch.cuda.current_device() == 0:
                model.eval()
                p = 0
                s = 0
                count = 0
                for lr, hr, filename in data_loader_test:
                    count += 1
                    lr = lr.to('cuda')
                    filename = filename[0]
                    with torch.no_grad():
                        sr = model(lr)
                    sr = sr.detach().cpu().squeeze(0).numpy().transpose(1, 2, 0)
                    sr = sr * 255.
                    sr = np.clip(sr.round(), 0, 255).astype(np.uint8)
                    hr = hr.squeeze(0).numpy().transpose(1, 2, 0)
                    hr = hr * 255.
                    hr = np.clip(hr.round(), 0, 255).astype(np.uint8)

                    sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
                    hr = cv2.cvtColor(hr, cv2.COLOR_RGB2BGR)
                    
                    psnr = util_calculate_psnr_ssim.calculate_psnr(sr, hr, crop_border=4, test_y_channel=True)
                    ssim = util_calculate_psnr_ssim.calculate_ssim(sr, hr, crop_border=4, test_y_channel=True)
                    p += psnr
                    s += ssim
                    logger.info('{}: {}, {}'.format(filename, psnr, ssim))

                p /= count
                s /= count

                logger.info("Epoch: {}, Step: {}, psnr: {}. ssim: {}.".format(epoch, current_step, p, s))
                model.train()

        if endflag:
            break
    logger.info('Done')

if __name__ == '__main__':
    main()