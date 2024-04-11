import os
import argparse

import torch
import cv2
import numpy as np
import yaml

from models import model_rrdb, model_swinir
from datasets import srdata_test
from torch.utils import data

import logging
from utils import utils_logger, util_calculate_psnr_ssim

def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--opt', type=str, help='path to option file', required=True)
    parser.add_argument('--output_path', type=str, help='path to your output', required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Initialization
    with open(args.opt, 'r') as f:
        opt = yaml.safe_load(f)
    opt['name'] = opt['name'].replace('RRDB', opt['model_type'])
    print(opt)

    ckpt_path = opt['ckpt_path']

    weight = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    weight = weight['model']

    # Models
    if opt['model_type'].lower() == 'rrdb':
        model = model_rrdb.RRDBNet(**opt['model']['rrdb']).to('cuda')
    elif opt['model_type'].lower() == 'swinir':
        model = model_swinir.SwinIR(**opt['model']['swinir']).to('cuda')
    else:
        raise ValueError(f"Model {opt['model_type']} is currently unsupported!")

    model.load_state_dict(weight)
    model = model.cuda()

    # Datasets
    testset = srdata_test.Test(**opt['test'])
    data_loader_test = data.DataLoader(
        testset, 
        **opt['dataloader']['test'],
        shuffle=False,
    )

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    if opt['test']['use_hr']:
        logger_name = opt['stage']
        utils_logger.logger_info(logger_name, os.path.join(args.output_path, logger_name+'.log'), mode='w')
        logger = logging.getLogger(logger_name)
        p = 0
        s = 0    
        count = 0

    # Start testing
    model.eval()
    for batch in data_loader_test:
        lr = batch['lr']
        fn = batch['fn'][0]
        if opt['test']['use_hr']:
            hr = batch['hr']

        lr = lr.to('cuda')
        with torch.no_grad():
            sr = model(lr)
        sr = sr.detach().cpu().squeeze(0).numpy().transpose(1, 2, 0)
        sr = sr * 255.
        sr = np.clip(sr.round(), 0, 255).astype(np.uint8)
        sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.output_path, fn), sr)
        
        if opt['test']['use_hr']:
            hr = hr.squeeze(0).numpy().transpose(1, 2, 0)
            hr = hr * 255.
            hr = np.clip(hr.round(), 0, 255).astype(np.uint8)
            hr = cv2.cvtColor(hr, cv2.COLOR_RGB2BGR)
            
            psnr = util_calculate_psnr_ssim.calculate_psnr(sr, hr, crop_border=4, test_y_channel=True)
            ssim = util_calculate_psnr_ssim.calculate_ssim(sr, hr, crop_border=4, test_y_channel=True)
            p += psnr
            s += ssim
            count += 1

            logger.info('{}: {}, {}'.format(fn, psnr, ssim))

    if opt['test']['use_hr']:
        p /= count
        s /= count
        logger.info("Avg psnr: {}. ssim: {}. count: {}.".format(p, s, count))

    print('Testing finished!')
