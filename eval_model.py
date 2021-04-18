import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
import pytorch_warmup as warmup
from torchsummary import summary

import model.loss as module_loss
from model.metric import *
from model.model import Unet
from dataloader.polyp_data import *
from trainer.trainer import *
from utils import *
from arguments import get_args_training


device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = get_args_training()


test_root = args.test_root
train_size = args.train_size

file_path = args.backbone + '_' + args.bridge + '_' + args.decoder
file_name = args.loss_func + '_size_' + str(args.train_size) + '_bs_' + str(args.batch_size) \
                            + '_nfilters_' + str(args.n_filters)

if 'MHSA' in args.bridge:
    file_name = file_name + '_nblocks_' + str(args.n_blocks)


# Checkpoint setting
if args.augmentation:
    args.saved_checkpoint = os.path.join(args.saved_checkpoint, 'Augmentation')
    create_dir(args.saved_checkpoint)
checkpoint_path = os.path.join(args.saved_checkpoint, file_path)
create_dir(checkpoint_path)
checkpoint_name = os.path.join(checkpoint_path, file_name + '.pth.tar')
args.checkpoint_name = checkpoint_name

# Log setting
create_dir(args.result_path)
logging_path = args.result_path
if args.augmentation:
    logging_path = os.path.join(logging_path, 'Augmentation')   # results/Augmentation
    create_dir(logging_path)
logging_path = os.path.join(logging_path, file_path)    # results/[Augmentation]/arch_name/
create_dir(logging_path)
logging_name = os.path.join(logging_path, file_name + '.log') # results/[Augmentation]/arch_name/setting.log
if os.path.exists(logging_name):
    os.remove(logging_name)
logging.basicConfig(filename=logging_name, format='%(asctime)s %(message)s', filemode='a')       
logger = logging.getLogger()
logger.setLevel(logging.INFO)
args.logger = logger

# Model setting
model = Unet(in_channels=3).to(device)

# Loading model
if os.path.exists(checkpoint_name):
    # print(checkpoint_path)
    load_checkpoint(torch.load(checkpoint_name), model)
else:
    print(f'Does not exist checkpoint file in path {checkpoint_name}!')
    exit()

model.eval()

for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
# for _data_name in ['CVC-ColonDB', 'ETIS-LaribPolypDB']:
    data_path = '{}/{}/'.format(args.test_root, _data_name)
    save_path = '{}/{}/{}/{}'.format(args.saved_img, file_path, file_name, _data_name)
    # create_dir(save_path)

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    
    # Load data
    test_loader = test_dataset(image_root, gt_root, args.train_size)

    # metrics_logger initialization
    metrics = Metrics(['recall', 'specificity', 'precision', 'F1', 'F2',
                       'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean'])
    total_batch = int(test_loader.size)

    with torch.no_grad():
        for i in tqdm.tqdm(range(test_loader.size), desc=_data_name):
            image, gt, name, real_image = test_loader.load_data()
            image = image.to(device)
            gt = gt.to(device)
            pred_mask = model(image)

            exp_h, exp_w = gt.shape[2:]
            # Upsample to be equal with ground truth mask
            pred_mask = F.upsample(pred_mask, size=(exp_h, exp_w), mode='bilinear', align_corners=False)
            pred_mask = F.sigmoid(pred_mask)

            # Save all predicting masks
            # plot_results(pred_masks=pred_mask, true_masks=gt, images=real_image, result_path=save_path, name=name)

            # Calculate all scores
            _recall, _specificity, _precision, _F1, _F2, \
            _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean = evaluate(pred_mask, gt)

            metrics.update(recall= _recall, specificity= _specificity, precision= _precision, 
                            F1= _F1, F2= _F2, ACC_overall= _ACC_overall, IoU_poly= _IoU_poly, 
                            IoU_bg= _IoU_bg, IoU_mean= _IoU_mean
                        )
    metrics_result = metrics.mean(total_batch)

    # with open(log_file, 'a') as f:
    logger.info(f'RESULT FOR DATASET {_data_name}\n')
    logger.info('='*30)
    logger.info('\n')
    for k,v in metrics_result.items():
        logger.info(f'{k}: {v}\n')
    logger.info('='*30)
    logger.info('\n')






        