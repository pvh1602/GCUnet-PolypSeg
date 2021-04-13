import os
import torch
import torch.nn as nn
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
from model.model import *
from dataloader.polyp_data import *
from trainer.trainer import *
from utils import *
from arguments import get_args_training

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'



def main(args):

    print('=' * 100)
    print('Arguments =')
    for arg in vars(args):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 100)

    img_root = args.img_root
    gt_root = args.gt_root
    train_size = args.train_size
    batch_size = args.batch_size
    n_epochs = args.n_epochs

    file_path = args.backbone + '_' + args.bridge + '_' + args.decoder
    file_name = args.loss_func + '_size_' + str(args.train_size) + '_bs_' + str(args.batch_size)

    # Checkpoint setting
    checkpoint_path = os.path.join(args.saved_checkpoint, file_path)
    create_dir(checkpoint_path)
    checkpoint_name = os.path.join(args.saved_checkpoint, file_name + '.pth.tar')
    args.checkpoint_name = checkpoint_name

    # Log setting
    logging_path = os.path.join(args.logging, file_path)
    create_dir(logging_path)
    logging_name = os.path.join(logging_path, file_name + '.log')
    # if os.path.exists(logging_name):
    #     os.remove(logging_name)
    logging.basicConfig(filename=logging_name, format='%(asctime)s %(message)s', filemode='a')       
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    args.logger = logger

    # Model setting
    model = Unet(in_channels=3).to(device)
    # summary(model, (3,256,256))
    criterion = getattr(module_loss, args.loss_func)
    metric = AvgMeter() 

    # Dataloader
    data_loader = get_loaders(
        image_root=img_root, gt_root=gt_root, batchsize=batch_size, trainsize=train_size
    )

    # Optimizer setting
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    n_steps = n_epochs * len(data_loader)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)
    scaler = torch.cuda.amp.GradScaler()
    warmup_schduler = None

    # Load model
    args.loaded_epoch = None
    if args.resume:
        print("Load model")
        checkpoint = torch.load(checkpoint_name)
        load_checkpoint(checkpoint, model, optimizer, lr_scheduler)
        args.loaded_epoch = 1 +  checkpoint['epoch']


    trainer = Trainer(
        model=model, 
        criterion=criterion, 
        optimizer=optimizer, 
        device=device, 
        data_loader=data_loader,
        metric=metric,
        lr_scheduler=lr_scheduler,
        warmup_schduler=warmup_schduler,
        scaler=scaler,
        args=args,
        n_epochs=n_epochs
    )
    trainer.train()
    


if __name__ == '__main__':
    # args = argparse.ArgumentParser(description='Unet')
    # args.add_argument('--n_epochs', type=int, default=100, help='The number of epoch')
    # args.add_argument('--batch_size', type=int, default=8)
    # args.add_argument('--lr', type=float, default=0.001)
    # args.add_argument('--loss_func', type=str, default='structure_loss', help='Loss function')
    # args.add_argument('--train_size', type=int, default=256)
    # args.add_argument('--img_root', type=str, default='../data/TrainDataset/image/')
    # args.add_argument('--gt_root', type=str, default='../data/TrainDataset/mask/')
    # args.add_argument('--saved_path', type=str, default='./saved_models')
    # args.add_argument('--logging', type=str, default='./logging')
    # args.add_argument('--backbone', type=str, default='ResEncoder')
    # args.add_argument('--bridge', type=str, default='ResBlock')
    # args.add_argument('--decoder', type=str, default='ResDecoder')
    # args.add_argument('--n_blocks', type=int, default=1)
    args = get_args_training()

    main(args)