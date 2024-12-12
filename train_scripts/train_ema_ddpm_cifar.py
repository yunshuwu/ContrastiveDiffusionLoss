import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import os
import argparse
from diffusers import UNet2DModel

import sys
sys.path.append("..")
from ckpt_util import get_ckpt_path
# from network_arch.unets.unet import WrapUNet2DModel, UNet2DModel_DDPM
from network_arch.ema_ddpm_unet import Model, WrapModel
from diffusion_models.diffusionmodel_test import *
from datasets.data_generator import *

#--------------------------------------------------------------------------------------
def train(args):
    # 1. data preparison
    train_dl, train_ds, val_dl, val_ds, x_shape = CIFAR10_dataloader(args.batch_size, data_fp='../datasets/CIFAR10')

    # 2. Define diffusion model and network architecture
    cifar10_cfg = {
        "resolution": 32,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": (1,2,2,2),
        "num_res_blocks": 2,
        "attn_resolutions": (16,),
        "dropout": 0.1,
    }

    if args.diffusion_name.lower() == 'cdl':
        denoiser = WrapModel(**cifar10_cfg)
    elif args.diffusion_name.lower() == 'ddpm':
        denoiser = Model(**cifar10_cfg)

    print(f'Finetuning {args.dataset}')
    ckpt_path = get_ckpt_path(args.dataset) # ema_cifar10
    print(f"Loading checkpoint {ckpt_path}")
    ckpt = torch.load(ckpt_path)
    denoiser.load_state_dict(ckpt)
    # print(denoiser)

    if args.diffusion_name.lower() == 'cdl':
        dm = CDL_DiffusionModel_EMA(denoiser, x_shape, 
                                    ema_decay=0.9999, 
                                    learning_rate=args.learning_rate, 
                                    cdl_loss_weight=1., std_loss_weight=1.)
        for name, param in dm.model.named_parameters():
            print(f'{name} requires_grad: {param.requires_grad}')

        print(f'Check EMA model ...')
        for name, param in dm.ema_model.named_parameters():
            print(f'{name} requires_grad: {param.requires_grad}')

        dm.dataset_info(train_dl, diagonal=args.diagonal, dataset_name="cifar10")

    elif args.diffusion_name.lower() == 'ddpm':
        dm = DDPM_DiffusionModel(denoiser, x_shape, 
                                 learning_rate=args.learning_rate)

    else:
        print(f'Wrong loss input... No such a diffusion loss...')

    # 3. Training / Finetuning
    print("Finetuning...")
    print(f'enabling checkpointing? {args.enable_checkpointing}')
    logger = TensorBoardLogger("../lightning_logs", name=args.diffusion_name)
    trainer_cdl = pl.Trainer(max_epochs=args.max_epochs, 
                             enable_checkpointing=args.enable_checkpointing, 
                             accelerator=args.accelerator, 
                             devices=[1,7], 
                             logger=logger, 
                             default_root_dir=args.checkpoint_root_dir)
    trainer_cdl.fit(dm, train_dl, val_dl)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Model Choice
    parser.add_argument("--diffusion_name", type=str, default="cdl", choices=["cdl", "ddpm"], help="Which diffusion model to be used.")
    # Dataset
    parser.add_argument("--dataset", type=str, default="ema_cifar10", choices=["ema_cifar10", "cifar10"], help="Type of the dataset to be used.")
    # Training configs
    # CDL related
    parser.add_argument("--diagonal", type=bool, help="Use diagonal approximation for dataset covariance matrix estimation.")
    # training related
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size used during training")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of epochs")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of parallel data loading workers")
    parser.add_argument("--persistent_workers", action="store_true", help="Set persistent_workers=True if this flag is present")
    parser.add_argument("--enable_checkpointing", action="store_true", help="Set enable_checkpointing=True if this flag is present")
    parser.add_argument("--accelerator", type=str, default='gpu', help="Use GPU or CPU as accelerator to train")
    parser.add_argument("--checkpoint_root_dir", type=str, default="../", help="Which dict to store checkpoints")

    args = parser.parse_args()

    if args.diffusion_name in ['cdl', 'itd']:
        if not args.diagonal:
            parser.error("--diagonal is required when --diffusion_name is ITD or CDL")

    train(args=args)