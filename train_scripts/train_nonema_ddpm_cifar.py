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
from network_arch.unets.unet import WrapUNet2DModel, UNet2DModel_DDPM
from diffusion_models.diffusionmodel_test import *
from datasets.data_generator import *


def train(args):
    """Finetune diffusion models on real-world datasets."""
    
    # 1. data preparison
    train_dl, train_ds, val_dl, val_ds, x_shape = CIFAR10_dataloader(args.batch_size, data_fp='../datasets/CIFAR10')

    # 2. Define diffusion model and network architecture
    model_id = "google/ddpm-cifar10-32"
    modelB = UNet2DModel.from_pretrained(model_id)
    # denoiser = UNet2DModel_DDPM(**modelB.config) # TODO: change the Unet name, should be something telling that forward function takes (x, t) as inputs. # WrapUNet2DModel(**modelB.config)
    # denoiser.load_state_dict(modelB.state_dict())

    print(f'Training on {args.diffusion_name} loss ...')
    if args.diffusion_name.lower() == 'cdl':
        denoiser = WrapUNet2DModel(**modelB.config) 
        denoiser.load_state_dict(modelB.state_dict())
        dm = CDL_DiffusionModel(denoiser, x_shape, 
                                learning_rate=args.learning_rate, 
                                cdl_loss_weight=1., std_loss_weight=1.) 
        dm.dataset_info(train_dl, diagonal=args.diagonal, dataset_name=args.dataset)

    elif args.diffusion_name.lower() == 'ddpm':
        denoiser = UNet2DModel_DDPM(**modelB.config) 
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
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10"], help="Type of the dataset to be used.")
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

    if args.diffusion_name in ['cdl']:
        if not args.diagonal:
            parser.error("--diagonal is required when --diffusion_name is CDL") 

    train(args=args)