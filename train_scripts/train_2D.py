import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import os
import argparse

import sys
sys.path.append("..")
from network_arch.nets import *
from diffusion_models.diffusionmodel import *
from datasets.flow_synthetic_2Ddata import sample_2d_synthetic
from datasets.data_generator import *

def train(args):
    """Train diffusion models."""
    
    # 1. Training/Validation data preparison
    if args.dataset.lower() == "real_dino":
        train_dl, val_dl, x_shape, train_ds = dino_dataset(n=100000, batch_size=args.batch_size, path_2d="../datasets/assets/DatasaurusDozen.tsv")
    else:
        train_ds, val_ds, x_shape = sample_2d_synthetic(args.dataset)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers, persistent_workers=args.persistent_workers)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers, persistent_workers=args.persistent_workers)


    # 2. Define diffusion model and network arch
    # now using network architecture with positional encoding, to ensure that training is complete
    denoiser = BasicDiscreteTimeModel(d_model=128, n_layers=2)
    
    print(f'Training on {args.diffusion_name} loss...') 
    if args.diffusion_name.lower() == "cdl":
        ckpt_path = '../lightning_logs/einstein_cdl/version_0/checkpoints/epoch=1999-step=7814000.ckpt'
        ckpt = torch.load(ckpt_path)
        dm = CDL_DiffusionModel(denoiser, **ckpt['hyper_parameters'])

    elif args.diffusion_name.lower() == "ddpm":
        ckpt_path = '../lightning_logs/einstein_ddpm/version_0/checkpoints/epoch=1999-step=7814000.ckpt'
        ckpt = torch.load(ckpt_path)
        dm = DDPM_DiffusionModel(denoiser, **ckpt['hyper_parameters']) 

    else:
        print(f'Wrong loss input... No such a diffusion loss...')

    # Calculate the integral bound
    print(f"Calculating integral bound...")
    if args.diffusion_name.lower() == "cdl" and args.diagonal:
        dm.dataset_info(train_dl, diagonal=args.diagonal, dataset_name=args.dataset)


    # 3. Training
    logger = TensorBoardLogger("../lightning_logs", name=f'{args.dataset}_{args.diffusion_name}')

    trainer = pl.Trainer(max_epochs=args.max_epochs, 
                         enable_checkpointing=args.enable_checkpointing, 
                         accelerator=args.accelerator, devices=[0,1,2], 
                         logger=logger, 
                         default_root_dir=args.checkpoint_root_dir)
    trainer.fit(dm, train_dl, val_dl) 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Model Choice
    parser.add_argument("--diffusion_name", type=str, default="cdl", choices=["cdl", "ddpm"], help="Which diffusion model to be used.")
    # Dataset
    parser.add_argument("--dataset", type=str, default="real_dino", choices=["dino", "real_dino", "spirals", "moons", "eight_gaussian", "checkerboard", "circle"], help="Type of the dataset to be used.")
    # Training configs
    # CDL related
    parser.add_argument("--diagonal", type=bool, help="Use diagonal approximation for dataset covariance matrix estimation.")
    # dataset related
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size used during training")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of parallel data loading workers")
    parser.add_argument("--persistent_workers", action="store_true", help="Set persistent_workers=True if this flag is present")
    # network arch related
    parser.add_argument("--in_dim", type=int, default=2, help="Network input dimension, for 2D data, input dim=2")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Dimensionality of hidden layers")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of network layers, for 2D we use 3-layer MLP")
    # training related
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=20, help="Maximum number of epochs")
    parser.add_argument("--enable_checkpointing", action="store_true", help="Set enable_checkpointing=True if this flag is present")
    parser.add_argument("--accelerator", type=str, default='gpu', help="Use GPU or CPU as accelerator to train")
    parser.add_argument("--checkpoint_root_dir", type=str, default="../", help="Which dict to store checkpoints")

    args = parser.parse_args()

    if args.diffusion_name in ['cdl']:
        if not args.diagonal:
            parser.error("--diagonal is required when --name is ITD or CDL")

    train(args=args)