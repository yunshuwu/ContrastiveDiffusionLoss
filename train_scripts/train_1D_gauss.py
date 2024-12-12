"""
Train diffusion model on 1D two-mode Gaussian data, mainly for visualizing the converging process of trajectory (gif)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import os
import argparse

import sys
sys.path.append("..")
from network_arch.nets import *
from diffusion_models.diffusionmodel import *
from datasets.gmm import GMMDist

def train(args):
    """Train diffusion models."""
    
    # 1. Training/Validation data preparison
    mix_onedim_gauss = GMMDist(dim=1, mix_probs=0.5, means=5)
    NUM_SAMPLES = 100000 
    torch.manual_seed(42)
    train_samples = mix_onedim_gauss.sample((NUM_SAMPLES,))
    torch.manual_seed(66)
    val_samples = mix_onedim_gauss.sample((NUM_SAMPLES,))
    x_shape = train_samples.shape[1:]

    train_dl = DataLoader(TensorDataset(train_samples), batch_size=args.batch_size, num_workers=args.num_workers, persistent_workers=args.persistent_workers)
    val_dl = DataLoader(TensorDataset(val_samples), batch_size=args.batch_size, num_workers=args.num_workers, persistent_workers=args.persistent_workers)

    # 2. Define diffusion model and network arch
    denoiser = MLP(args.in_dim, args.hidden_dim, args.n_layers, 
                   activation=nn.ReLU, dropout=0.)
    
    print(f'Training on {args.diffusion_name} loss...') 
    if args.diffusion_name.lower() == "cdl":
        dm = CDL_DiffusionModel(denoiser, x_shape, 
                                learning_rate=args.learning_rate, 
                                cdl_loss_weight=1., std_loss_weight=1.)
    elif args.diffusion_name.lower() == "ddpm":
        dm = DDPM_DiffusionModel(denoiser, x_shape, 
                                 learning_rate=args.learning_rate)
    else:
        print(f'Wrong loss input... No such a diffusion loss...')

    # Calculate the integral bound
    print(f"Calculating integral bound...")
    if args.diffusion_name.lower() == "cdl" and args.diagonal:
        dm.dataset_info(train_dl, diagonal=args.diagonal, dataset_name="1D_gauss")


    # 3. Training
    logger = TensorBoardLogger("../lightning_logs", name=f'1D_gaussian_{args.diffusion_name}')

    trainer = pl.Trainer(max_epochs=args.max_epochs, 
                         enable_checkpointing=args.enable_checkpointing, 
                         accelerator=args.accelerator, devices=[0], 
                         logger=logger, 
                         default_root_dir=args.checkpoint_root_dir)
    trainer.fit(dm, train_dl, val_dl)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

  
    # Model Choice
    parser.add_argument("--diffusion_name", type=str, default="cdl", choices=["cdl", "ddpm"], help="Which diffusion model to be used.")
    # ITD or CDL related
    parser.add_argument("--diagonal", type=bool, help="Use diagonal approximation for dataset covariance matrix estimation.")
    # Training configs
    # dataset related
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size used during training")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of parallel data loading workers")
    parser.add_argument("--persistent_workers", action="store_true", help="Set persistent_workers=True if this flag is present")
    # network arch related
    parser.add_argument("--in_dim", type=int, default=1, help="Network input dimension, for 1D data, input dim=1; for 2D data, input dim=2")
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
            parser.error("--diagonal is required when --name is CDL")

    train(args=args)