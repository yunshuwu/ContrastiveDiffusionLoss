#~/bin/bash


python train_1D_gauss.py --diffusion_name cdl --diagonal True --max_epochs 200 --persistent_workers --enable_checkpointing 

python train_1D_gauss.py --diffusion_name ddpm --max_epochs 200 --persistent_workers --enable_checkpointing 
