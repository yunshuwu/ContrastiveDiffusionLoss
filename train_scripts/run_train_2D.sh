#~/bin/bash


echo "Train diffusion models on real-dino..." >> output_train_2D_dm.txt

python train_2D.py --diffusion_name cdl --diagonal True --dataset real-dino --max_epochs 2000 --persistent_workers --enable_checkpointing --checkpoint_root_dir ../ 

python train_2D.py --diffusion_name ddpm --dataset real-dino --max_epochs 2000 --persistent_workers --enable_checkpointing --checkpoint_root_dir ../