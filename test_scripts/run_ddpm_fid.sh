#~/bin/bash

########### uncond Cifar ddpm ##########
echo "seed=5000-9999" >>  ./neurips24_rebuttal/fid_para_karras_baseline_0709.txt
echo "ddpm..." >> ./neurips24_rebuttal/fid_para_karras_baseline_0709.txt
CUDA_VISIBLE_DEVICES=1,2 python parallel_sampling_realworld_data.py --diffusion_loss=ddpm --dataset=ema_cifar10 --batch_size=64 --seeds=5000-9999 --outdir=./neurips24_rebuttal/ema_CIFAR10_ddpm_5000
torchrun --standalone --nproc_per_node=1 fid_mp.py calc --images=neurips24_rebuttal/ema_CIFAR10_ddpm_5000 --ref=../datasets/CIFAR10/cifar10-32x32.npz --num=5000 >> ./neurips24_rebuttal/fid_para_karras_baseline_0709.txt

echo "cdl..." >> ./neurips24_rebuttal/fid_para_karras_baseline_0709.txt
CUDA_VISIBLE_DEVICES=1,2 python parallel_sampling_realworld_data.py --diffusion_loss=cdl --dataset=ema_cifar10 --batch_size=64 --seeds=5000-9999 --outdir=./neurips24_rebuttal/ema_CIFAR10_cdl_5000
torchrun --standalone --nproc_per_node=1 fid_mp.py calc --images=neurips24_rebuttal/ema_CIFAR10_cdl_5000 --ref=../datasets/CIFAR10/cifar10-32x32.npz --num=5000 >> ./neurips24_rebuttal/fid_para_karras_baseline_0709.txt


echo "seed=10000-14999" >>  ./neurips24_rebuttal/fid_para_karras_baseline_0709.txt
echo "ddpm..." >> ./neurips24_rebuttal/fid_para_karras_baseline_0709.txt
CUDA_VISIBLE_DEVICES=1,2 python parallel_sampling_realworld_data.py --diffusion_loss=ddpm --dataset=ema_cifar10 --batch_size=64 --seeds=10000-14999 --outdir=./neurips24_rebuttal/ema_CIFAR10_ddpm_10000
torchrun --standalone --nproc_per_node=1 fid_mp.py calc --images=neurips24_rebuttal/ema_CIFAR10_ddpm_10000 --ref=../datasets/CIFAR10/cifar10-32x32.npz --num=5000 >> ./neurips24_rebuttal/fid_para_karras_baseline_0709.txt

echo "cdl..." >> ./neurips24_rebuttal/fid_para_karras_baseline_0709.txt
CUDA_VISIBLE_DEVICES=1,2 python parallel_sampling_realworld_data.py --diffusion_loss=cdl --dataset=ema_cifar10 --batch_size=64 --seeds=10000-14999 --outdir=./neurips24_rebuttal/ema_CIFAR10_cdl_10000
torchrun --standalone --nproc_per_node=1 fid_mp.py calc --images=neurips24_rebuttal/ema_CIFAR10_cdl_10000 --ref=../datasets/CIFAR10/cifar10-32x32.npz --num=5000 >> ./neurips24_rebuttal/fid_para_karras_baseline_0709.txt
