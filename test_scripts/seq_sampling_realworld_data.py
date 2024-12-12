import argparse
import os
import click
import re

import torch.distributed
import torch.distributed
import tqdm
import torch
import torch.distributed
import torch.nn as nn
from torch.nn import functional as F
import PIL.Image
from diffusers import UNet2DModel

# from contextlib import contextmanager
# from pathlib import Path
# import urllib
# import shutil
# import hashlib
# from cleanfid.inception_torchscript import InceptionV3W
# from torchmetrics.image.fid import FrechetInceptionDistance # fid2() need this

import sys
sys.path.append("..")
from diffusion_models._para_samplers import *
from diffusion_models.sampling_schedule import *
from diffusion_models.diffusionmodel import *
from diffusion_models.seq_samplers import *
from network_arch.unets.unet import WrapUNet2DModel, WrapUNetModel, UNet2DModel_DDPM
from datasets.data_generator import CIFAR10_dataloader

# # Karras internal libs
# sys.path.insert(0, "/home/ywu380/Diffusion/Contrastive-Diffusion")
# import dnnlib
# from torch_utils import distributed as dist


############################################ Sequential DDPM sampler, just sanity check if FID~=3.17 ############################################
@torch.no_grad()
def ddpm_sampler(dm, batch_size=64, 
                 start_schedule=0.0001, end_schedule=0.02, num_steps=1000):
    """DDPM sampler"""
    dm.eval()

    print(f'DDPM sampler...')

    # Calcuate betas of DDPM schedule
    betas = torch.linspace(start_schedule, end_schedule, num_steps)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)

    x = torch.randn([batch_size, *dm.shape], device=dm.device)
    for i in reversed(range(num_steps)):
        # print(f'timestep = {i}')
        timestep = torch.full((1,), i, dtype=torch.long, device=dm.device)

        betas_t = get_by_idx(betas, timestep)
        sqrt_one_minusalphas_cumprod = get_by_idx(torch.sqrt(1. - alphas_cumprod), timestep) # mixing ratio (1-alpha_bar_t)
        sqrt_recip_alphas_t = get_by_idx(torch.sqrt(1. / alphas), timestep)

        # if dm_loss.lower() == 'cdl' or dm_loss.lower() == 'itd':
        #     # FIXME: I think here should be a bug...
        #     # translate t to logsnr
        #     logsnr = t2logsnr(timestep, alphas_cumprod).repeat(batch_size)
        #     # print(f'    logsnr.shape = {logsnr.shape}')
        #     # Estimated noisy data x_t
        #     mean_t = sqrt_recip_alphas_t * (x - betas_t * dm(x, logsnr) / sqrt_one_minusalphas_cumprod)
        #     posterior_var_t = betas_t
        # elif dm_loss.lower() == 'ddpm':
        t_ = timestep.repeat(batch_size) # FIXME: do I really need this? yes
        mean_t = sqrt_recip_alphas_t * (x - betas_t * dm(x, t_) / sqrt_one_minusalphas_cumprod)
        posterior_var_t = betas_t

        if timestep == 0:
            x = mean_t # if t=0, do not add noise
        else:
            eps = torch.randn_like(x)
            var_t = torch.sqrt(posterior_var_t) * eps
            x = mean_t + var_t

    return x


@torch.no_grad()
def karras_sampler(dm, batch_size=64, num_inference_steps=18):
    """Karras EDM sampler"""
    schedule, step_function = prepreation_karras(device=dm.device, num_steps=num_inference_steps)
    x = generate(dm, step_function, schedule, batch_size=batch_size)
    return x


def main(seeds, outdir, batch_size, diffusion_loss, sampler_name, num_inference_steps, device=torch.device('cuda:7')):
    """ Generate random images, code is parallelized in dpt."""

    train_dl, train_ds, val_dl, val_ds, x_shape = CIFAR10_dataloader(batch_size=batch_size, data_fp='../datasets/CIFAR10')
    print(f'x_shape = {x_shape}')

    # Load pretrained diffusionmodels
    model_id = "google/ddpm-cifar10-32"
    print(f'Load pretrained network from {model_id}...')
    modelB = UNet2DModel.from_pretrained(model_id)

    if diffusion_loss.lower() == 'ddpm' and sampler_name.lower() == 'ddpm':
        # for ddpm loss, just use the pretrained model, no need to finetune
        denoiser = UNet2DModel_DDPM(**modelB.config) # forward() input arguments: (x, timestep)
        denoiser.load_state_dict(modelB.state_dict())

        ckpt_path = "../lightning_logs/ddpm/version_0/checkpoints/epoch=9-step=1960.ckpt" 
        checkpoint = t.load(ckpt_path)
        # for ddpm loss, just use the pretrained model, no need to finetune
        dm = DDPM_DiffusionModel(denoiser, x_shape=x_shape) # wrap the pretrained model, so that can call dm's members
        dm.load_state_dict(checkpoint['state_dict'])

    elif diffusion_loss.lower() == 'ddpm' and sampler_name.lower() == 'karras':
        # karras sampler implementation is different from DDPM sampler
        denoiser = WrapUNet2DModel(**modelB.config) # call warpped UNet, forward() input arguments: (x, logsnr)
        denoiser.load_state_dict(modelB.state_dict())
        dm = DDPM_DiffusionModel(denoiser, x_shape=x_shape)
    elif diffusion_loss.lower() == 'cdl' and sampler_name.lower() == 'ddpm':
        denoiser = UNet2DModel_DDPM(**modelB.config) 
        # denoiser.load_state_dict(modelB.state_dict())
        # load checkpoint
        ckpt_path = "../lightning_logs/nonema_ddpm_pretrained_cdl/version_7/checkpoints/epoch=9-step=3910.ckpt" 
        checkpoint = t.load(ckpt_path)

        cont_weight = checkpoint['hyper_parameters']['cdl_loss_weight']
        diff_weight = checkpoint['hyper_parameters']['std_loss_weight']
        print(f'cdl : std = {cont_weight} : {diff_weight} ')
        logsnr_loc, logsnr_scale = checkpoint['hyper_parameters']['logsnr_loc'], checkpoint['hyper_parameters']['logsnr_scale']
        print(f'logsnr loc = {logsnr_loc}, scale = {logsnr_scale} ')

        dm = CDL_DiffusionModel(denoiser, **checkpoint['hyper_parameters'])
        dm.load_state_dict(checkpoint['state_dict'])
        dm.dataset_info(train_dl, diagonal=True, dataset_name='cifar10')


    print(f'Finish loading the model...')
    dm.to(device)
    dm.eval()
    
    # 3. Run parallel sampling
    print(f'len(seeds) = {len(seeds)}')
    num_batches = (len(seeds) // (batch_size))
    if num_batches == 0:
        all_batches = (torch.as_tensor(seeds), )
    else:
        all_batches = torch.as_tensor(seeds).tensor_split(num_batches)

    # Loop over batches.
    print(f'Generating {len(seeds)} images to "{outdir}"...')
    for batch_seeds in all_batches:
        cur_bz_size = len(batch_seeds)
        print(f'Current batch size = {cur_bz_size}')

        # Generate images.
        if sampler_name.lower() == 'ddpm':
            # print(f'Number DDPM sampling steps = {num_inference_steps}.')
            samples = ddpm_sampler(dm, batch_size=cur_bz_size, num_steps=num_inference_steps)
        elif sampler_name.lower() == 'karras':
            # print(f'Number Karras EDM sampling steps = {num_inference_steps}.')
            samples = karras_sampler(dm, batch_size=cur_bz_size, num_inference_steps=num_inference_steps)
        
        # Save images.
        images_np = (samples * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}')
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
            else:
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)

    # Done.
    print('Done.')


def parse_int_list(s):
    """This function is from karras EDM implementation: """
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Model Choice
    parser.add_argument("--diffusion_loss", type=str, default="cdl", choices=["cdl", "itd", "ddpm", "ddim", "edm"], help="Which diffusion model to be used.")
    # Dataset
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10"], help="Type of the dataset to be used.")
    # number of samples to generate
    parser.add_argument("--schedule_name", type=str, default="ddpm", choices=["ddpm", "karras"], help="Which sampling scheduler to use.")
    parser.add_argument("--num_inference_steps", type=int, default=1000, help="Number of inference steps.")
    parser.add_argument("--batch_size", type=int, default=64, help="Maximum number of batch samples to generate from the pretrained diffusion model.")
    parser.add_argument("--seeds", type=parse_int_list, default='0-63', help="Random seeds (e.g. 1,2,5-10)")
    parser.add_argument("--outdir", type=str, help="Path to store the images generated by parallel sampling.")

    args = parser.parse_args()

    main(seeds=args.seeds, outdir=args.outdir, batch_size=args.batch_size, diffusion_loss=args.diffusion_loss, sampler_name=args.schedule_name, num_inference_steps=args.num_inference_steps)