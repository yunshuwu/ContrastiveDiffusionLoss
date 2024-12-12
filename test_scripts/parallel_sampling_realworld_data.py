"""
In this py file, we show how to do parallel sampling given a pretrained diffusion model 'diffusionmodel'

Parallel sampling still use multiprocessing as parallel computation, using producer/consumer design.
We spawn a single producer process on GPU:0 to run the main loop of ParaSam, and all GPUs (including GPU:0) are as consumer processes, running parallel part of the ParaSam

"""
import os
import re
import argparse
import json
import PIL.Image

import torch
import torch.nn as nn
import types
import torch.multiprocessing as mp
from diffusers import UNet2DModel

import sys
sys.path.append("..")
from ckpt_util import get_ckpt_path

from diffusion_models._para_samplers import *
from diffusion_models.sampling_schedule import *
from diffusion_models.diffusionmodel_test import *

from network_arch.unets.unet import WrapUNet2DModel, WrapUNetModel, UNet2DModel_DDPM
from network_arch.ema_ddpm_unet import Model, WrapModel

from datasets.data_generator import CIFAR10_dataloader


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])



def run(rank, total_ranks, queues, seeds, 
        diffusion_loss='cdl', dataset_name='cifar10', 
        schedule_name='ddpm', num_inference_steps=1000, batch_size=64, tolerance=1e-1, 
        outdir='./para_fid/CIFAR10_ddpm'):
    """
    Args: 
        rank: GPU:No, which GPU node
        total_ranks: number of total GPUs to be used
        queues: in total two queues for parallel computations, queues[0] for main-loop, queues[1] for parallel workers to store results
        diffusionmodel: pretrained diffusion model, we sample from its distribution
    """
    # 1. Load pretrained diffusionmodels
    train_dl, train_ds, val_dl, val_ds, x_shape = CIFAR10_dataloader(batch_size=64, data_fp='../datasets/CIFAR10')
    print(f'x_shape = {x_shape}')

    if dataset_name.lower() == 'cifar10':
        model_id = "google/ddpm-cifar10-32"
        modelB = UNet2DModel.from_pretrained(model_id)

        if diffusion_loss.lower() == 'ddpm':
            denoiser = UNet2DModel_DDPM(**modelB.config) #  WrapUNet2DModel(**modelB.config)
            denoiser.load_state_dict(modelB.state_dict())
            dm = DDPM_DiffusionModel(denoiser, x_shape=x_shape) # wrap the pretrained model

        elif diffusion_loss.lower() == 'cdl':
            denoiser = WrapUNet2DModel(**modelB.config)
            ckpt_path = "../lightning_logs/nonema_ddpm_pretrained_cdl/version_7/checkpoints/epoch=9-step=3910.ckpt" 
            checkpoint = t.load(ckpt_path)

            cont_weight = checkpoint['hyper_parameters']['cdl_loss_weight']
            diff_weight = checkpoint['hyper_parameters']['std_loss_weight']
            print(f'cdl : std = {cont_weight} : {diff_weight} ')
            logsnr_loc, logsnr_scale = checkpoint['hyper_parameters']['logsnr_loc'], checkpoint['hyper_parameters']['logsnr_scale']
            print(f'logsnr loc = {logsnr_loc}, scale = {logsnr_scale} ')

            dm = CDL_DiffusionModel(denoiser, **checkpoint['hyper_parameters'])
            dm.load_state_dict(checkpoint['state_dict'])

    elif dataset_name.lower() == 'ema_cifar10':
        print(f'Loading EMA-CIFAR10...')
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
        if diffusion_loss.lower() == 'cdl':
            denoiser = WrapModel(**cifar10_cfg)
            ckpt_path = "../lightning_logs/cdl/version_7/checkpoints/epoch=9-step=3910.ckpt" 
            checkpoint = t.load(ckpt_path)

            cont_weight = checkpoint['hyper_parameters']['cdl_loss_weight']
            diff_weight = checkpoint['hyper_parameters']['std_loss_weight']
            print(f'cdl : std = {cont_weight} : {diff_weight} ')
            logsnr_loc, logsnr_scale = checkpoint['hyper_parameters']['logsnr_loc'], checkpoint['hyper_parameters']['logsnr_scale']
            print(f'logsnr loc = {logsnr_loc}, scale = {logsnr_scale} ')

            dm_model = CDL_DiffusionModel_EMA(denoiser, **checkpoint['hyper_parameters'])
            dm_model.model = dm_model.ema_model
            print(dm_model)
            dm_model.load_state_dict(checkpoint['state_dict'])
            dm = dm_model
            

        elif diffusion_loss.lower() == 'ddpm':
            denoiser = Model(**cifar10_cfg)

            # load the pretrained model
            ckpt_path = get_ckpt_path(dataset_name) # ema_cifar10
            print(f"Loading checkpoint {ckpt_path}")
            ckpt = torch.load(ckpt_path)
            denoiser.load_state_dict(ckpt)

            dm = DDPM_DiffusionModel(denoiser, x_shape)
        

    dm.eval()

    num_consumers = total_ranks
    PARALLEL = 32 # NOTE: in the original paper they use 32
    
    # 2. Get sampling schedule
    if schedule_name.lower() == 'ddpm':
        scheduler = DDPM_Scheduler(num_inference_timesteps=num_inference_steps)
    elif schedule_name.lower() == 'ddim':
        scheduler = DDIM_Scheduler(num_inference_timesteps=num_inference_steps)

    # 3. Run parallel sampling
    print(f'len(seeds) = {len(seeds)}')
    num_batches = (len(seeds) // (batch_size))
    if num_batches == 0:
        all_batches = (torch.as_tensor(seeds), )
    else:
        all_batches = torch.as_tensor(seeds).tensor_split(num_batches)

    # For testing FID score
    stats_pass = []
    stats_time = []
    # cnt=0
    for batch_seeds in all_batches:
        cur_bz_size = len(batch_seeds)
        if rank != -1:
            # parallel GPU workers, not the main loop
            dm.to(f"cuda:{rank}")
            parasam_forward_worker(model=dm, mp_queues=queues, device=f"cuda:{rank}")
        else:
            # GPU:0 runs the main loop
            dm = dm.to(f"cuda:0")

            rnd = StackedRandomGenerator(device="cuda:0", seeds=batch_seeds)
            
            # warmup, do not use these outputs
            print(f'Warmup call of parasam_forward() ...')
            _, _ = parasam_forward(dm, scheduler, 
                                   randn=rnd.randn,
                                   diffusionmodel_name=diffusion_loss.lower(), 
                                   num_samples=cur_bz_size,
                                   parallel=PARALLEL, 
                                   tolerance=tolerance, mp_queues=queues, 
                                   device=f"cuda:0", 
                                   num_consumers=num_consumers)
            
            # generate samples, run the main loop
            print(f'Parallel sampling ...')
            samples, stats = parasam_forward(dm, scheduler, 
                                             randn=rnd.randn, 
                                             diffusionmodel_name=diffusion_loss.lower(), 
                                             num_samples=cur_bz_size, 
                                             parallel=PARALLEL, 
                                             tolerance=tolerance, mp_queues=queues, 
                                             device=f"cuda:0", 
                                             num_consumers=num_consumers)
            
            # Save the generated samples 
            images_np = (samples * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for seed, image_np in zip(batch_seeds, images_np):
                image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}')
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join(image_dir, f'{seed:06d}.png')
                if image_np.shape[2] == 1:
                    PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
                else:
                    PIL.Image.fromarray(image_np, 'RGB').save(image_path)

            print(f'pass_count = {stats["pass_count"]}')
            print(f'time = {stats["time"]}')
            print(f'current batch size = {cur_bz_size}')

            stats['batch_size'] = cur_bz_size

            stats_file_path = os.path.join(outdir, f'stats.jsonl')
            with open(stats_file_path, 'a') as json_file:
                json.dump(stats, json_file)
                json_file.write('\n')

            # shutdown workers
            for _ in range(total_ranks):
                queues[0].put(None) 
                


def main(args): 
    torch.autograd.set_detect_anomaly(True)
    mp.set_start_method('spawn', force=True)
    queues = mp.Queue(), mp.Queue() # Yunshu: actually I think only need two queues, since no need to get user inputs from keyboard

    processes = []
    num_processes = torch.cuda.device_count()

    for rank in range(-1, num_processes):
        p = mp.Process(target=run, args=(rank, num_processes, queues, args.seeds, args.diffusion_loss, args.dataset, args.schedule_name, args.num_inference_steps, args.batch_size, args.tolerance, args.outdir))
        p.start()
        processes.append(p)

    # wait for all subprocesses to finish
    for p in processes:
        p.join()


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
    parser.add_argument("--dataset", type=str, default="ema_cifar10", choices=["ema_cifar10", "cifar10"], help="Type of the dataset to be used.")
    # number of samples to generate
    parser.add_argument("--schedule_name", type=str, default="ddpm", choices=["ddpm", "ddim"], help="Which sampling scheduler to use.")
    parser.add_argument("--num_inference_steps", type=int, default=1000, help="Number of inference steps.")
    parser.add_argument("--batch_size", type=int, default=64, help="Maximum number of batch samples to generate from the pretrained diffusion model.")
    parser.add_argument("--seeds", type=parse_int_list, default='0-63', help="Random seeds (e.g. 1,2,5-10)")
    parser.add_argument("--tolerance", type=float, default=1e-1, help="Tolerance of parallel sampling, controls when to stop iterations by picard iteration's convergence.")
    parser.add_argument("--outdir", type=str, help="Path to store the images generated by parallel sampling.")
    # parser.add_argument("--mmd_threshold", type=float, default=1e-4, help="MMD threshold of parallel sampling, controls when to stop iterations by MMD score. Provice 1e10 if you don't want set this control")

    args = parser.parse_args()
    # assert len(args.seeds) >= args.batch_size, "Sampling batch-size should be smaller than "

    main(args=args)