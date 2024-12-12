"""
In this py file, we show how to do parallel sampling (a simplified ParaSam Algo) given a pretrained diffusion model 'diffusionmodel'

Parallel sampling still use multiprocessing as parallel computation, using producer/consumer design.
We spawn a single producer process on GPU:0 to run the main loop of ParaSam, and all GPUs (including GPU:0) are as consumer processes, running parallel part of the ParaSam
"""
import argparse
import json

import torch
import torch.nn as nn
import types
import torch.multiprocessing as mp

import sys
sys.path.append("..")
from diffusion_models.para_samplers_visualization import *
from diffusion_models.sampling_schedule import *
from diffusion_models.diffusionmodel import *
from network_arch.nets import *
from datasets.gmm import GMMDist

def run(rank, total_ranks, queues, 
        diffusion_loss='cdl', 
        schedule_name='ddpm', num_inference_steps=1000, num_samples=1, 
        tolerance=1e-1, mmd_threshold=1e-4):
    """
    Args: 
        rank: GPU:No, which GPU node
        total_ranks: number of total GPUs to be used
        queues: in total two queues for parallel computations, queues[0] for main-loop, queues[1] for parallel workers to store results
        diffusionmodel: pretrained diffusion model, we sample from its distribution
    """
    # 1. Load pretrained diffusionmodels
    denoiser = MLP(1, 64, 3, activation=nn.ReLU, dropout=0.)

    if diffusion_loss.lower() == 'cdl':
        print(f'Loading CDL contrastive diffusion loss trained 1D two-mode dataset...')
        ckpt_path = '../lightning_logs/1D_gaussian_cdl/version_2/checkpoints/epoch=29-step=23460.ckpt'
        ckpt = torch.load(ckpt_path)
        dm = CDL_DiffusionModel(denoiser, **ckpt['hyper_parameters'])
    elif diffusion_loss.lower() == 'ddpm':
        print(f'Loading DDPM diffusion loss trained 1D two-mode dataset...')
        ckpt_path = '../lightning_logs/1D_gaussian_ddpm/version_1/checkpoints/epoch=199-step=156400.ckpt'
        ckpt = torch.load(ckpt_path)
        dm = DDPM_DiffusionModel(denoiser, **ckpt['hyper_parameters'])

    dm.load_state_dict(ckpt['state_dict'])
    dm.eval()
        
    if diffusion_loss.lower() == 'cdl':
        print(f'loc = {dm.hparams.logsnr_loc}')
        print(f'scale = {dm.hparams.logsnr_scale}')
        
    # 1.2 If checking MMD, need ground truth dataset
    MMD_FLAG = 1e9
    if mmd_threshold < MMD_FLAG:
        mix_onedim_gauss = GMMDist(dim=1, mix_probs=0.5, means=5)
        torch.manual_seed(42)
        true_samples = mix_onedim_gauss.sample((num_samples,))
        print(f'Get ground truth data samples...')

    num_consumers = total_ranks
    
    # Visualization: choose a fixed prior point, plot it's trajectories among K picard iterations
    seed = 42
    
    # 2. Get sampling schedule
    if schedule_name.lower() == 'ddpm':
        scheduler = DDPM_Scheduler(num_inference_timesteps=num_inference_steps)

    # 3. Run parallel sampling
    if rank != -1:
        # parallel GPU workers, not the main loop
        dm.to(f"cuda:{rank}")
        parasam_forward_worker(model=dm, mp_queues=queues, device=f"cuda:{rank}")
    else:
        # GPU:0 runs the main loop
        dm = dm.to(f"cuda:0")
        
        # warmup, do not use these outputs
        print(f'Warmup call of parasam_forward() ...')
        if mmd_threshold < MMD_FLAG:
            _, _, _ = parasam_forward_mmd(dm, scheduler, diffusionmodel_name=diffusion_loss, 
                                          num_samples=num_samples, 
                                          true_samples=true_samples,
                                          tolerance=tolerance,
                                          mmd_threshold=mmd_threshold, 
                                          mp_queues=queues, 
                                          device=f"cuda:0", 
                                          num_consumers=num_consumers, 
                                          seed=seed)
        else: # No MMD calculation required
            _, _, _ = parasam_forward(dm, scheduler, diffusionmodel_name=diffusion_loss,
                                      num_samples=num_samples, 
                                      tolerance=tolerance, mp_queues=queues, 
                                      device=f"cuda:0", 
                                      num_consumers=num_consumers, 
                                      seed=seed)

        # generate samples, run the main loop
        print(f'Parallel sampling ...')
        # NOTE: trajectory_buffer only store the very first sample in the whole batch! This is for visualization
        if mmd_threshold < MMD_FLAG:
            samples, stats, trajectory_buffer = parasam_forward_mmd(dm, scheduler, diffusionmodel_name=diffusion_loss, 
                                                                    num_samples=num_samples, 
                                                                    true_samples=true_samples,
                                                                    tolerance=tolerance,
                                                                    mmd_threshold=mmd_threshold, 
                                                                    mp_queues=queues, 
                                                                    device=f"cuda:0", 
                                                                    num_consumers=num_consumers, 
                                                                    seed=seed)
        else: # No MMD calculation required
            samples, stats, trajectory_buffer = parasam_forward(dm, scheduler, diffusionmodel_name=diffusion_loss, 
                                                                num_samples=num_samples, 
                                                                tolerance=tolerance, mp_queues=queues, 
                                                                device=f"cuda:0", 
                                                                num_consumers=num_consumers, 
                                                                seed=seed)
            
        # save these generated samples, plot the 2D points shape to check if sampling is correct
        # also need to check the final mmd, comparing with the ground-truth training samples
        torch.save(samples, f'./results_{schedule_name}_scheduler/parallel_sampling_{diffusion_loss}_1DGauss.pt')
        torch.save(trajectory_buffer, f'./results_{schedule_name}_scheduler/trajectory_buffer_parallel_sampling_{diffusion_loss}_1DGauss.pt')
        # store stats to file
        stats_file_path = './results/stats_{}_1DGauss.json'.format(diffusion_loss)
        with open(stats_file_path, 'w') as json_file:
            json.dump(stats, json_file)
        print(f'Generated samples and stats saved to {stats_file_path}.')


        # shutdown workers
        for _ in range(total_ranks):
            queues[0].put(None) 


def main(args): 
    torch.autograd.set_detect_anomaly(True)
    mp.set_start_method('spawn', force=True)
    queues = mp.Queue(), mp.Queue() 

    processes = []
    num_processes = torch.cuda.device_count()

    for rank in range(-1, num_processes):
        p = mp.Process(target=run, args=(rank, num_processes, queues, args.diffusion_name, args.schedule_name, args.num_inference_steps, args.num_samples, args.tolerance, args.mmd_threshold))
        p.start()
        processes.append(p)

    # wait for all subprocesses to finish
    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model Choice
    parser.add_argument("--diffusion_name", type=str, default="cdl", choices=["cdl", "itd", "ddpm", "ddim", "edm"], help="Which diffusion model to be used.")
    parser.add_argument("--schedule_name", type=str, default="ddpm", help="Which sampling scheduler to use.")
    parser.add_argument("--num_inference_steps", type=int, default=1000, help="Number of inference steps.")
    parser.add_argument("--num_samples", type=int, default=6666, help="Number of samples to generate from the pretrained diffusion model.")
    parser.add_argument("--tolerance", type=float, default=1e-1, help="Tolerance of parallel sampling, controls when to stop iterations.")
    parser.add_argument("--mmd_threshold", type=float, default=1e-4, help="MMD threshold of parallel sampling, controls when to stop iterations by MMD score. Provice 1e10 if you don't want set this control")

    args = parser.parse_args()

    main(args=args)