"""
Implementation of parallel sampling from paper:  https://arxiv.org/abs/2305.16317 
This is exactly the original algorithm, but fit our code structure
"""


from typing import Any, Optional

import torch
from tqdm.auto import trange
from functools import partial

from .sampling_schedule import * 
from .seq_samplers import t2logsnr

import sys
sys.path.append("..")
from utils import *


@torch.no_grad()
def parasam_forward(
    model: Any, 
    scheduler: Any, 
    randn: Any,                                         # to generate the initial prior samples by given batch_seeds
    diffusionmodel_name: str = "cdl",                   # which diffusion loss is running
    num_samples: int = 1,                               # number of samples to generate, NOTE: batch_size in the original paradigms_mp.py
    parallel: int = 32,                                 # sliding window size
    tolerance: float = 0.1,                             # tolerance when picard iteration converges
    mp_queues: Optional[torch.FloatTensor] = None,      # multiprocess queues
    device: Optional[str] = None,                       # device for the main loop
    num_consumers: Optional[int] = int,                 # number of GPUs to be used
):
    """Main loop of ParaSampler. """
    print("parallel pipeline!", flush=True)

    # 1. Prepare sampling schedule
    schedule = scheduler.get_sampling_schedule(device=device)

    # 2. Sample from prior ~ N(0,I), shape = [num_samples, *model.shape]
    # TODO: in order to test the FID with different seed, this randn should be generated from outside and passed to this current function
    x_init = randn([num_samples, *model.shape], device=device)

    # 3. Denoising loop
    stats_pass_count = 0 # number of picard iterations till converge
    stats_flop_count = 0 # total number of model network evaluations
    parallel = min(parallel, len(schedule)) # handle the case where the sliding window size is larger than sampling steps=scheduler.num_inference_timesteps

    begin_idx = 0
    end_idx = parallel # the end of the current window
    samples_time_evolution_buffer = torch.stack([x_init] * (len(schedule) + 1)) # buffer.shape=[T+1, *x_init.shape]

    noise_array = torch.zeros_like(samples_time_evolution_buffer) # line 2
    for j in range(len(schedule)):
        base_noise = torch.randn_like(x_init)
        var_j = scheduler.get_variance(scheduler.schedule[j]) # NOTE get the variance schedule[j], different sampler has different way of calculating var_j
        noise = (var_j ** 0.5) * base_noise
        noise_array[j] = noise.clone() # FIXME: why need .clone() here???

    # We will be dividing the norm of the noise, so we store its inverse here to avoid a division at every step
    inverse_variance_norm = 1. / torch.tensor([scheduler.get_variance(scheduler.schedule[j]) for j in range(len(schedule))] + [0]).to(noise_array.device) # FIXME: why append [0] to the end? why zero?
    sample_dim = model.d 
    sample_dim = noise_array[0,0].numel() # calculate data dim D = model.d, no need to calculate again

    assert model.d == noise_array[0,0].numel(), "diffusion model dimension calculation wrong!"
    inverse_variance_norm = inverse_variance_norm[:,None] / sample_dim # shape = [T+1, 1]. # [:,None]: This part is known as array slicing. It selects all rows of the array (or tensor) and adds a new axis of size 1 at the end.

    scaled_tolerance = (tolerance ** 2)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    while begin_idx < len(schedule):
        parallel_len = end_idx - begin_idx # current window size

        # 3.1 prepare x and t
        block_samples = samples_time_evolution_buffer[begin_idx:end_idx]
        block_t = scheduler.schedule[begin_idx:end_idx, None].repeat(1, num_samples)
        t_vec = block_t

        start1 = torch.cuda.Event(enable_timing=True)
        end1 = torch.cuda.Event(enable_timing=True)
        
        # 3.2 calculate mu_t(x_t,x_0) mean of forward posterior in parallel
        start1.record()

        chunks = torch.arange(parallel_len).tensor_split(num_consumers) # create a 1-dim tensor=[0, 1, ..., parallel_len-1], then .tensor_split() it into num_consumers chunks.
        num_chunks = min(parallel_len, num_consumers) # possibly the final window size is smaller than num_consumers

        for i in range(num_chunks):
            mp_queues[0].put(
                (block_samples, t_vec, chunks[i], i, begin_idx, diffusionmodel_name, scheduler.alphas_cumprod)
            ) 

        model_output = [None for _ in range(num_chunks)]
        for i in range(num_chunks):
            ret = mp_queues[1].get()
            model_output_chunk, idx = ret
            model_output[idx] = model_output_chunk.to(device)
            del ret

        model_output = torch.cat(model_output)

        end1.record()

        torch.cuda.synchronize()

        print(f'    current window: [{begin_idx}, {end_idx})', flush=True)
        elapsed = start1.elapsed_time(end1)
        elapsed_per_t = elapsed / parallel_len
        print(f'    elapsed = {elapsed}, elapsed_per_t = {elapsed_per_t}')

        model_output = model_output.reshape(parallel_len * num_samples, *model.shape)

        # 3.3 Calcuate drifts
        block_samples_denoise = scheduler.sample_denoise(
            model_output=model_output, 
            timesteps=block_t.flatten(0,1), 
            block_samples=block_samples.flatten(0,1), 
        ).reshape(block_samples.shape)

        delta = block_samples_denoise - block_samples
        cumulative_delta = torch.cumsum(delta, dim=0)
        cumulative_noise = torch.cumsum(noise_array[begin_idx:end_idx], dim=0)

        if scheduler._is_ode_scheduler:
            cumulative_noise = 0.

        # NOTE: Algo line-6～9
        block_samples_new = samples_time_evolution_buffer[begin_idx][None,] + cumulative_delta + cumulative_noise # line-6 # NOTE: buffer[idx][None,].shape=[1, *buffer.shape[1:]], adds a new dim to the beginning
        cur_err_vec = ( block_samples_new - samples_time_evolution_buffer[begin_idx+1:end_idx+1] ).reshape(parallel_len, num_samples, -1) # reshape the error tensor to calculate norm of each single data sample
        cur_err = torch.linalg.norm(cur_err_vec, dim=-1).pow(2) # error per data sample; cur_err.shape = [parallel_len, num_samples]
        err_ratio = cur_err * inverse_variance_norm[begin_idx+1:end_idx+1] # calculate the inverse_var_norm=1/(var_j^2*D) since: err_j/D > (tau^2 * var_j^2). 

        err_ratio = torch.nn.functional.pad(
            err_ratio, (0,0,0,1), value=1e9
        )
        any_err_at_time = torch.max(err_ratio > scaled_tolerance, dim=1).values.int()
        ind = torch.argmax(any_err_at_time).item()

        new_begin_idx = begin_idx + min(1+ind, parallel)
        new_end_idx = min(new_begin_idx + parallel, len(schedule))

        samples_time_evolution_buffer[begin_idx+1:end_idx+1] = block_samples_new
        # initialize the new sliding window latents with the end of the current window, should be better than random initialization
        samples_time_evolution_buffer[end_idx:new_end_idx+1] = samples_time_evolution_buffer[end_idx][None,]

        begin_idx = new_begin_idx
        end_idx = new_end_idx

        stats_pass_count += 1
        stats_flop_count += parallel_len
        print('\n')

    final_samples = samples_time_evolution_buffer[-1]
    print(f'final_samples shape = {final_samples.shape}')
    print(f'min, max = {final_samples.min()}, {final_samples.max()}')
    
    print("pass count", stats_pass_count)
    print("flop count", stats_flop_count)

    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    print(start.elapsed_time(end))
    print("done", flush=True)

    stats = {
        'pass_count': stats_pass_count,
        'flops_count': stats_flop_count,
        'time': start.elapsed_time(end),
    }

    return final_samples, stats



@torch.no_grad()
def parasam_forward_worker(
    model: Any,                                             # pretrained diffusion model, call: model(x, t), here x.shape=[batch, *model.shape], t.shape=[batch]
    mp_queues: Optional[torch.FloatTensor] = None,          # multiprocessing queues
    device: Optional[str] = None,                           # which GPU worker takes this job
):
    """Parallel GPU workers here"""
    while True:
        ret = mp_queues[0].get() # take inputs from queues[0], and then put results into queues[1]
        if ret is None: # no input, done! This means that all the workers shouldbe shutdown, check run() function
            del ret
            return
        
        # get inputs, diffusion model needs these input arguments to denoise
        # z, logsnrs: length equals to num_inference_steps, which corresponds to all points on the sampling trajectory
        (z, timesteps, chunk, idx, begin_idx, diffusionloss_name, alphas_cumprod) = ret

        if diffusionloss_name.lower() == 'ddpm':
            # call diffusion model to denoise
            model_output_chunk = model(
                z[chunk].flatten(0,1).to(device), # z.shape = [len_traje, num_samples, *model.shape], need to flatten first two dim to [len_traje*num_samples, *model.shape] to have diffusionmodel(x,t) to handle
                timesteps[chunk].flatten(0,1).to(device), # timesteps.shape = [len_traje, num_samples], flatten first two dim!
            ) # model_output is the estimated noise, in my way of implementation
        elif diffusionloss_name.lower() == 'cdl' or diffusionloss_name.lower() == 'itd':
            t_chunk = timesteps[chunk].flatten(0,1)
            logsnr_chunk = t2logsnr(t_chunk, alphas_cumprod.to(t_chunk.device))
            # call diffusion model to denoise
            model_output_chunk = model(
                z[chunk].flatten(0,1).to(device), # z.shape = [len_traje, num_samples, *model.shape], need to flatten first two dim to [len_traje*num_samples, *model.shape] to have diffusionmodel(x,t) to handle
                logsnr_chunk.to(device), # timesteps.shape = [len_traje, num_samples], flatten first two dim!
            ) # model_output is the estimated noise, in my way of implementation

        del ret

        mp_queues[1].put(
            (model_output_chunk, idx) # idx tells which GPU owns this chunk
        )
