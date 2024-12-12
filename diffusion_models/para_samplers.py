"""
Implementation of parallel sampling from paper: https://arxiv.org/abs/2305.16317 
This is a simplified version, which doesn't use sliding window since 2D data is small enough to fit GPU memory
"""

from typing import Any, Callable, Dict, List, Optional, Union

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
    model: Any,                                         # on GPU:0! the pretrained diffusion model, Yunshu: I don't wanna specify the class name for this argument, so use 'Any' meaning any type is acceptable
    scheduler: Any,                                     # scheduler class
    randn: Any,                                         # to generate the initial prior samples by given batch_seeds
    diffusionmodel_name: str = "cdl",                   # which diffusion loss is running
    num_samples: int = 1,                               # number of samples to generate
    tolerance: float = 0.1,                             # tolerance when picard iteration converges
    mp_queues: Optional[torch.FloatTensor] = None,      # multiprocess queues
    device: Optional[str] = None,                       # device for the main loop
    num_consumers: Optional[int] = int,                 # number of GPUs to be used
):
    """Main loop of ParaSampler."""
    print("parallel pipeline!", flush=True)
    
    # 1. Prepare sampling schedule
    schedule = scheduler.get_sampling_schedule(device=device) # no matter which one is used (logsnr/sigma/timesteps), all of them are indexed by timestep! And use get_XXX() function to get values.

    # 2. Sample from prior ~ N(0,I), shape=[num_samples, *model.shape]
    x_init = randn([num_samples, *model.shape], device=device) # NOTE: in order to test the FID with different seed, this randn should be generated from outside and passed to this current function


    # 3. Denoising loop
    stats_pass_count = 0 # number of picard iterations till converge
    stats_flop_count = 0 # total number of model network evaluations
    
    # HACK: this ParaSamp algo doesn't have sliding window, for 2D case only
    begin_idx = 0 # starting index of the un-converged points on whole trajectory
    end_idx = len(schedule)  # number of inference steps
    samples_time_evolution_buffer = torch.stack([x_init] * (len(schedule) + 1)) # buffer.shape=[T+1, *x_init.shape], buffer[-1] is the final sample
    
    # ParaSamp Algo line-2: the up-front sampling of noise (for SDE), generate before parallel sampling
    noise_array = torch.zeros_like(samples_time_evolution_buffer) # the last entry noise_array[-1] should be zero, since no need to add noise to the final sample
    for j in range(len(schedule)): 
        base_noise = torch.randn_like(x_init)
        var_j = scheduler.get_variance(scheduler.schedule[j]) # NOTE get the variance schedule[j], different sampler has different way of calculating var_j
        noise = (var_j ** 0.5) * base_noise
        noise_array[j] = noise.clone() 

    # We will be dividing the norm of the noise, so we store its inverse here to avoid a division at every step
    inverse_variance_norm = 1. / torch.tensor([scheduler.get_variance(scheduler.schedule[j]) for j in range(len(schedule))] + [0]).to(noise_array.device) # FIXME: why append [0] to the end? why zero?
    sample_dim = model.d # sample_dim = noise_array[0,0].numel() # calculate data dim D = model.d, no need to calculate again
    inverse_variance_norm = inverse_variance_norm[:,None] / sample_dim # shape = [T+1, 1]. # [:,None]: This part is known as array slicing. It selects all rows of the array (or tensor) and adds a new axis of size 1 at the end.

    scaled_tolerance = (tolerance ** 2)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    # Parallel/picard iterations, estimates the ground truth trajectory
    while begin_idx < len(schedule):
        # current part of trajectory which has not converged yet. parallel_len is at most T=len(schedule). 
        parallel_len = end_idx - begin_idx

        # 3.1 Prepare x and t
        block_samples = samples_time_evolution_buffer[begin_idx:end_idx] # current un-converged trajectory points, shape = [parallel_len, num_samples, *model.shape]
        block_t = scheduler.schedule[begin_idx:end_idx] # returns a 1D torch tensor, shape = [parallel_len]
        block_t = block_t[:,None].repeat(1, num_samples) # in order to call diffusionmodel(x,timestep), timestep need to have shape=[parallel_len, num_samples]. NOTE: timestep should not have [len_traje, num_samples, *model.shape], all timestep/logsnr/sigma which go to diffusionmodel(x,t) do not have *model.shape, its shape should be [len_traje, num_samples]
        t_vec = block_t

        start1 = torch.cuda.Event(enable_timing=True)
        end1 = torch.cuda.Event(enable_timing=True)

        # 3.2 Calculate p_theta(block_samples, t_vec) in parallel
        # record parallel computing time cost
        start1.record()

        chunks = torch.arange(parallel_len).tensor_split(num_consumers) # Assign each consumer a part of the current un-converged trajectory. Yunshu: create a 1-dim tensor=[0,1,..., (parallel_len-1)], then split it into num_consumers chunks
        num_chunks = min(parallel_len, num_consumers) # in case in final round, number un-converged points < num_consumers

        for i in range(num_chunks):
            mp_queues[0].put(
                (block_samples, t_vec, chunks[i], i, begin_idx)
            )
        
        model_output = [None for _ in range(num_chunks)]
        for i in range(num_chunks):
            ret = mp_queues[1].get() 
            model_output_chunk, idx = ret # model_output_chunk is estimated noise eps_hat, this is CDL's way of parameterize diffusion model
            model_output[idx] = model_output_chunk.to(device) # put to main-loop's device
            del ret
            
        model_output = torch.cat(model_output)

        end1.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()

        print(f'current window: [{begin_idx}, {end_idx})', flush=True)
        elapsed = start1.elapsed_time(end1)
        elapsed_per_t = elapsed / parallel_len
        print(f'elapsed = {elapsed}, elapsed_per_t = {elapsed_per_t}')

        model_output = model_output.reshape(parallel_len * num_samples, *model.shape) 


        # 3.3 Calcuate drifts
        # compute mu_t, the mean of the forward process posterior q(x_t-1 | x_t,x_0) = N(x_t-1; mu_t, beta*I) /or the previous sample
        block_samples_denoise = scheduler.sample_denoise(
            model_output=model_output, # output of diffusionmodel(block_samples, block_t), is estimated noise eps_hat, shape=[parallel_len*num_samples, *model.shape]
            timesteps=block_t.flatten(0,1), # t, in shape = [parallel_len * num_samples]
            block_samples=block_samples.flatten(0,1), # x_t, in shape=[parallel_len * num_samples, *model.shape]
        ).reshape(block_samples.shape)


        delta = block_samples_denoise - block_samples # drift
        cumulative_delta = torch.cumsum(delta, dim=0) # shape = [len_traje, num_samples, *model.shape]
        cumulative_noise = torch.cumsum(noise_array[begin_idx:end_idx], dim=0)

        # if we are using ODE-like schduler, we don't need to add noise
        if scheduler._is_ode_scheduler:
            cumulative_noise = 0.

        # NOTE: Algo line-6～9
        block_samples_new = samples_time_evolution_buffer[begin_idx][None,] + cumulative_delta + cumulative_noise # line-6 # NOTE: buffer[idx][None,].shape=[1, *buffer.shape[1:]], adds a new dim to the beginning
        cur_err_vec = ( block_samples_new - samples_time_evolution_buffer[begin_idx+1:end_idx+1] ).reshape(parallel_len, num_samples, -1) # reshape the error tensor to calculate norm of each single data sample
        cur_err = torch.linalg.norm(cur_err_vec, dim=-1).pow(2) # error per data sample; cur_err.shape = [parallel_len, num_samples]
        err_ratio = cur_err * inverse_variance_norm[begin_idx+1:end_idx+1] # calculate the inverse_var_norm=1/(var_j^2*D) since: err_j/D > (tau^2 * var_j^2). 

        # 3.4 Find the first index of the vector err_ratio that is grater than error tolerance, move the left point to this point
        err_ratio = torch.nn.functional.pad(
            err_ratio, (0, 0, 0, 1), value=1e9 # pad err=1e9 to the bottom of the err_ratio[parallel_len, num_samples] --> err_ratio[parallel_len+1, num_samples]
        )  # handle the case when everything is below ratio, by padding the end of parallel_len dimension
        any_err_at_time = torch.max(err_ratio > scaled_tolerance, dim=1).values.int()
        ind = torch.argmax(any_err_at_time).item() # ind is the local index of the first point which is not converged in this round; ind.max=parallel_len+1, means all points converged, in total parallel_len of them

        # compute the new begin, no need for updating the end, since no sliding window
        num_converged_points = min((1 + ind), parallel_len)
        new_begin_idx = begin_idx + num_converged_points 
        
        # store the computed latents for the current update in the global buffer
        samples_time_evolution_buffer[begin_idx+1:end_idx+1] = block_samples_new
        # since we are not using sliding window, no need for initializing new sliding window samples by the end of the current window

        begin_idx = new_begin_idx

        stats_pass_count += 1 # number of picard iterations ++
        stats_flop_count += parallel_len # in this picard iteration, we do "parallel_len" network evaluations
        print("\n")

    final_samples = samples_time_evolution_buffer[-1]

    print("pass count", stats_pass_count)
    print("flop count", stats_flop_count)

    end.record()

    # 4. Waits for everything to finish running 
    torch.cuda.synchronize()

    print(start.elapsed_time(end))
    print("done", flush=True)

    stats = {
        'pass_count': stats_pass_count, 
        'flops_count': stats_flop_count, 
        'time': start.elapsed_time(end), # for real-world dataset, we can report sec generating per image, but in 2D case only 'pass_count' is enough
    }

    return final_samples, stats


@torch.no_grad()
def parasam_forward_mmd(
    model: Any,                                         # on GPU:0! the pretrained diffusion model, Yunshu: I don't wanna specify the class name for this argument, so use 'Any' meaning any type is acceptable
    scheduler: Any,                                     # scheduler class
    randn: Any,                                         # to generate the initial prior samples by given batch_seeds
    diffusionmodel_name: str = "cdl",                   # which diffusion loss is running
    num_samples: int = 1,                               # number of samples to generate
    true_samples: Optional[torch.FloatTensor] = None,   # ground truth samples, shape = [num_samples, *model.shape]
    tolerance: float = 0.1,                             # tolerance when picard iteration converges
    mmd_threshold: float = 4e-4,                        # NOTE: set this MMD threshold since "tolerance" controls both convergence speed and the sample quality, set a MMD threshold can show how fast each model converges given a fixed sample quality
    mp_queues: Optional[torch.FloatTensor] = None,      # multiprocess queues
    device: Optional[str] = None,                       # device for the main loop
    num_consumers: Optional[int] = int,                 # number of GPUs to be used
):
    """Main loop of ParaSampler, this one comes with MMD calculation, if reaches the MMD threashold then converge!"""
    print("parallel pipeline!", flush=True)

    mmd_flag = False
    mmd_buffer = []
    BW = torch.tensor(3e-02) # this is for real-dino dataset
    
    # 1. Prepare sampling schedule
    schedule = scheduler.get_sampling_schedule(device=device) # no matter which one is used (logsnr/sigma/timesteps), all of them are indexed by timestep! And use get_XXX() function to get values.

    # 2. Sample from prior ~ N(0,I), shape=[num_samples, *model.shape]
    x_init = randn([num_samples, *model.shape], device=device) # NOTE: in order to test the FID with different seed, this randn should be generated from outside and passed to this current function

    # 3. Denoising loop
    stats_pass_count = 0 # number of picard iterations till converge
    stats_flop_count = 0 # total number of model network evaluations
    
    # HACK: this ParaSamp algo doesn't have sliding window, for 2D case only
    begin_idx = 0 # starting index of the un-converged points on whole trajectory
    end_idx = len(schedule)  # number of inference steps
    samples_time_evolution_buffer = torch.stack([x_init] * (len(schedule) + 1)) # buffer.shape=[T+1, *x_init.shape], buffer[-1] is the final sample
    
    # ParaSamp Algo line-2: the up-front sampling of noise (for SDE), generate before parallel sampling
    noise_array = torch.zeros_like(samples_time_evolution_buffer) # the last entry noise_array[-1] should be zero, since no need to add noise to the final sample
    for j in range(len(schedule)): 
        base_noise = torch.randn_like(x_init)
        var_j = scheduler.get_variance(scheduler.schedule[j]) # NOTE get the variance schedule[j], different sampler has different way of calculating var_j
        noise = (var_j ** 0.5) * base_noise
        noise_array[j] = noise.clone() # FIXME: why need .clone() here???

    # We will be dividing the norm of the noise, so we store its inverse here to avoid a division at every step
    inverse_variance_norm = 1. / torch.tensor([scheduler.get_variance(scheduler.schedule[j]) for j in range(len(schedule))] + [0]).to(noise_array.device) # FIXME: why append [0] to the end? why zero?
    sample_dim = model.d # sample_dim = noise_array[0,0].numel() # calculate data dim D = model.d, no need to calculate again
    inverse_variance_norm = inverse_variance_norm[:,None] / sample_dim # shape = [T+1, 1]. # [:,None]: This part is known as array slicing. It selects all rows of the array (or tensor) and adds a new axis of size 1 at the end.

    scaled_tolerance = (tolerance ** 2)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    # Parallel/picard iterations, estimates the ground truth trajectory
    while begin_idx < len(schedule):
        # current part of trajectory which has not converged yet. parallel_len is at most T=len(schedule). 
        parallel_len = end_idx - begin_idx

        # 3.1 Prepare x and t
        block_samples = samples_time_evolution_buffer[begin_idx:end_idx] # current un-converged trajectory points, shape = [parallel_len, num_samples, *model.shape]
        block_t = scheduler.schedule[begin_idx:end_idx] # returns a 1D torch tensor, shape = [parallel_len]
        block_t = block_t[:,None].repeat(1, num_samples) # in order to call diffusionmodel(x,timestep), timestep need to have shape=[parallel_len, num_samples]. NOTE: timestep should not have [len_traje, num_samples, *model.shape], all timestep/logsnr/sigma which go to diffusionmodel(x,t) do not have *model.shape, its shape should be [len_traje, num_samples]
        t_vec = block_t

        start1 = torch.cuda.Event(enable_timing=True)
        end1 = torch.cuda.Event(enable_timing=True)

        # 3.2 Calculate p_theta(block_samples, t_vec) in parallel
        # record parallel computing time cost
        start1.record()

        chunks = torch.arange(parallel_len).tensor_split(num_consumers) # Assign each consumer a part of the current un-converged trajectory. Yunshu: create a 1-dim tensor=[0,1,..., (parallel_len-1)], then split it into num_consumers chunks
        num_chunks = min(parallel_len, num_consumers) # in case in final round, number un-converged points < num_consumers

        for i in range(num_chunks):
            mp_queues[0].put(
                (block_samples, t_vec, chunks[i], i, begin_idx)
            )
        
        model_output = [None for _ in range(num_chunks)]
        for i in range(num_chunks):
            ret = mp_queues[1].get() 
            model_output_chunk, idx = ret # model_output_chunk is estimated noise eps_hat, this is CDL's way of parameterize diffusion model
            model_output[idx] = model_output_chunk.to(device) # put to main-loop's device
            del ret
        model_output = torch.cat(model_output)

        end1.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()

        print(f'current window: [{begin_idx}, {end_idx})', flush=True)
        elapsed = start1.elapsed_time(end1)
        elapsed_per_t = elapsed / parallel_len
        print(f'elapsed = {elapsed}, elapsed_per_t = {elapsed_per_t}')

        # reshape the model_output
        model_output = model_output.reshape(parallel_len * num_samples, *model.shape)

        # 3.3 Calcuate drifts
        # compute mu_t, the mean of the forward process posterior q(x_t-1 | x_t,x_0) = N(x_t-1; mu_t, beta*I) /or the previous sample
        block_samples_denoise = scheduler.sample_denoise(
            model_output=model_output, # output of diffusionmodel(block_samples, block_t), is estimated noise eps_hat, shape=[parallel_len*num_samples, *model.shape]
            timesteps=block_t.flatten(0,1), # t, in shape = [parallel_len * num_samples]
            block_samples=block_samples.flatten(0,1), # x_t, in shape=[parallel_len * num_samples, *model.shape]
        ).reshape(block_samples.shape)


        delta = block_samples_denoise - block_samples # drift
        cumulative_delta = torch.cumsum(delta, dim=0) # shape = [len_traje, num_samples, *model.shape]
        cumulative_noise = torch.cumsum(noise_array[begin_idx:end_idx], dim=0)

        # if we are using ODE-like schduler, we don't need to add noise
        if scheduler._is_ode_scheduler:
            cumulative_noise = 0.

        # NOTE: Algo line-6～9
        block_samples_new = samples_time_evolution_buffer[begin_idx][None,] + cumulative_delta + cumulative_noise # line-6 # NOTE: buffer[idx][None,].shape=[1, *buffer.shape[1:]], adds a new dim to the beginning
        cur_err_vec = ( block_samples_new - samples_time_evolution_buffer[begin_idx+1:end_idx+1] ).reshape(parallel_len, num_samples, -1) # reshape the error tensor to calculate norm of each single data sample
        cur_err = torch.linalg.norm(cur_err_vec, dim=-1).pow(2) # error per data sample; cur_err.shape = [parallel_len, num_samples]
        err_ratio = cur_err * inverse_variance_norm[begin_idx+1:end_idx+1] # calculate the inverse_var_norm=1/(var_j^2*D) since: err_j/D > (tau^2 * var_j^2). 

        # 3.4 Find the first index of the vector err_ratio that is grater than error tolerance, move the left point to this point
        err_ratio = torch.nn.functional.pad(
            err_ratio, (0, 0, 0, 1), value=1e9 # pad err=1e9 to the bottom of the err_ratio[parallel_len, num_samples] --> err_ratio[parallel_len+1, num_samples]
        )  # handle the case when everything is below ratio, by padding the end of parallel_len dimension
        any_err_at_time = torch.max(err_ratio > scaled_tolerance, dim=1).values.int()
        ind = torch.argmax(any_err_at_time).item() # ind is the local index of the first point which is not converged in this round; ind.max=parallel_len+1, means all points converged, in total parallel_len of them

        # compute the new begin, no need for updating the end, since no sliding window
        num_converged_points = min((1 + ind), parallel_len)
        new_begin_idx = begin_idx + num_converged_points 
        
        # store the computed latents for the current update in the global buffer
        samples_time_evolution_buffer[begin_idx+1:end_idx+1] = block_samples_new
        # since we are not using sliding window, no need for initializing new sliding window samples by the end of the current window

        begin_idx = new_begin_idx

        stats_pass_count += 1 # number of picard iterations ++
        stats_flop_count += parallel_len # in this picard iteration, we do "parallel_len" network evaluations

        # 4. Calculate MMD, for MMD tolerance checking
        def mmd(gt_samples, est_samples):
            bw = BW # for dino
            kyy = kernel(gt_samples, gt_samples, bw)
            kxx = kernel(est_samples, est_samples, bw)
            kxy = kernel(est_samples, gt_samples, bw)
            mmd_mean = mmd_ci_kernel(kxx, kxy, kyy)
            return mmd_mean
        current_mmd = mmd(true_samples.to(device), samples_time_evolution_buffer[-1])
        mmd_buffer.append(current_mmd)
        print(f'    MMD = {current_mmd}')
        if current_mmd <= mmd_threshold: # reach sample quality threashold! 
            print(f'Reach MMD threshold! ')
            mmd_threshold = -math.inf
            mmd_flag = True
            mmd_reach_iter = stats_pass_count
            # break

        print("\n")

    final_samples = samples_time_evolution_buffer[-1]

    print("pass count", stats_pass_count)
    print("flop count", stats_flop_count)

    end.record()

    # 4. Waits for everything to finish running 
    torch.cuda.synchronize()

    print(start.elapsed_time(end))
    print("done", flush=True)

    if mmd_flag:
        stats = {
            'pass_count': stats_pass_count, 
            'flops_count': stats_flop_count, 
            'time': start.elapsed_time(end), # for real-world dataset, we can report sec generating per image, but in 2D case only 'pass_count' is enough
            'MMD thredshold reached?': mmd_flag,
            'If MMD threshold reached, at No.iter': mmd_reach_iter, 
        }
    else:
        stats = {
            'pass_count': stats_pass_count, 
            'flops_count': stats_flop_count, 
            'time': start.elapsed_time(end), # for real-world dataset, we can report sec generating per image, but in 2D case only 'pass_count' is enough
            'MMD thredshold reached?': mmd_flag,
        }

    return final_samples, stats, mmd_buffer


@torch.no_grad()
def parasam_forward_worker(
    model: Any,                                             # pretrained diffusion model, call: model(x, t), here x.shape=[batch, *model.shape], t.shape=[batch]
    mp_queues: Optional[torch.FloatTensor] = None,          # multiprocessing queues
    device: Optional[str] = None,                           # which GPU worker takes this job
):
    """Parallel workers here"""
    while True:
        ret = mp_queues[0].get() # take inputs from queues[0], and then put results into queues[1]
        if ret is None: # no input, done! This means that all the workers shouldbe shutdown, check run() function
            del ret
            return
        
        # get inputs, diffusion model needs these input arguments to denoise
        # z, logsnrs: length equals to num_inference_steps, which corresponds to all points on the sampling trajectory
        (z, timesteps, chunk, idx, begin_idx) = ret
        

        # call diffusion model to denoise
        model_output_chunk = model(
            z[chunk].flatten(0,1).to(device), # z.shape = [len_traje, num_samples, *model.shape], need to flatten first two dim to [len_traje*num_samples, *model.shape] to have diffusionmodel(x,t) to handle
            timesteps[chunk].flatten(0,1).to(device), # timesteps.shape = [len_traje, num_samples], flatten first two dim!
        ) # model_output is the estimated noise, in my way of implementation

        del ret

        mp_queues[1].put(
            (model_output_chunk, idx) # idx tells which GPU owns this chunk
        )


