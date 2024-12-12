"""
The sampling schedule, specified by the sampler
"""

import math 
import numpy as np
import torch

import sys
sys.path.append("..")
import utils



class DDPM_Scheduler:
    def __init__(self, num_inference_timesteps=1000, st_schedule=0.0001, ed_schedule=0.02):
        self.num_inference_timesteps = num_inference_timesteps
        self.betas = torch.linspace(st_schedule, ed_schedule, num_inference_timesteps) 
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        # follows hugging-face DDPM way of scheduling, it's reverse ordering of the original paper; 
        # 1D tensor, shape=[num_inference_timesteps],  schedule in increasing order
        self.schedule = torch.from_numpy(np.arange(0, num_inference_timesteps)[::-1].copy()) 
        
        self._is_ode_scheduler = False

    def get_sampling_schedule(self, device):
        """schedules should be a sequence of integers"""
        return self.schedule.to(device) # put timestep schedule onto device, this is important since this is parallel sampling GPU:No matters a lot


    def get_variance(self, t):
        """Get variance at timestep t, variance[t]."""
        prev_t = t - 1 # NOTE: hugging-face way of implementing is prev_t = t - num_train_timesteps//num_inference_steps

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # For t > 0, compute predicted variance \bar beta_t=βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample: 
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # we always take the log of variance, so clamp it to ensure it's not 0!
        variance = torch.clamp(variance, min=1e-20)

        return variance


    def get_timestep(self, st, ed):
        """Get timesteps in range [st,ed), excluding ed"""
        return self.schedule[st:ed]


    def sample_denoise(
            self, 
            model_output: torch.FloatTensor,            # output of diffusionmodel(x_t, t), in CDL's way of defining, this output should be eps_hat
            timesteps: torch.Tensor, 
            block_samples: torch.FloatTensor,           # this sample is x_t (block_samples), not the clean data, this is the initial sample value x_t^k in the current iteration k
    ):
        """This predict the previous sample µ_t by reversing the SDE, or say the mean of the forward process posterior. 其实这个我没有完全明白为什么这东西能算drift, 但的确是这么算的
            Check formula (7) from https://arxiv.org/pdf/2006.11239.pdf"""
        t = timesteps
        num_inference_steps = self.num_inference_timesteps 
        prev_t = t - 1 

        left = (-1,) + (1,) * (model_output.ndim - 1) # model_output.shape = [(parallel_len*num_samples), x_shape]
        t = t.view(left) # prepare broadcasting
        prev_t = prev_t.view(left)

        # 1. compute alphas, betas
        self.alphas_cumprod = self.alphas_cumprod.to(model_output.device)
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[ torch.clip(prev_t, min=0) ] # clip the prev_t to min=0, prevent prev_t<0 which is an invalid timestep
        alpha_prod_t_prev[ prev_t < 0 ] = torch.tensor(1.0) # for all previous_timestep < 0, which means the current t is the starting t, alpha_prod_t_prev=1, corresponding to only data no noise

        one_minus_alpha_prod_t = 1 - alpha_prod_t # check formula (7)
        one_minus_alpha_prod_t_prev = 1 - alpha_prod_t_prev 
        alpha_t = alpha_prod_t / alpha_prod_t_prev # alpha_t = alpha_proc_t / alpha_prod_{t-1}
        beta_t = 1 - alpha_t # by definition: alpha_t = 1 - beta_t, Yunshu: i don't understand why they named 1-alpha_cumprod as beta???? weird

        # 2. Compute estimated original sample from estimated noise, since in my setting network output is epsilon
        est_original_sample = (block_samples - (one_minus_alpha_prod_t ** 0.5) * model_output) / (alpha_prod_t ** 0.5) # check formula (15)

        # 3. Compute coefficients for pred_original_sample: x_0, and current sample block_samples: x_t
        # see formula (7)
        est_original_sample_coeff = ((alpha_prod_t_prev **0.5) * beta_t) / one_minus_alpha_prod_t # coeff for x_0
        current_sample_coeff = ((alpha_t ** 0.5) * one_minus_alpha_prod_t_prev) / one_minus_alpha_prod_t # coeff for x_t

        # 4. Compute estimated previous sample mu_t
        est_prev_sample = est_original_sample_coeff * est_original_sample + current_sample_coeff * block_samples

        return est_prev_sample
    
