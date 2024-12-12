import torch
from tqdm.auto import trange
from functools import partial

import sys
sys.path.append("..")
from utils import *


def prepreation_karras(device='cpu', logsnr_min=-8.76, logsnr_max=12.42, num_steps=18):
    sigma_min, sigma_max = logsnr2sigma(logsnr_max), logsnr2sigma(logsnr_min)
    schedule = get_sigmas_karras(num_steps, sigma_min, sigma_max, device=device)
    # s_churn = 0. is deterministic. Values of 0.01~0.15 were effective in Karras paper
    step_function = get_step(order=2, s_churn=0) # previous: order=2, s_churn=0.1
    return schedule, step_function


@torch.no_grad()
def sample_fn_ddpm(model, num_samples, start_schedule=0.0001, end_schedule=0.02, num_timesteps=1000):
    """DDPM sampler."""
    model.eval()

    x = torch.randn([num_samples, *model.shape], device=model.device)
    for i in reversed(range(num_timesteps)):
        timestep = torch.full((1,), i, dtype=torch.long, device=model.device)
        x = ddpm_step_fn(model, x, timestep)
    
    return x


@torch.no_grad()
def sample_fn_karras(model, num_samples, logsnr_min=-9., logsnr_max=12., num_steps=10):
    """For simple dataset, Karras sampler takes less than 40 NFEs."""
    model.eval()

    sigma_min, sigma_max = logsnr2sigma(logsnr_max), logsnr2sigma(logsnr_min)
    schedule = get_sigmas_karras(num_steps, sigma_min, sigma_max, device=model.device)
    # s_churn=0. is deterministic, ODE sampler. Values of 0.01~0.15 were effective in Karras paper (but this is for CIFAR/AFHQ/FFHQ)
    step_function = get_step(order=2, s_churn=0.1) # Yunshu: try hyper-parameter search

    x = torch.randn([num_samples, *model.shape], device=model.device) * schedule[0]

    for i in trange(len(schedule) - 1, desc='Generating images via Karras sampler'):
        x = step_function(model, x, schedule[i], schedule[i+1])

    return x 


#### DDPM Sampler Dependencies ####
def t2logsnr(timestep, alphas_cumprod):
    """ \bar{alpha}_t=alphas_cumprod = sigmoid(logsnr) """
    selected_alphas_cumprod = get_by_idx(alphas_cumprod, timestep)
    logsnr = torch.log(selected_alphas_cumprod / (1. - selected_alphas_cumprod)).to(timestep.device)
    return logsnr

def get_by_idx(values, timestep):
    """ pick the \bar{alpha}_t from alphas_cumprod, according to the indices stored in 'timestep' """
    selected_val = values.gather(-1, timestep.cpu())
    selected_val = selected_val.to(timestep.device)
    return selected_val # need further selected_val.view(left), this is w/o changing view for broadcasting

def ddpm_step_fn(model, x, timestep):
    betas_t = model.get_by_idx(model.betas, timestep)
    sqrt_one_minusalphas_cumprod = model.get_by_idx(torch.sqrt(1. - model.alphas_cumprod), timestep) # mixing ratio (1-alpha_bar_t)
    sqrt_recip_alphas_t = model.get_by_idx(torch.sqrt(1. / model.alphas), timestep)

    # translate t to logsnr
    selected_alphas_cumprod = model.get_by_idx(model.alphas_cumprod, timestep)
    logsnr = torch.log(selected_alphas_cumprod / (1. - selected_alphas_cumprod)).to(timestep.device)
    
    # Estimated noisy data x_t
    mean_t = sqrt_recip_alphas_t * (x - betas_t * model(x, logsnr) / sqrt_one_minusalphas_cumprod)
    posterior_var_t = betas_t

    if timestep == 0:
        return mean_t # if t=0, do not add noise
    else:
        eps = torch.randn_like(x)
        var_t = torch.sqrt(posterior_var_t) * eps
        return mean_t + var_t



#### Karras Sampler Dependencies ####
def logsnr2sigma(logsnrs):
    return torch.exp(-to_tensor(logsnrs) / 2)

def sigma2logsnr(sigmas):
    return -2 * torch.log(to_tensor(sigmas))


@torch.no_grad()
def generate(model, step_function, schedule, batch_size=1):
    """Generate a batch of images from a diffusion model and a noise schedule, specified in karras style in terms of sigma."""
    model.eval()
    x = torch.randn([batch_size, *model.shape], device=model.device) * schedule[0]
    for i in trange(len(schedule) - 1, desc='Generating images'):
        x = step_function(model, x, schedule[i], schedule[i+1])
    return x


@torch.no_grad()
def inpaint(model, step_function, schedule, x0, mask):
    """Inpaint images using a diffusion model and a binary mask, where '1' is the masked part to inpaint"""
    model.eval()
    x = torch.randn_like(x0) * schedule[0]
    for i in trange(len(schedule) - 1, desc='Inpainting images'):
        sigma = schedule[i]
        noisy_original = x0 + sigma * torch.rand_like(x0) # calculate the noisy x0 by given sigma
        x = x * mask + noisy_original * (1 - mask) # project onto original (noisified) for unmasked part
        x = step_function(model, x, schedule[i], schedule[i+1])
    x = x * mask + x0 * (1 - mask)
    return x


def translate_denoiser(model, x, sigma_hat):
    """Translate CDL denoiser (var presering) to karras conventions. """
    s_in = x.new_ones([x.shape[0]])
    logsnr = sigma2logsnr(sigma_hat)
    scale = torch.sqrt(torch.sigmoid(logsnr)) # "scale" of x_t in karras convention
    x = x * scale # Scale input to variance presering convention in CDL denoiser, karras's denosier is var exploding
    # print(f'    logsnr*s_in shape = {(logsnr*s_in).shape}')
    # print(f'    data.shape = {x.shape}')
    eps_hat = model(x, logsnr * s_in)
    denoised = x / torch.sqrt(torch.sigmoid(logsnr)) - torch.exp(-logsnr / 2) * eps_hat # predicts original x
    return denoised


def get_step(order=1, s_churn=0.):
    """Note that s_churn is defined as the term that Karras calls s_churn / (len(schedule) - 1). Yunshu: their way of defining s_churn is problematic, cannot directly extend"""
    return partial(stochastic_step, order=order, s_churn=s_churn)


@torch.no_grad()
def stochastic_step(model, x, sigma0, sigma1, order=1, s_churn=0., s_noise=1.):
    """Implements Algorithm *2* (and with s_churn=0, also Alg. 1) from Karras et al. (2022).
    Code assumes Karras conventions, with commented wrappers to use our denoiser"""
    gamma = min(s_churn, 2 ** 0.5 - 1)  # s_churn = 0 turns off noise
    # gamma = min(s_churn, 2 ** 0,5 - 1) if 0.01 <= sigma0 <= 1. else 0.
    eps = torch.randn_like(x) * s_noise  # In Karras, they used 1.007 for s_noise, but 1. works well/simpler/more principled
    sigma_hat = sigma0 * (gamma + 1)  # Increase the first sigma, by adding noise
    # print(f'    sigma.shape = {sigma_hat.shape}')
    if gamma > 0: 
        x = x + eps * (sigma_hat ** 2 - sigma0 ** 2) ** 0.5    

    # Euler step.
    denoised = translate_denoiser(model, x, sigma_hat)  # original had a pre-wrapped model: model(x, sigma_hat * s_in)
    d = to_d(x, sigma_hat, denoised)
    dt = sigma1 - sigma_hat

    # Apply another correction
    if order == 1 or sigma1 == 0:
        x = x + d * dt  # Euler method
    elif order == 2:  # Heun's method
        x_2 = x + d * dt
        denoised_2 = translate_denoiser(model, x_2, sigma1)
        d_2 = to_d(x_2, sigma1, denoised_2)
        d_prime = (d + d_2) / 2
        x = x + d_prime * dt
    else:
        assert False, "first and second order only supported"
    return x


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    else: 
        return torch.tensor(x)

