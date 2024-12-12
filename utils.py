# Utilities, including visualization and integration routine
import torch as t
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm


def logistic_integrate(npoints, loc, scale, clip=4., device='cpu', deterministic=False):
    """Return sample point and weights for integration, using
    a truncated logistic distribution as the base, and importance weights.
    """
    loc, scale, clip = t.tensor(loc, device=device), t.tensor(scale, device=device), t.tensor(clip, device=device)
    # IID samples from uniform, use inverse CDF to transform to target distribution
    if deterministic:
        t.manual_seed(0)
    ps = t.rand(npoints, device=device)
    ps = t.sigmoid(-clip) + (t.sigmoid(clip) - t.sigmoid(-clip)) * ps  # Scale quantiles to clip
    logsnr = loc + scale * t.logit(ps)  # Using quantile function for logistic distribution

    # importance weights
    weights = scale * t.tanh(clip / 2) / (t.sigmoid((logsnr - loc)/scale) * t.sigmoid(-(logsnr - loc)/scale))
    return logsnr, weights


def logsnr_uniform_selector(npoints, low, high, device='cpu'):
    """ Return logsnr chosen from a uniform distribution by the following"""
    ps = t.rand(npoints, device=device) 

    range_checker = (ps >= 0.) & (ps <= 1.)
    if not range_checker.all():
        print(f'uniform alpha (logsnr) NOT in [0, 1]')
    
    logsnr = low + (high - low) * ps
    return logsnr


#### DDPM timestep schedule to logsnr ####
def get_ddpm_schedule(npoints, start_schedule=0.0001, end_schedule=0.02, num_train_timesteps=1000, device='cpu'):
    """ sample timesteps ~ U[0,T], then translate to logsnr. Get ddpm timesteps, according to ddpm way of sampling timesteps. \bar{alpha}_t = sigmoid(logsnr) """
    # Uniformly randomly pick npoints timestep
    timesteps = t.randint(0, num_train_timesteps, (npoints,)).long().to(device)
    
    # Calculate alphas_cumprod
    betas = t.linspace(start_schedule, end_schedule, num_train_timesteps)
    alphas = 1 - betas
    alphas_cumprod = t.cumprod(alphas, axis=0)

    # pick the \bar{alpha}_t from alphas_cumprod, according to the indices stored in 'timestep'
    selected_alphas_cumprod = alphas_cumprod.gather(-1, timesteps.cpu()).to(timesteps.device)
    logsnrs = t.log(selected_alphas_cumprod / (1. - selected_alphas_cumprod)) # logsnr = logit(selected_alphas_cumprod)
    return logsnrs


def logsnr2t(logsnr):
    num_diffusion_steps = 10000 # improve the timestep precision, more t here  # [Yunshu: this is just make logsnr->timestep more precise, no more timesteps here]
    alphas_cumprod = t.sigmoid(logsnr)
    scale = 1000 / num_diffusion_steps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = np.linspace(beta_start, beta_end, num_diffusion_steps, dtype=np.float64) # ddpm has linear schedule for drawing betas
    alphas = 1.0 - betas
    alphabarGT = t.tensor(np.cumprod(alphas, axis=0), device=alphas_cumprod.device)
    timestep = t.argmin(abs(alphabarGT-alphas_cumprod.unsqueeze(1)), dim=1) * scale # bring the timestep back to the original range [0,T]
    return timestep



def score(dist, x):
    """Compute score function (gradient of log p(x)) for a Pytorch distribution, dist, at points, x."""
    x = t.autograd.Variable(x, requires_grad=True)
    return t.autograd.grad(dist.log_prob(x).sum(), [x])[0]

def gauss_contour(cov, r, npoints=100):
    """Output a contour of a 2D Gaussian distribution."""
    theta = np.linspace(0, 2 * np.pi, npoints)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    xy = np.stack((x, y), axis=1)
    mat = sqrtm(cov)
    return np.einsum('ij,kj->ki', mat, xy)

#### EVALUATION METRICS ####
# functions for MMD test
def median_heuristic(x):
    """Gretton et al heuristic, use the median of all distances as the bandwidth"""
    x = x.view(x.shape[0], x.shape[1], 1)
    return t.square(x - x.transpose(0, 2)).sum(axis=1).median()


def kernel(x, y, bw):
    """Gaussian kernel with adjustable bandwidth."""
    if not len(x.shape) == len(y.shape) == 2:
        raise ValueError('Both inputs should be matrices.')
    if x.shape[1] != y.shape[1]:
        raise ValueError('The number of features should be the same.')

    x = x.view(x.shape[0], x.shape[1], 1)
    dist = t.square(x - y.T).sum(axis=1)
    return t.exp(- 0.5 * dist / bw) / t.sqrt(2. * np.pi * bw)


def mmd_ci_kernel(kxx, kxy, kyy):
    """Maximum mean discrepancy from the kernels. It's faster not to re-compute some kernels each time for my plots.
    This is a special version where we also estimate the bootstrap conf. interval of the estimator for each x_i
    """
    iuy = t.triu_indices(len(kyy), len(kyy), 1)
    cost = t.mean(kyy[iuy[0], iuy[1]])
    cost = cost + (t.sum(kxx, axis=1) - kxx.diag()) / (len(kxx) - 1)
    cost = cost - 2 * t.mean(kxy, axis=1)
    return cost.mean() 




#### VISUALIZATION UTILITIES ####

def plot_mse(logsnrs, mses, mmse_g):
    fig, ax = plt.subplots(1, 1)
    ax.plot(logsnrs, mses, label="MSE")
    ax.plot(logsnrs, mmse_g, label='MMSE Gaussian')
    ax.set_ylabel('$E[(\epsilon - \hat \epsilon)^2]$')
    ax.set_xlabel('log SNR ($\\alpha$)')
    ax.legend()
    return fig

def plot_cdl_alphas(logsnrs):
    fig, ax = plt.subplots(1, 1)
    ax.hist(logsnrs, bins=50)
    ax.set_ylabel('frequency')
    ax.set_xlabel('log SNR ($\\alpha$) values')
    ax.legend()
    return fig

def plot_mse_alphas(logsnrs, mse_alpha, mse_alpha1):
    fig, ax = plt.subplots(1, 1)
    ax.plot(logsnrs, mse_alpha, label="MSE$_{\\alpha}$")
    ax.plot(logsnrs, mse_alpha1, label="MSE$_{\\alpha+\\Delta}$")
    ax.set_ylabel('$E_{p(x)}[(\epsilon - \hat \epsilon)^2]$')
    ax.set_xlabel('log SNR ($\\alpha$)')
    ax.legend()
    return fig


def plot_density(x, logp1, logp2, labels):
    fig, axs = plt.subplots(1, len(labels), sharex=True, sharey=False, figsize=(6 * len(labels), 6))
    for i, ax in enumerate(axs):
        ax.set_title(labels[i])
        ax.plot(x, logp1[i], label='Diff. estimate')
        ax.plot(x, logp2[i], label='True')
        ax.set_ylabel('$\log p_{\\alpha}(\\vec x + \eta \cdot \\vec v)$')
        ax.set_xlabel('$\eta$')
        ax.legend()
    return fig


def vector_field_xgrids(x_grids, grads, labels, scale=None):
    """Given a list of x_grids, plot vector field"""
    fig, axs = plt.subplots(1, len(grads), sharex=True, sharey=True, figsize=(6 * len(grads), 6))
    for i, ax in enumerate(axs):
        x_grid = x_grids[i]
        xv, yv = x_grid[:,0], x_grid[:,1]
        if i > 0:
            scale = q.scale
            print("scale", scale)
        ax.set_title(labels[i])
        q = ax.quiver(xv, yv, grads[i][:,0], grads[i][:,1], scale=scale, scale_units='inches')
        q._init()
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
    return fig


def vector_field(x_grid, grads, labels, scale=None):
    """Plot vector field for some function, grad, on Axis, ax."""
    xv, yv = x_grid[:,0], x_grid[:,1]
    fig, axs = plt.subplots(1, len(grads), sharex=True, sharey=True, figsize=(6 * len(grads), 6))
    for i, ax in enumerate(axs):
        if i > 0:
            scale = q.scale
            print("scale", scale)
        ax.set_title(labels[i])
        # grads = grad_fs[i](x_grid)
        q = ax.quiver(xv, yv, grads[i][:,0], grads[i][:,1], scale=scale, scale_units='inches')
        q._init()
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
    return fig

def vector_field2(x_grid, grad1, grad2, labels1, labels2, scale=None):
    """Plot vector fields on two rows, with various values for columns."""
    fig, axs = plt.subplots(2, len(grad2), sharex=True, sharey=True, figsize=(6 * len(grad2), 6*2))
    for i in range(len(grad1)):
        ax = axs[0, i]
        if i > 0:
            scale = q.scale
        ax.set_title(labels1[i])
        q = ax.quiver(x_grid[:,0], x_grid[:,1], grad1[i][:,0] - grad2[i][:,0], grad1[i][:,1] - grad2[i][:,1], scale=scale, scale_units='inches')
        q._init()
        print('scale', q.scale)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')

    for i in range(len(grad2)):
        ax = axs[1, i]
        scale = None  # auto-scale each time
        ax.set_title(labels2[i])
        q = ax.quiver(x_grid[:,0], x_grid[:,1], grad2[i][:,0], grad2[i][:,1], scale=scale, scale_units='inches')
        q._init()
        print('scale row 2', q.scale)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')

    fig.delaxes(axs[0, -1])
    return fig

def vector_field3(x_grid, grad1, grad2, labels1, labels2, scale=None):
    """Plot vector fields on two rows, with various values for columns."""
    fig, axs = plt.subplots(1, len(grad1), sharex=True, sharey=True, figsize=(6 * len(grad1), 6))
    for i in range(len(grad1)):
        ax = axs[i]
        ax.set_title(labels1[i])
        q = ax.quiver(x_grid[:,0], x_grid[:,1], grad1[i][:,0], grad1[i][:,1], scale=scale, scale_units='inches', label='Est.')
        q._init()
        ax.quiver(x_grid[:, 0], x_grid[:, 1], grad2[i][:, 0], grad2[i][:, 1], scale=scale, scale_units='inches', label='True', color='r', alpha=0.5)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.legend()

    return fig


# yunshu: add the following plot functions for single logsnr vector plot of gradient
def vector_field2_single(x_grid, grad1, grad2, labels1, labels2, scale=None):
    """Plot vector fields on two rows, with various values for columns."""
    fig, axs = plt.subplots(2, len(grad2), sharex=True, sharey=True, figsize=(6 * len(grad2), 6*2))
    
    ax1 = axs[0]
    ax1.set_title(labels1[0])
    q = ax1.quiver(x_grid[:,0], x_grid[:,1], grad1[0][:,0] - grad2[0][:,0], grad1[0][:,1] - grad2[0][:,1], scale=scale, scale_units='inches')
    q._init()
    print('scale', q.scale)
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')

    ax2 = axs[1]
    scale = q.scale
    ax2.set_title(labels2[0])
    q = ax2.quiver(x_grid[:,0], x_grid[:,1], grad2[0][:,0], grad2[0][:,1], scale=scale, scale_units='inches')
    q._init()
    print('scale row 2', q.scale)
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')

    # yunshu: not understand what the following code is doing??????
    # fig.delaxes(axs[0]) # yunshu: this will remove the first subplot (specified by axs[0]) 
    return fig


def vector_field3_single(x_grid, grad1, grad2, labels, scale=None):
    """Plot vector fields on two rows, with various values for columns."""
    fig, axs = plt.subplots(1, len(grad1), sharex=True, sharey=True, figsize=(6 * len(grad1), 6))

    axs.set_title(labels[0])
    q = axs.quiver(x_grid[:,0], x_grid[:,1], grad1[0][:,0], grad1[0][:,1], scale=scale, scale_units='inches', label='Est.')
    q._init()
    axs.quiver(x_grid[:, 0], x_grid[:, 1], grad2[0][:, 0], grad2[0][:, 1], scale=q.scale, scale_units='inches', label='True', color='r', alpha=0.5)
    axs.set_xlabel('$x_1$')
    axs.set_ylabel('$x_2$')
    axs.legend()

    return fig

