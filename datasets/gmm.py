# Copied from Yang Song's NCSC repo 
# File usage: generate synthetic dataset: mixture of Gaussian
# TODO: seems like log_prob() and score() are having issues now


import torch as t
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal
import torch.autograd as autograd

class GMMDist(object):
    def __init__(self, dim, mix_probs=0.5, means=5):
        # self.mix_probs = t.tensor([0.8, 0.2])
        self.mix_probs = t.tensor([mix_probs, 1.-mix_probs])
        # self.means = torch.stack([5 * torch.ones(dim), -torch.ones(dim) * 5], dim=0)
        # self.mix_probs = torch.tensor([0.1, 0.1, 0.8])
        # self.means = torch.stack([5 * torch.ones(dim), torch.zeros(dim), -torch.ones(dim) * 5], dim=0)
        self.means = t.stack([means * t.ones(dim), -t.ones(dim) * means], dim=0)
        self.sigma = 1

    def sample(self, n, sigma=1, logsnr=None):
        n = n[0]
        mix_idx = t.multinomial(self.mix_probs, n, replacement=True)
        
        if logsnr is not None:
            logsnr = t.tensor(logsnr)
            left_view = (-1, 1)
            mean_ratio = t.sqrt(t.sigmoid(logsnr.view(left_view))) # mean_logsnr = \sqrt{sigmoid(\logsnr)}
            means = mean_ratio * self.means[mix_idx]
        else:
            means = self.means[mix_idx]
        return t.randn_like(means) * sigma + means


    def log_prob(self, samples, logsnr=None, sigma=1):
        """ samples are comming in batch, get the shape of samples to create view for left multiplying batch of samples """
        if logsnr is not None:
            logsnr = t.tensor(logsnr)
            x_shape = samples.shape[1:]
            left_view = (-1,) + (1,) * (len(x_shape))
            mean_ratio = t.sqrt(t.sigmoid(logsnr.view(left_view)))

        logps = []
        for i in range(len(self.mix_probs)):
            if logsnr is not None:
                mean_i = mean_ratio * self.means[i]
                logps.append((-((samples - mean_i) ** 2).sum(dim=-1) / (2 * sigma ** 2) - 0.5 * np.log(
                    2 * np.pi * sigma ** 2)) + self.mix_probs[i].log())
            else:
                logps.append((-((samples - self.means[i]) ** 2).sum(dim=-1) / (2 * sigma ** 2) - 0.5 * np.log(
                    2 * np.pi * sigma ** 2)) + self.mix_probs[i].log())

        logp = t.logsumexp(t.stack(logps, dim=0), dim=0)
        
        return logp

    def score(self, samples, logsnr=None, sigma=1):
        with t.enable_grad():
            samples = samples.detach()
            samples.requires_grad_(True)
            log_probs = self.log_prob(samples, logsnr, sigma).sum()
            return autograd.grad(log_probs, samples)[0]

