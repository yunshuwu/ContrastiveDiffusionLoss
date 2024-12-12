"""
Main information theoretic diffusion model class
"""

import math
import numpy as np
import torch
import pytorch_lightning as pl

import utils

class ITD_DiffusionModel_StandardNormalGauss(pl.LightningModule):
    def __init__(self, denoiser, x_shape=(2,),
                 learning_rate=0.001, logsnr_loc=2., logsnr_scale=3.):
        super().__init__()
        self.save_hyperparameters(ignore=["denoiser"]) # save full arg to self.hparams
        self.model = denoiser
        self.d = np.prod(x_shape)
        self.h_g = 0.5 * self.d * math.log(2 * math.pi * math.e)  # Differential entropy for N(0,I)
        self.left = (-1,) + (1,) * (len(x_shape))  # View for left multiplying a batch of samples
        self.automatic_optimization = False  # Pytorch Lightning flag
        self.shape = x_shape

    
    def forward(self, x, timestep):
        return self.model(x, timestep)

    def score(self, x, alpha):
        """\nabla_z \log p_\alpha(z), converges to data dist. score in large SNR limit."""
        return -self.model(x, alpha) / torch.sqrt(torch.sigmoid(-alpha.view(self.left)))

    def training_step(self, batch, batch_idx):
        self.optimizers().zero_grad()

        # ITD loss
        itd_loss = self.nll(batch)
        self.manual_backward(itd_loss)

        self.optimizers().step()

        # logging
        self.log("train_itd_loss", itd_loss)
        return itd_loss

    def validation_step(self, batch, batch_idx):
        itd_loss = self.nll(batch)
        self.log("val_itd_loss", itd_loss)

        if batch_idx == 0: # plot and log MSE curve
            mses = []
            loc, s = self.hparams.logsnr_loc, self.hparams.logsnr_scale
            x = batch[0]
            logsnrs = torch.linspace(loc - 3 * s, loc + 3 * s, 100, device=self.device)
            mmse_g = self.d * torch.sigmoid(logsnrs)
            # mmse_g = self.mmse_gauss(logsnrs)
            for logsnr in logsnrs:
                mses.append(self.mse(x, torch.ones(len(x), device=self.device) * logsnr).mean().cpu())
            tb = self.logger.experiment
            fig = utils.plot_mse(logsnrs.cpu(), mses, mmse_g.cpu())
            tb.add_figure('mses', fig)
        
        return itd_loss
    
    def configure_optimizers(self):
        """Pytorch Lightning optimizer hook."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def noisy_channel(self, x, logsnr):
        """Add Gaussian noise to x, return "z" and epsilon."""
        logsnr = logsnr.view(self.left)  # View for left multiplying
        eps = torch.randn((len(logsnr),) + self.hparams.x_shape, device=self.device)
        return torch.sqrt(torch.sigmoid(logsnr)) * x + torch.sqrt(torch.sigmoid(-logsnr)) * eps, eps

    def mse(self, x, logsnr, alpha=None):
        """MSE for recovering epsilon from noisy channel, for given log SNR values."""
        z, eps = self.noisy_channel(x, logsnr)
        if alpha is not None:  # Used for log p_alpha(x) = p(x) convolved with Gaussian noise
            logsnr_prime = torch.logit(torch.sigmoid(logsnr) / (1. + math.exp(-alpha)))
            scale = torch.sqrt(torch.sigmoid(-logsnr) / torch.sigmoid(-logsnr_prime)).view(self.left)
            timestep_prime = utils.logsnr2t(logsnr_prime)
            eps_hat = scale * self(z, timestep_prime) # eps_hat = scale * self(z, logsnr_prime)
        else: 
            timestep = utils.logsnr2t(logsnr)
            eps_hat = self(z, timestep) # eps_hat = self(z, logsnr)

        error = (eps - eps_hat).flatten(start_dim=1)
        return torch.einsum('ij,ij->i', error, error)  # MSE per sample

    def nll(self, batch, alpha=None):
        """Estimate of negative log likelihood for a batch, log p_alpha(z),
        for z= sqrt-sigmoid(alpha) x + sqrt-sigmoid(-alpha) epsilon
        alpha=None corresponds to infinity, which is log p(x), the data distribution. """
        ###---- use logsnr training schedule ----### 
        # x = batch[0]
        # logsnr, weights = utils.logistic_integrate(len(x), self.hparams.logsnr_loc, self.hparams.logsnr_scale, device=self.device)
        # # print(f'    logsnr.shape = {logsnr.shape}')
        # mses = self.mse(x, logsnr, alpha=alpha)
        # mmse_gap = mses - self.d * torch.sigmoid(logsnr)  # MSE gap compared to using optimal denoiser for N(0,I)
        # return self.h_g + 0.5 * (weights * mmse_gap).mean()  # Interpretable as differential entropy (nats)

        ###---- use ddpm training schedule ----### 
        x = batch[0]
        logsnr = utils.get_ddpm_schedule(len(x), device=self.device)
        mses = self.mse(x, logsnr, alpha=alpha)
        mmse_gap = mses - self.d * torch.sigmoid(logsnr)  # MSE gap compared to using optimal denoiser for N(0,I)
        # mmse_gap = mses - self.mmse_gauss(logsnr)
        return self.h_g + 0.5 * mmse_gap.mean()  # Interpretable as differential entropy (nats)

    def nll_x(self, x, npoints=100, alpha=None):
        """-log p(x) for a single sample, x"""
        return self.nll([x.unsqueeze(0).expand((npoints,) + self.hparams.x_shape)], alpha=alpha)
    


class ITD_DiffusionModel(pl.LightningModule):
    def __init__(self, denoiser, x_shape=(2,),
                 learning_rate=0.001, logsnr_loc=2., logsnr_scale=3.):
        super().__init__()
        self.save_hyperparameters(ignore=["denoiser"]) # save full arg to self.hparams
        self.model = denoiser
        self.d = np.prod(x_shape)
        self.h_g = 0.5 * self.d * math.log(2 * math.pi * math.e)  # Differential entropy for N(0,I)
        self.left = (-1,) + (1,) * (len(x_shape))  # View for left multiplying a batch of samples
        self.automatic_optimization = False  # Pytorch Lightning flag
        self.shape = x_shape

    
    def forward(self, x, timestep):
        return self.model(x, timestep)

    def score(self, x, alpha):
        """\nabla_z \log p_\alpha(z), converges to data dist. score in large SNR limit."""
        return -self.model(x, alpha) / torch.sqrt(torch.sigmoid(-alpha.view(self.left)))

    def training_step(self, batch, batch_idx):
        self.optimizers().zero_grad()

        # ITD loss
        itd_loss = self.nll(batch)
        self.manual_backward(itd_loss)

        self.optimizers().step()

        # logging
        self.log("train_itd_loss", itd_loss)
        return itd_loss

    def validation_step(self, batch, batch_idx):
        itd_loss = self.nll(batch)
        self.log("val_itd_loss", itd_loss)

        if batch_idx == 0: # plot and log MSE curve
            mses = []
            loc, s = self.hparams.logsnr_loc, self.hparams.logsnr_scale
            x = batch[0]
            logsnrs = torch.linspace(loc - 3 * s, loc + 3 * s, 100, device=self.device)
            # mmse_g = self.d * torch.sigmoid(logsnrs)
            mmse_g = self.mmse_gauss(logsnrs)
            for logsnr in logsnrs:
                mses.append(self.mse(x, torch.ones(len(x), device=self.device) * logsnr).mean().cpu())
            tb = self.logger.experiment
            fig = utils.plot_mse(logsnrs.cpu(), mses, mmse_g.cpu())
            tb.add_figure('mses', fig)
        
        return itd_loss
    
    def configure_optimizers(self):
        """Pytorch Lightning optimizer hook."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def noisy_channel(self, x, logsnr):
        """Add Gaussian noise to x, return "z" and epsilon."""
        logsnr = logsnr.view(self.left)  # View for left multiplying
        eps = torch.randn((len(logsnr),) + self.hparams.x_shape, device=self.device)
        return torch.sqrt(torch.sigmoid(logsnr)) * x + torch.sqrt(torch.sigmoid(-logsnr)) * eps, eps

    def mse(self, x, logsnr, alpha=None):
        """MSE for recovering epsilon from noisy channel, for given log SNR values."""
        z, eps = self.noisy_channel(x, logsnr)
        if alpha is not None:  # Used for log p_alpha(x) = p(x) convolved with Gaussian noise
            logsnr_prime = torch.logit(torch.sigmoid(logsnr) / (1. + math.exp(-alpha)))
            scale = torch.sqrt(torch.sigmoid(-logsnr) / torch.sigmoid(-logsnr_prime)).view(self.left)
            timestep_prime = utils.logsnr2t(logsnr_prime)
            eps_hat = scale * self(z, timestep_prime) # eps_hat = scale * self(z, logsnr_prime)
        else:  
            timestep = utils.logsnr2t(logsnr)
            eps_hat = self(z, timestep) # eps_hat = self(z, logsnr)

        error = (eps - eps_hat).flatten(start_dim=1)
        return torch.einsum('ij,ij->i', error, error)  # MSE per sample

    def nll(self, batch, alpha=None):
        """Estimate of negative log likelihood for a batch, log p_alpha(z),
        for z= sqrt-sigmoid(alpha) x + sqrt-sigmoid(-alpha) epsilon
        alpha=None corresponds to infinity, which is log p(x), the data distribution. """
        ###---- use logsnr training schedule ----### 
        # x = batch[0]
        # logsnr, weights = utils.logistic_integrate(len(x), self.hparams.logsnr_loc, self.hparams.logsnr_scale, device=self.device)
        # # print(f'    logsnr.shape = {logsnr.shape}')
        # mses = self.mse(x, logsnr, alpha=alpha)
        # mmse_gap = mses - self.d * torch.sigmoid(logsnr)  # MSE gap compared to using optimal denoiser for N(0,I)
        # return self.h_g + 0.5 * (weights * mmse_gap).mean()  # Interpretable as differential entropy (nats)

        ###---- use ddpm training schedule ----### 
        x = batch[0]
        logsnr = utils.get_ddpm_schedule(len(x), device=self.device)
        mses = self.mse(x, logsnr, alpha=alpha)
        mmse_gap = mses - self.mmse_gauss(logsnr)
        return self.h_g + 0.5 * mmse_gap.mean()  # Interpretable as differential entropy (nats)

    def nll_x(self, x, npoints=100, alpha=None):
        """-log p(x) for a single sample, x"""
        return self.nll([x.unsqueeze(0).expand((npoints,) + self.hparams.x_shape)], alpha=alpha)
    
    @property
    def gauss_differential_entropy(self):
        """Differential entropy for a N(mu, Sigma), where Sigma matches data, with same dimension as data."""
        return 0.5 * self.d * math.log(2 * math.pi * math.e) + 0.5 * self.log_eigs.sum().item()

    def mmse_gauss(self, logsnr):
        """The analytic MMSE for a Gaussian with the same eigenvalues as the data in a Gaussian noise channel."""
        self.log_eigs = self.log_eigs.to(self.device)
        return torch.sigmoid(logsnr + self.log_eigs.view((-1, 1))).sum(axis=0)  # *logsnr integration, see note
    
    def dataset_info(self, dataloader, diagonal=False, dataset_name="fake cifar"):
        """Get logsnr loc, scale, and dataset stats"""
        for batch in dataloader:
            break
        data = batch[0].to('cpu')
        print(f'Number of samples per batch = {len(data)}')

        self.d = len(data[0].flatten())
        if not diagonal: # not using diagonal approximate
            assert len(data) > self.d, f"Use a batch with more samples {len(data)} than dim {self.d}"

        self.shape = data[0].shape
        self.left = (-1,) + (1,) * (len(self.shape))

        # Get the approximate data mean and variance
        x = data.flatten(start_dim=1).to(torch.float32)
        
        var, self.mu = torch.var_mean(x, 0)
        x = x - self.mu
        if diagonal:
            self.log_eigs = torch.log(var)
            self.U = None # there is no U in diagonal approximation
        else:
            _, eigs, self.U = torch.linalg.svd(x, full_matrices=False)  # U.T diag(eigs^2/(n-1)) U = covariance
            self.log_eigs = 2 * torch.log(eigs) - math.log(len(x) - 1)  # Eigs of covariance are eigs**2/(n-1)  of SVD

        # Used to estimate good range for integration
        print(f'log_eigs = {self.log_eigs}')
        self.loc_logsnr = -self.log_eigs.mean().item()
        if len(self.log_eigs) > 2:
            self.scale_logsnr = torch.sqrt(1 + 3. / math.pi * self.log_eigs.var()).item()
        else: 
            self.scale_logsnr = 3. # Yunshu: try huristic one # torch.sqrt(1 + 3. / math.pi * torch.tensor([0.])).item()
        self.hparams.logsnr_loc, self.hparams.logsnr_scale = self.loc_logsnr, self.scale_logsnr

        if dataset_name.lower() == "cifar":
            self.hparams.logsnr_loc, self.hparams.logsnr_scale = 6.261363983154297, 3.0976245403289795 # For CIFAR10, use the heuristic
        if dataset_name.lower() == "fake cifar":
            self.hparams.logsnr_loc, self.hparams.logsnr_scale = 4., 4. # For fake CIFAR10, use the heuristic

        print(f'loc = {self.hparams.logsnr_loc}')
        print(f'scale = {self.hparams.logsnr_scale}')

        self.h_g = self.gauss_differential_entropy



class ITD_DiffusionModel_loss_test(pl.LightningModule):
    """
    The only difference is that nll() is not actual nll, it only calculates the mses.mean() as the loss, which is the same as torch.nn.functional.mse_loss I think?
    """
    def __init__(self, denoiser, x_shape=(2,),
                 learning_rate=0.001, logsnr_loc=2., logsnr_scale=3.):
        super().__init__()
        self.save_hyperparameters(ignore=["denoiser"]) # save full arg to self.hparams
        self.model = denoiser
        self.d = np.prod(x_shape)
        self.h_g = 0.5 * self.d * math.log(2 * math.pi * math.e)  # Differential entropy for N(0,I)
        self.left = (-1,) + (1,) * (len(x_shape))  # View for left multiplying a batch of samples
        self.automatic_optimization = False  # Pytorch Lightning flag
        self.shape = x_shape

    
    def forward(self, x, timestep):
        return self.model(x, timestep)

    def score(self, x, alpha):
        """\nabla_z \log p_\alpha(z), converges to data dist. score in large SNR limit."""
        return -self.model(x, alpha) / torch.sqrt(torch.sigmoid(-alpha.view(self.left)))

    def training_step(self, batch, batch_idx):
        self.optimizers().zero_grad()

        # ITD loss
        itd_loss = self.nll(batch)
        self.manual_backward(itd_loss)

        self.optimizers().step()

        # logging
        self.log("train_itd_loss", itd_loss)
        return itd_loss

    def validation_step(self, batch, batch_idx):
        itd_loss = self.nll(batch)
        self.log("val_itd_loss", itd_loss)

        if batch_idx == 0: # plot and log MSE curve
            mses = []
            loc, s = self.hparams.logsnr_loc, self.hparams.logsnr_scale
            x = batch[0]
            logsnrs = torch.linspace(loc - 3 * s, loc + 3 * s, 100, device=self.device)
            # mmse_g = self.d * torch.sigmoid(logsnrs)
            mmse_g = self.mmse_gauss(logsnrs)
            for logsnr in logsnrs:
                mses.append(self.mse(x, torch.ones(len(x), device=self.device) * logsnr).mean().cpu())
            tb = self.logger.experiment
            fig = utils.plot_mse(logsnrs.cpu(), mses, mmse_g.cpu())
            tb.add_figure('mses', fig)
        
        return itd_loss
    
    def configure_optimizers(self):
        """Pytorch Lightning optimizer hook."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def noisy_channel(self, x, logsnr):
        """Add Gaussian noise to x, return "z" and epsilon."""
        logsnr = logsnr.view(self.left)  # View for left multiplying
        eps = torch.randn((len(logsnr),) + self.hparams.x_shape, device=self.device)
        return torch.sqrt(torch.sigmoid(logsnr)) * x + torch.sqrt(torch.sigmoid(-logsnr)) * eps, eps

    def mse(self, x, logsnr, alpha=None):
        """MSE for recovering epsilon from noisy channel, for given log SNR values."""
        z, eps = self.noisy_channel(x, logsnr)
        if alpha is not None:  # Used for log p_alpha(x) = p(x) convolved with Gaussian noise
            logsnr_prime = torch.logit(torch.sigmoid(logsnr) / (1. + math.exp(-alpha)))
            scale = torch.sqrt(torch.sigmoid(-logsnr) / torch.sigmoid(-logsnr_prime)).view(self.left)
            timestep_prime = utils.logsnr2t(logsnr_prime)
            eps_hat = scale * self(z, timestep_prime) # eps_hat = scale * self(z, logsnr_prime)
        else: 
            timestep = utils.logsnr2t(logsnr)
            eps_hat = self(z, timestep) # eps_hat = self(z, logsnr)

        error = (eps - eps_hat).flatten(start_dim=1)
        return torch.einsum('ij,ij->i', error, error)  # MSE per sample

    def nll(self, batch, alpha=None):
        """Estimate of negative log likelihood for a batch, log p_alpha(z),
        for z= sqrt-sigmoid(alpha) x + sqrt-sigmoid(-alpha) epsilon
        alpha=None corresponds to infinity, which is log p(x), the data distribution. """

        ###---- use ddpm training schedule ----### 
        x = batch[0]
        logsnr = utils.get_ddpm_schedule(len(x), device=self.device)
        mses = self.mse(x, logsnr, alpha=alpha)
        # mmse_gap = mses - self.mmse_gauss(logsnr) # mmse_gap = mses - self.d * torch.sigmoid(logsnr)  # MSE gap compared to using optimal denoiser for N(0,I)
        return mses.mean() # return self.h_g + 0.5 * mmse_gap.mean()  # Interpretable as differential entropy (nats)

    def nll_x(self, x, npoints=100, alpha=None):
        """-log p(x) for a single sample, x"""
        return self.nll([x.unsqueeze(0).expand((npoints,) + self.hparams.x_shape)], alpha=alpha)
    
    @property
    def gauss_differential_entropy(self):
        """Differential entropy for a N(mu, Sigma), where Sigma matches data, with same dimension as data."""
        return 0.5 * self.d * math.log(2 * math.pi * math.e) + 0.5 * self.log_eigs.sum().item()

    def mmse_gauss(self, logsnr):
        """The analytic MMSE for a Gaussian with the same eigenvalues as the data in a Gaussian noise channel."""
        self.log_eigs = self.log_eigs.to(self.device)
        return torch.sigmoid(logsnr + self.log_eigs.view((-1, 1))).sum(axis=0)  # *logsnr integration, see note
    
    def dataset_info(self, dataloader, diagonal=False, dataset_name="fake cifar"):
        """Get logsnr loc, scale, and dataset stats"""
        for batch in dataloader:
            break
        data = batch[0].to('cpu')
        print(f'Number of samples per batch = {len(data)}')

        self.d = len(data[0].flatten())
        if not diagonal: # not using diagonal approximate
            assert len(data) > self.d, f"Use a batch with more samples {len(data)} than dim {self.d}"

        self.shape = data[0].shape
        self.left = (-1,) + (1,) * (len(self.shape))

        # Get the approximate data mean and variance
        x = data.flatten(start_dim=1).to(torch.float32)
        
        var, self.mu = torch.var_mean(x, 0)
        x = x - self.mu
        if diagonal:
            self.log_eigs = torch.log(var)
            self.U = None # there is no U in diagonal approximation
        else:
            _, eigs, self.U = torch.linalg.svd(x, full_matrices=False)  # U.T diag(eigs^2/(n-1)) U = covariance
            self.log_eigs = 2 * torch.log(eigs) - math.log(len(x) - 1)  # Eigs of covariance are eigs**2/(n-1)  of SVD

        # Used to estimate good range for integration
        print(f'log_eigs = {self.log_eigs}')
        self.loc_logsnr = -self.log_eigs.mean().item()
        if len(self.log_eigs) > 2:
            self.scale_logsnr = torch.sqrt(1 + 3. / math.pi * self.log_eigs.var()).item()
        else: 
            self.scale_logsnr = 3. # Yunshu: try huristic one # torch.sqrt(1 + 3. / math.pi * torch.tensor([0.])).item()
        self.hparams.logsnr_loc, self.hparams.logsnr_scale = self.loc_logsnr, self.scale_logsnr

        if dataset_name.lower() == "cifar":
            self.hparams.logsnr_loc, self.hparams.logsnr_scale = 6.261363983154297, 3.0976245403289795 # For CIFAR10, use the heuristic
        if dataset_name.lower() == "fake cifar":
            self.hparams.logsnr_loc, self.hparams.logsnr_scale = 4., 4. # For fake CIFAR10, use the heuristic

        print(f'loc = {self.hparams.logsnr_loc}')
        print(f'scale = {self.hparams.logsnr_scale}')

        self.h_g = self.gauss_differential_entropy



class CDL_DiffusionModel_StandardNormalGauss(pl.LightningModule):
    def __init__(self, denoiser, x_shape=(2,), 
                 std_loss_weight=1., cdl_loss_weight=1., 
                 loss_name='cdl1',
                 alpha_dist='logistic',
                 learning_rate=0.001, 
                 clip=3., logsnr_loc=2., logsnr_scale=3.):
        super().__init__()
        self.save_hyperparameters(ignore=["denoiser"]) # Save full argument dict to self.hparams
        self.model = denoiser
        self.shape = x_shape
        self.d = np.prod(x_shape) # Total data dimensionality
        self.h_g = 0.5 * self.d * math.log(2 * math.pi * math.e) # Differential entropy for N(0,I)
        self.left = (-1,) + (1,) * (len(x_shape)) # View for left multiplying a batch of samples
        self.automatic_optimization = False
        self.clip = clip
        self.loss_name = loss_name
        self.alpha_dist = alpha_dist
        self.flag = 1 # Only eval on self.flag number of data point per batch on CDL loss

    def forward(self, x, timestep):
        return self.model(x, timestep)

    def score(self, x, alpha):
        """\nabla_z \log p_\alpha(z), converges to data dist score in large SNR limit, but score isn't defined when SNR->inf """
        return -self.model(x, alpha) / torch.sqrt(torch.sigmoid(-alpha.view(self.left)))

    def training_step(self, batch, batch_idx):
        self.optimizers().zero_grad()

        if self.hparams.std_loss_weight > 0.:
            std_loss = self.nll(batch)
            self.manual_backward(std_loss * self.hparams.std_loss_weight)
        else:
            std_loss = 0.

        total_cdl_loss = 0. 
        if self.hparams.cdl_loss_weight > 0.:
            if self.loss_name.lower() == 'cdl1':
                for i, xi in enumerate(batch[0]):  # Requires many evaluations per sample, so do one sample at a time
                    cdl_loss = self.contrastive_loss(xi, len(batch[0]))
                    cdl_loss = cdl_loss / len(batch[0]) 
                    self.manual_backward(cdl_loss * self.hparams.cdl_loss_weight)
                    total_cdl_loss += cdl_loss
                    if i >= self.flag:
                        break # only eval on one sample per batch

        self.optimizers().step()

        loss = self.hparams.std_loss_weight * std_loss + self.hparams.cdl_loss_weight * total_cdl_loss
        self.log("train_std_loss", std_loss)
        self.log("train_cdl_loss", total_cdl_loss)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        std_loss = self.nll(batch)

        cdl_loss = 0.
        if self.hparams.cdl_loss_weight > 0.:
            if self.loss_name.lower() == 'cdl1':
                for i, xi in enumerate(batch[0]):
                    cdl_loss += self.contrastive_loss(xi, len(batch[0])) / len(batch[0])
                    if i >= self.flag:
                        break
        
        loss = self.hparams.std_loss_weight * std_loss + self.hparams.cdl_loss_weight * cdl_loss
        self.log("val_std_loss", std_loss)
        self.log('val_cdl_loss', cdl_loss)
        self.log("val_loss", loss)

        if batch_idx == 0:
            # plot the log p(x) MSE curve
            mses = []
            loc, s = self.hparams.logsnr_loc, self.hparams.logsnr_scale
            x = batch[0]
            logsnrs = torch.linspace(loc - self.clip * s, loc + self.clip * s, 100, device=self.device) # sample 100 points to eval MSE
            # mmse_g = self.mmse_gauss(logsnrs) # mmse of base measurement Gaussian with same mean and cov with data
            mmse_g = self.d * torch.sigmoid(logsnrs)
            for logsnr in logsnrs:
                mses.append(self.mse(x, torch.ones(len(x), device=self.device) * logsnr).mean().cpu()) 
            tb = self.logger.experiment
            fig = utils.plot_mse(logsnrs.cpu(), mses, mmse_g.cpu())
            tb.add_figure('mses', fig)
        
        return loss

    def configure_optimizers(self):
        """Pytorch Lightning optimizer hook."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def noisy_channel(self, x, logsnr):
        """Add Gaussian noise to x, return z and eps"""
        logsnr = logsnr.view(self.left)
        eps = torch.randn((len(logsnr),) + self.hparams.x_shape, device=self.device)
        return torch.sqrt(torch.sigmoid(logsnr)) * x + torch.sqrt(torch.sigmoid(-logsnr)) * eps, eps
        
    def mse(self, x, logsnr, alpha=None):
        """MSE for recovering eps from noisy channel, for given log SNR values. """
        z, eps = self.noisy_channel(x, logsnr)
        if alpha is not None:
            logsnr_prime = torch.logit(torch.sigmoid(logsnr) / (1. + math.exp(-alpha)))
            scale = torch.sqrt(torch.sigmoid(-logsnr) / torch.sigmoid(-logsnr_prime)).view(self.left)
            timestep_prime = utils.logsnr2t(logsnr_prime)
            eps_hat = scale * self(z, timestep_prime) # eps_hat = scale * self(z, logsnr_prime)
        else: 
            timestep = utils.logsnr2t(logsnr)
            eps_hat = self(z, timestep) # eps_hat = self(z, logsnr)
        
        error = (eps - eps_hat).flatten(start_dim=1)
        return torch.einsum('ij,ij->i', error, error) # MSE per sample

    def nll(self, batch, alpha=None, logsnr_loc=None, logsnr_s=None):
        """Estimate the negative log likelihood for a batch, log p_alpha(z), alpha=None corresponds to inf, which is data density log p(x). """
        ###---- use logsnr training schedule ----### 
        # x = batch[0]
        # if logsnr_loc is not None: # sample logsnrs for CDL
        #     logsnrs, weights = utils.logistic_integrate(len(x), logsnr_loc, logsnr_s, clip=self.clip, device=self.device)
        # else:
        #     logsnrs, weights = utils.logistic_integrate(len(x), self.hparams.logsnr_loc, self.hparams.logsnr_scale, clip=self.clip, device=self.device)

        # mses = self.mse(x, logsnrs, alpha=alpha)
        # mmse_g = self.d * torch.sigmoid(logsnrs) # mmse for N(0,I)
        # mmse_gap = mses - mmse_g
        # # mmse_gap = mses - self.mmse_gauss(logsnrs)
        # return self.h_g + 0.5 * (weights * mmse_gap).mean() # interpretable as differential entropy (nats)

        ###---- use ddpm training schedule ----### 
        x = batch[0]
        logsnr = utils.get_ddpm_schedule(len(x), device=self.device)
        mses = self.mse(x, logsnr, alpha=alpha)
        mmse_gap = mses - self.d * torch.sigmoid(logsnr)  # MSE gap compared to using optimal denoiser for N(0,I)
        return self.h_g + 0.5 * mmse_gap.mean()  # Interpretable as differential entropy (nats)


    def nll_x(self, x, npoints=100, alpha=None, logsnr_loc=None, logsnr_s=None):
        """-log p(x) for single sample, x, estimated as expectation of npoints of xs"""
        return self.nll([x.unsqueeze(0).expand((npoints,) + self.hparams.x_shape)], alpha=alpha, logsnr_loc=logsnr_loc, logsnr_s=logsnr_s)

    def contrastive_loss(self, x, batch_size):
        """Contrastive loss for a single sample, x, using "batch_size" points to estimate log likelihood. 
        A batch size of a few hundred seems good in practice to estimate continous density (see ITD about continuous density), 
        but this is computationally infeasible."""

        # pick zeta in the paper (here zeta is named alpha, weird, need to be fixed)
        if self.alpha_dist == "logistic":
            loc, scale, clip = 10.5, 1.5, 3.
            logsnrs = utils.logistic_integrate(1, loc, scale, clip, device=self.device)[0]
        elif self.alpha_dist == "uniform":
            logsnrs = utils.logsnr_uniform_selector(1, 6., 15., device=self.device)
        
        # now for 2D cases, just use self.hparams.logsnr_loc and scale for CDL. Later change them!!
        cdl_loc, cdl_scale = self.hparams.logsnr_loc, self.hparams.logsnr_scale

        if torch.rand(1)[0] < 0.5:
            z = self.noisy_channel(x.unsqueeze(0), logsnrs[0:1])[0].squeeze(0) # z~p_alpha
            y = -1.
        else:
            z = x
            y = 1.

        log_p_alpha = -self.nll_x(z, batch_size, alpha=logsnrs[0], logsnr_loc=cdl_loc, logsnr_s=cdl_scale) # y = -1
        log_px = -self.nll_x(z, batch_size, alpha=None, logsnr_loc=cdl_loc, logsnr_s=cdl_scale) # y = 1

        cross_entropy = torch.nn.functional.softplus(y * (log_p_alpha - log_px))
        return cross_entropy



class CDL_DiffusionModel(pl.LightningModule):
    def __init__(self, denoiser, x_shape=(2,), 
                 std_loss_weight=1., cdl_loss_weight=1., 
                 loss_name='cdl1',
                 alpha_dist='logistic',
                 learning_rate=0.001, 
                 clip=3., logsnr_loc=2., logsnr_scale=3.):
        super().__init__()
        self.save_hyperparameters(ignore=["denoiser"]) # Save full argument dict to self.hparams
        self.model = denoiser
        self.shape = x_shape
        self.loc_logsnr, self.scale_logsnr = None, None
        self.d = np.prod(x_shape) # Total data dimensionality
        self.h_g = 0.5 * self.d * math.log(2 * math.pi * math.e) # Differential entropy for N(0,I)
        self.left = (-1,) + (1,) * (len(x_shape)) # View for left multiplying a batch of samples
        self.automatic_optimization = False
        self.clip = clip
        self.loss_name = loss_name
        self.alpha_dist = alpha_dist
        self.flag = 1 # Only eval on self.flag number of data point per batch on CDL loss

    def forward(self, x, logsnr):
        return self.model(x, logsnr)

    def score(self, x, alpha):
        """\nabla_z \log p_\alpha(z), converges to data dist score in large SNR limit, but score isn't defined when SNR->inf """
        return -self.model(x, alpha) / torch.sqrt(torch.sigmoid(-alpha.view(self.left)))

    def training_step(self, batch, batch_idx):
        self.optimizers().zero_grad()

        if self.hparams.std_loss_weight > 0.:
            std_loss = self.nll(batch)
            self.manual_backward(std_loss * self.hparams.std_loss_weight)
        else:
            std_loss = 0.

        total_cdl_loss = 0. 
        if self.hparams.cdl_loss_weight > 0.:
            if self.loss_name.lower() == 'cdl1':
                for i, xi in enumerate(batch[0]):  # Requires many evaluations per sample, so do one sample at a time
                    cdl_loss = self.contrastive_loss(xi, len(batch[0]))
                    cdl_loss = cdl_loss / len(batch[0]) 
                    self.manual_backward(cdl_loss * self.hparams.cdl_loss_weight)
                    total_cdl_loss += cdl_loss
                    if i >= self.flag:
                        break # only eval on one sample per batch

        self.optimizers().step()

        loss = self.hparams.std_loss_weight * std_loss + self.hparams.cdl_loss_weight * total_cdl_loss
        self.log("train_std_loss", std_loss)
        self.log("train_cdl_loss", total_cdl_loss)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        std_loss = self.nll(batch)

        cdl_loss = 0.
        if self.hparams.cdl_loss_weight > 0.:
            if self.loss_name.lower() == 'cdl1':
                for i, xi in enumerate(batch[0]):
                    cdl_loss += self.contrastive_loss(xi, len(batch[0])) / len(batch[0])
                    if i >= self.flag:
                        break
        
        loss = self.hparams.std_loss_weight * std_loss + self.hparams.cdl_loss_weight * cdl_loss
        self.log("val_std_loss", std_loss)
        self.log('val_cdl_loss', cdl_loss)
        self.log("val_loss", loss)

        if batch_idx == 0:
            # plot the log p(x) MSE curve
            mses = []
            loc, s = self.hparams.logsnr_loc, self.hparams.logsnr_scale
            x = batch[0]
            logsnrs = torch.linspace(loc - self.clip * s, loc + self.clip * s, 100, device=self.device) # sample 100 points to eval MSE
            mmse_g = self.mmse_gauss(logsnrs) # mmse of base measurement Gaussian with same mean and cov with data
            # mmse_g = self.d * torch.sigmoid(logsnrs)
            for logsnr in logsnrs:
                mses.append(self.mse(x, torch.ones(len(x), device=self.device) * logsnr).mean().cpu()) 
            tb = self.logger.experiment
            fig = utils.plot_mse(logsnrs.cpu(), mses, mmse_g.cpu())
            tb.add_figure('mses', fig)
        
        return loss

    def configure_optimizers(self):
        """Pytorch Lightning optimizer hook."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def noisy_channel(self, x, logsnr):
        """Add Gaussian noise to x, return z and eps"""
        logsnr = logsnr.view(self.left)
        eps = torch.randn((len(logsnr),) + self.hparams.x_shape, device=self.device)
        return torch.sqrt(torch.sigmoid(logsnr)) * x + torch.sqrt(torch.sigmoid(-logsnr)) * eps, eps
        
    def mse(self, x, logsnr, alpha=None):
        """MSE for recovering eps from noisy channel, for given log SNR values. """
        z, eps = self.noisy_channel(x, logsnr)
        if alpha is not None:
            logsnr_prime = torch.logit(torch.sigmoid(logsnr) / (1. + math.exp(-alpha))) 
            scale = torch.sqrt(torch.sigmoid(-logsnr) / torch.sigmoid(-logsnr_prime)).view(self.left)
            eps_hat = scale * self(z, logsnr_prime)
            # timestep_prime = utils.logsnr2t(logsnr_prime).long()
            # eps_hat = scale * self(z, timestep_prime) 
        else: 
            eps_hat = self(z, logsnr)
            # timestep = utils.logsnr2t(logsnr).long()
            # eps_hat = self(z, timestep) 
        
        error = (eps - eps_hat).flatten(start_dim=1)
        return torch.einsum('ij,ij->i', error, error) # MSE per sample

    def nll_loss(self, batch, alpha=None):
        """Estimate the negative log likelihood for a batch, log p_alpha(z), alpha=None corresponds to inf, which is data density log p(x). """
        ###---- use logsnr training schedule ----### 
        # x = batch[0]
        # if logsnr_loc is not None: # sample logsnrs for CDL
        #     logsnrs, weights = utils.logistic_integrate(len(x), logsnr_loc, logsnr_s, clip=self.clip, device=self.device)
        # else:
        #     logsnrs, weights = utils.logistic_integrate(len(x), self.hparams.logsnr_loc, self.hparams.logsnr_scale, clip=self.clip, device=self.device)

        # mses = self.mse(x, logsnrs, alpha=alpha)
        # mmse_g = self.d * torch.sigmoid(logsnrs) # mmse for N(0,I)
        # mmse_gap = mses - mmse_g
        # # mmse_gap = mses - self.mmse_gauss(logsnrs)
        # return self.h_g + 0.5 * (weights * mmse_gap).mean() # interpretable as differential entropy (nats)

        ###---- use ddpm training schedule ----### 
        x = batch[0]
        logsnr = utils.get_ddpm_schedule(len(x), device=self.device)
        mses = self.mse(x, logsnr, alpha=alpha)
        # mmse_gap = mses - self.d * torch.sigmoid(logsnr)  # MSE gap compared to using optimal denoiser for N(0,I)
        mmse_gap = mses - self.mmse_gauss(logsnr)
        return self.h_g + 0.5 * mmse_gap.mean()  # Interpretable as differential entropy (nats)

    def nll(self, batch, alpha=None, logsnr_loc=None, logsnr_s=None):
        """Estimate the negative log likelihood for a batch, log p_alpha(z), alpha=None corresponds to inf, which is data density log p(x). """
        x = batch[0]
        if logsnr_loc is not None: # sample logsnrs for CDL
            logsnrs, weights = utils.logistic_integrate(len(x), logsnr_loc, logsnr_s, clip=self.clip, device=self.device)
        else:
            logsnrs, weights = utils.logistic_integrate(len(x), self.hparams.logsnr_loc, self.hparams.logsnr_scale, clip=self.clip, device=self.device)

        mses = self.mse(x, logsnrs, alpha=alpha)
        mmse_gap = mses - self.mmse_gauss(logsnrs)
        return self.h_g + 0.5 * (weights * mmse_gap).mean() # interpretable as differential entropy (nats)


    def nll_x(self, x, npoints=100, alpha=None, logsnr_loc=None, logsnr_s=None):
        """-log p(x) for single sample, x, estimated as expectation of npoints of xs"""
        return self.nll([x.unsqueeze(0).expand((npoints,) + self.hparams.x_shape)], alpha=alpha, logsnr_loc=logsnr_loc, logsnr_s=logsnr_s)

    def contrastive_loss(self, x, batch_size):
        """Contrastive loss for a single sample, x, using "batch_size" points to estimate log likelihood. 
        A batch size of a few hundred seems good in practice to estimate continous density (see ITD about continuous density), 
        but this is computationally infeasible."""

        # pick zeta in the paper (here zeta is named alpha, weird, need to be fixed)
        if self.alpha_dist == "logistic":
            loc, scale, clip = 10.5, 1.5, 3.
            logsnrs = utils.logistic_integrate(1, loc, scale, clip, device=self.device)[0]
        elif self.alpha_dist == "uniform":
            logsnrs = utils.logsnr_uniform_selector(1, 6., 15., device=self.device)
        
        # now for 2D cases, just use self.hparams.logsnr_loc and scale for CDL. Later change them!!
        cdl_loc, cdl_scale = self.hparams.logsnr_loc, self.hparams.logsnr_scale

        if torch.rand(1)[0] < 0.5:
            z = self.noisy_channel(x.unsqueeze(0), logsnrs[0:1])[0].squeeze(0) # z~p_alpha
            y = -1.
        else:
            z = x
            y = 1.

        log_p_alpha = -self.nll_x(z, batch_size, alpha=logsnrs[0], logsnr_loc=cdl_loc, logsnr_s=cdl_scale) # y = -1
        log_px = -self.nll_x(z, batch_size, alpha=None, logsnr_loc=cdl_loc, logsnr_s=cdl_scale) # y = 1

        cross_entropy = torch.nn.functional.softplus(y * (log_p_alpha - log_px))
        return cross_entropy

    @property
    def gauss_differential_entropy(self):
        """Differential entropy for a N(mu, Sigma), where Sigma matches data, with same dimension as data."""
        return 0.5 * self.d * math.log(2 * math.pi * math.e) + 0.5 * self.log_eigs.sum().item()

    def mmse_gauss(self, logsnr):
        """The analytic MMSE for a Gaussian with the same eigenvalues as the data in a Gaussian noise channel."""
        self.log_eigs = self.log_eigs.to(self.device)
        return torch.sigmoid(logsnr + self.log_eigs.view((-1, 1))).sum(axis=0)  # *logsnr integration, see note
    
    def dataset_info(self, dataloader, diagonal=False, dataset_name="fake cifar"):
        """Get logsnr loc, scale, and dataset stats"""
        for batch in dataloader:
            break
        data = batch[0].to('cpu')
        print(f'Number of samples per batch = {len(data)}')

        self.d = len(data[0].flatten())
        if not diagonal: # not using diagonal approximate
            assert len(data) > self.d, f"Use a batch with more samples {len(data)} than dim {self.d}"

        self.shape = data[0].shape
        self.left = (-1,) + (1,) * (len(self.shape))

        # Get the approximate data mean and variance
        x = data.flatten(start_dim=1).to(torch.float32)
        
        var, self.mu = torch.var_mean(x, 0)
        x = x - self.mu
        if diagonal:
            self.log_eigs = torch.log(var)
            self.U = None # there is no U in diagonal approximation
        else:
            _, eigs, self.U = torch.linalg.svd(x, full_matrices=False)  # U.T diag(eigs^2/(n-1)) U = covariance
            self.log_eigs = 2 * torch.log(eigs) - math.log(len(x) - 1)  # Eigs of covariance are eigs**2/(n-1)  of SVD

        # Used to estimate good range for integration
        print(f'log_eigs = {self.log_eigs}')
        self.loc_logsnr = -self.log_eigs.mean().item()
        if len(self.log_eigs) > 2:
            self.scale_logsnr = torch.sqrt(1 + 3. / math.pi * self.log_eigs.var()).item()
        else: 
            self.scale_logsnr = 3. # Yunshu: try huristic one # torch.sqrt(1 + 3. / math.pi * torch.tensor([0.])).item()
        self.hparams.logsnr_loc, self.hparams.logsnr_scale = self.loc_logsnr, self.scale_logsnr

        if dataset_name.lower() == "cifar10":
            self.hparams.logsnr_loc, self.hparams.logsnr_scale = 4, 4 # For CIFAR10, use the heuristic
        if dataset_name.lower() == "fake cifar":
            self.hparams.logsnr_loc, self.hparams.logsnr_scale = 4., 4. # For fake CIFAR10, use the heuristic

        print(f'loc = {self.hparams.logsnr_loc}')
        print(f'scale = {self.hparams.logsnr_scale}')

        self.h_g = self.gauss_differential_entropy


import copy
class CDL_DiffusionModel_EMA(CDL_DiffusionModel):
    def __init__(self, denoiser, x_shape=(2, ), 
                 ema_decay=0.9999, 
                 std_loss_weight=1, cdl_loss_weight=1, 
                 loss_name='cdl1', 
                 alpha_dist='logistic', 
                 learning_rate=0.001, 
                 clip=3, logsnr_loc=2, logsnr_scale=3):
        super().__init__(denoiser, x_shape, std_loss_weight, cdl_loss_weight, loss_name, alpha_dist, learning_rate, clip, logsnr_loc, logsnr_scale)
        # self.save_hyperparameters(ignore=["denoiser"]) # Save full argument dict to self.hparams
        # self.ema_decay = ema_decay
        # self.ema_model = None
        self.ema_decay = ema_decay
        # self.init_ema_model(denoiser, x_shape, std_loss_weight, cdl_loss_weight, loss_name, alpha_dist, learning_rate, clip, logsnr_loc, logsnr_scale)
        self.init_ema_model()
        # NOTE: not to deepcopy in the class
        # self.ema_model = copy.deepcopy(self)
        # for param in self.ema_model.parameters():
        #     param.requires_grad = False

    def init_ema_model(self):
        self.ema_model = copy.deepcopy(self.model)
        for param in self.ema_model.parameters():
            param.requires_grad = False

    # def init_ema_model(self, denoiser, x_shape, std_loss_weight, cdl_loss_weight, loss_name, alpha_dist, learning_rate, clip, logsnr_loc, logsnr_scale):
    #     self.ema_model = CDL_DiffusionModel(denoiser, x_shape, std_loss_weight, cdl_loss_weight, loss_name, alpha_dist, learning_rate, clip, logsnr_loc, logsnr_scale)
    #     for param in self.ema_model.parameters():
    #         param.requires_grad = False

    def on_train_start(self):
        # self.ema_model = copy.deepcopy(self)
        # for param in self.ema_model.parameters():
        #     param.requires_grad = False
        # Ensure this method does not reintroduce any errors; it may not be necessary to deepcopy self anymore
        pass

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self.update_ema_weights()

    def update_ema_weights(self):
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.parameters()):
                ema_param.data = self.ema_decay * ema_param.data + (1 - self.ema_decay) * model_param.data


class DDPM_DiffusionModel(pl.LightningModule):
    def __init__(self, denoiser, x_shape=(2,), 
                 num_timesteps=1000, st_schedule=0.0001, end_schedule=0.02, 
                 learning_rate=0.001):
        """
        Args: denoiser: input noisy data z and timestep, output estimated noise eps_hat
        """
        super().__init__()
        self.save_hyperparameters(ignore=["denoiser"]) # save full arg to self.hparams
        self.automatic_optimization = False
        self.model = denoiser
        self.shape = x_shape
        self.d = np.prod(x_shape) # Total data dimensionality
        self.left = (-1,) + (1,) * (len(x_shape))  # View for left multiplying a batch of samples, broadcasting

        self.num_timesteps = num_timesteps # Total amount of timesteps
        # Calculating the alphas_cumprod, scalar func w.r.t. timestep for mixing data and noise
        self.betas = torch.linspace(st_schedule, end_schedule, num_timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)


    def forward(self, x, timestep):
        # No need to translate timestep to logsnr, since we are training every model from scratch
        return self.model(x, timestep)

    def training_step(self, batch, batch_idx):
        self.optimizers().zero_grad()
        loss = self.ddpm_loss(batch)
        self.manual_backward(loss)
        self.optimizers().step()

        # logging
        self.log("train_ddpm_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.ddpm_loss(batch)
        self.log("val_ddpm_loss", loss)
        return loss

    def configure_optimizers(self):
        """Pytorch Lightning optimizer hook."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def t2logsnr(self, timestep):
        """ \bar{alpha}_t=alphas_cumprod = sigmoid(logsnr) """
        selected_alphas_cumprod = self.get_by_idx(self.alphas_cumprod, timestep)
        logsnr = torch.log(selected_alphas_cumprod / (1. - selected_alphas_cumprod)).to(timestep.device)
        return logsnr.squeeze()

    def get_by_idx(self, values, timestep):
        """Pick val[t] from val[], according to the indices stored in timestep"""
        selected_val = values.gather(-1, timestep.cpu())
        selected_val = selected_val.view(self.left).to(self.device)
        return selected_val

    def noisy_channel(self, x, timestep):
        # print(f'len(x) = {len(x)}')
        eps = torch.randn((len(x),) + self.hparams.x_shape, device=self.device) 
        # print(f'eps.shape = {eps.shape}')

        # calculate mixing ratio
        sqrt_selected_alphas_cumprod = self.get_by_idx(self.alphas_cumprod.sqrt(), timestep) # mixing ratio alpha_bar_t.sqrt()
        sqrt_selected_one_minusalphas_cumprod = self.get_by_idx(torch.sqrt(1. - self.alphas_cumprod), timestep) # mixing ratio (1-alpha_bar_t).sqrt()
        # print(f'sqrt_alpha_t.shape = {sqrt_selected_alphas_cumprod.shape}')
        # print(f'sqrt_one_minus_alpha_t.shape = {sqrt_selected_one_minusalphas_cumprod.shape}')

        # mixture 
        mean = sqrt_selected_alphas_cumprod * x
        var = sqrt_selected_one_minusalphas_cumprod * eps
        z = mean + var # yunshu: write mean and var explicitly, hmmm, for fun lol

        return z, eps

    def noisy_channel_logsnr(self, x, logsnr):
        """Add Gaussian noise to x, return z and eps"""
        logsnr = logsnr.view(self.left)
        eps = torch.randn((len(logsnr),) + self.shape, device=self.device)
        return torch.sqrt(torch.sigmoid(logsnr)) * x + torch.sqrt(torch.sigmoid(-logsnr)) * eps, eps

    def ddpm_loss(self, batch):
        """No idea why I implement this way, natually happens now, need to adjust structure later"""
        x = batch[0] # batch of data and label, data=batch[0] and label=data[1]
        # logsnr = utils.get_ddpm_schedule(len(x), device=self.device)
        timestep = torch.randint(0, self.num_timesteps, (len(x),)).long().to(self.device) # pick random t for the current training batch
        # print(f'    timestep.shape = {timestep.shape}')
        # logsnr = self.t2logsnr(timestep) 
        # print(f'    logsnr.shape = {logsnr.shape}')

        z, eps = self.noisy_channel(x, timestep)
        # z, eps = self.noisy_channel_logsnr(x, logsnr)

        eps_hat = self(z, timestep) # predicted noise
        # eps_hat = self(z, logsnr)
        error = (eps - eps_hat).flatten(start_dim=1)
        return torch.einsum('ij,ij->i', error, error).mean() # return torch.nn.functional.mse_loss(eps, eps_hat)


class DDIM_DiffusionModel(pl.LightningModule):
    def __init__(self, denoiser, x_shape=(2,), 
                 learning_rate=0.001):
        """
        Args: denoiser: input noisy data z and timestep, output estimated noise eps_hat
        """
        super().__init__()
        self.save_hyperparameters(ignore=["denoiser"]) # save full arg to self.hparams
        self.automatic_optimization = False
        self.model = denoiser
        self.shape = x_shape
        self.left = (-1,) + (1,) * (len(x_shape))  # View for left multiplying a batch of samples



class EDM_DiffusionModel(pl.LightningModule):
    def __init__(self, denoiser, x_shape=(2,), 
                 learning_rate=0.001):
        """
        Args: denoiser: input noisy data z and timestep, output estimated noise eps_hat
        """
        super().__init__()
        self.save_hyperparameters(ignore=["denoiser"]) # save full arg to self.hparams
        self.automatic_optimization = False
        self.model = denoiser
        self.shape = x_shape
        self.left = (-1,) + (1,) * (len(x_shape))  # View for left multiplying a batch of samples

