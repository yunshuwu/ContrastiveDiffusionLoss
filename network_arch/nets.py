import math
import torch as t
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers, activation=nn.ReLU, dropout=0.):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.dropout = dropout

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim + 1, hidden_dim))  # Concatenate logsnr - should use embedding in high-d
        self.layers.append(activation())
        if dropout > 0:
            self.layers.append(nn.Dropout(dropout))
        for _ in range(n_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(activation())
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(hidden_dim, in_dim))

    def forward(self, x, logsnr):
        # print(f'    x.shape = {x.shape}, logsnr.unsqueeze(1) = {logsnr.unsqueeze(1).shape}')
        x = t.concat((x, logsnr.unsqueeze(1)), dim=1)  # concatenate logsnr
        for layer in self.layers:
            x = layer(x)
        return x


"""
Yunshu: I copied from some repo, a denoiser architecture with positional encoding
"""
class PositionalEncoding(nn.Module):
    """The classic positional encoding from the original Attention papers"""

    def __init__(
        self,
        d_model: int = 128,
        maxlen: int = 1024,
        min_freq: float = 1e-4,
        # device: str = "cpu",
        dtype=t.float32,
    ):
        """
        Args:
            d_model (int, optional): embedding dimension of each token. Defaults to 128.
            maxlen (int, optional): maximum sequence length. Defaults to 1024.
            min_freq (float, optional): use the magic 1/10,000 value! Defaults to 1e-4.
            device (str, optional): accelerator or nah. Defaults to "cpu".
            dtype (_type_, optional): torch dtype. Defaults to torch.float32.
        """
        super().__init__()
        pos_enc = self._get_pos_enc(d_model=d_model, maxlen=maxlen, min_freq=min_freq)
        self.register_buffer(
            # "pos_enc", t.tensor(pos_enc, dtype=dtype, device=device)
            "pos_enc", t.tensor(pos_enc, dtype=dtype)
        )

    def _get_pos_enc(self, d_model: int, maxlen: int, min_freq: float):
        position = np.arange(maxlen)
        freqs = min_freq ** (2 * (np.arange(d_model) // 2) / d_model)
        pos_enc = position[:, None] * freqs[None]
        pos_enc[:, ::2] = np.cos(pos_enc[:, ::2])
        pos_enc[:, 1::2] = np.sin(pos_enc[:, 1::2])
        return pos_enc

    def forward(self, x):
        return self.pos_enc[x]


class DiscreteTimeResidualBlock(nn.Module):
    """Generic block to learn a nonlinear function f(x, t), where
    t is discrete and x is continuous."""

    def __init__(self, d_model: int, maxlen: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.emb = PositionalEncoding(d_model=d_model, maxlen=maxlen)
        self.lin1 = nn.Linear(d_model, d_model)
        self.lin2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, x, t):
        return self.norm(x + self.lin2(self.act(self.lin1(x + self.emb(t)))))


class BasicDiscreteTimeModel(nn.Module):
    def __init__(self, d_model: int = 128, n_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.lin_in = nn.Linear(2, d_model)
        self.lin_out = nn.Linear(d_model, 2)
        self.blocks = nn.ParameterList(
            [DiscreteTimeResidualBlock(d_model=d_model) for _ in range(n_layers)]
        )

    def forward(self, x, t):
        x = self.lin_in(x)
        for block in self.blocks:
            x = block(x, t)
        return self.lin_out(x)


class GaussTrueMMSE(nn.Module):
    """For testing, we construct several ground truth quantities for a Gaussian, N(0, Sigma).
       Calling the (forward) module computes the optimal eps_hat(z, snr),
       for the optimal denoising estimator x_hat, which is written as,
       x_hat(z, snr) = (z - eps_hat)/sqrt(snr)   # to match Diffusion literature conventions
    """
    def __init__(self, cov, device):
        super().__init__()
        self.cov = t.from_numpy(cov).type(t.FloatTensor)
        self.cov = self.cov.to(device)
        self.d = len(self.cov)
        self.S, self.U = t.linalg.eigh(self.cov)
        self.prec = t.mm(t.mm(self.U, t.diag(1. / self.S)), self.U.t())  # Precision is inverse covariance
        self.logZ = 0.5 * self.d * math.log(2 * math.pi) + 0.5 * t.log(self.S).sum()  # Log normalization
        self.dummy = t.nn.Parameter(t.randn(2))
        self.register_parameter(name='dummy', param=self.dummy)  # Dummy parameter so it runs fit method

    def entropy(self):
        """Differential entropy for a Gaussian"""
        return self.logZ + self.d / 2

    def mmse(self, snr):
        """Minimum mean square error at a given SNR level."""
        return (1. / (1. / self.S + snr.view((-1, 1)))).sum(axis=1)

    def nll(self, x):
        """-log p(x)"""
        return 0.5 * t.mm(t.mm(x, self.prec), x.t()) + self.logZ

    def true_grad(self, x):
        """- \nabla_x log p(x)"""
        return t.matmul(x, self.prec)

    def true_grad_mmse(self, x, snr):
        """$- \nabla_x 1/2 mmse(x, snr)$"""
        a = 1. / t.square(snr * self.S + 1)
        M = t.mm(t.mm(self.U, t.diag(a)), self.U.t())
        return t.matmul(x, M)

    def forward(self, z, logsnr):
        """The DiffusionModel expects to get:
        eps_hat(z, snr), where z = sqrt(snr/(1+snr) x + sqrt(1/(1+snr) eps,
        and x_hat = sqrt((1+snr)/snr) * z - eps_hat / sqrt(snr)
        For Gaussians, we derive the optimal estimator:
        x_hat^* = sqrt(snr/(1+snr)) (snr/(1+snr) I + Sigma^-1/(1+snr))^-1 z
        The matrix inverses we handle with the precomputed SVD of Sigma (covariance for x).
        """
        snr = t.exp(logsnr)
        assert len(z) == len(snr)
        snr = snr.view((-1, 1))
        xhat = t.mm(z, self.U)
        xhat = ((1. + snr) / (snr + 1. / self.S)) * xhat
        xhat = t.mm(xhat, self.U.t())
        xhat = t.sqrt(snr / (1. + snr)) * xhat
        # Now return eps_hat estimator
        return t.sqrt(1+snr) * z - t.sqrt(snr) * xhat + 0. * self.dummy.sum()  # Have to include dummy param in forward
