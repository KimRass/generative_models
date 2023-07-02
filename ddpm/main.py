# References:
    # https://nn.labml.ai/diffusion/ddpm/index.html

import torch
import torch.nn as nn
import torch.nn.functional as F


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)


# def __init__(n_steps):
n_steps = 100
# self.n_steps = n_steps # $T$
beta = torch.linspace(0.0001, 0.02, n_steps) # $\beta_{t}$
alpha = 1 - beta # $\alpha_{t} = 1 - \beta_{t}$
alpha_bar = torch.cumprod(alpha, dim=0) # $\bar{\alpha_{t}} = \prod^{t}_{s=1}{\alpha_{s}}$
sigma_square = beta


# def q_xt_x0(x0, t):
x0 = torch.randn((4, 3, 200, 300))
t = torch.tensor([11])
mu_theta = torch.gather(alpha_bar, dim=0, index=t) ** 0.5 * x0 # $\sqrt{\bar{\alpha_{t}}}x_{0}$
sigma_theta = 1 - torch.gather(alpha_bar, dim=0, index=t) # $(1 - \bar{\alpha_{t}})\mathbf{I}$
