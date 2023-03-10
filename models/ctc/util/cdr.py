import torch
import torch.nn as nn
from torch.nn import Sequential as Seq

from .utils import *


class CDR_v000(nn.Module):
    def __init__(self, N):
        super().__init__()

        self.enc_latent = Seq(
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
        )

        self.enc_entropy_params = Seq(
            ResidualBlock(2 * N, 2 * N),
            ResidualBlock(2 * N, 2 * N),
            ResidualBlock(2 * N, 2 * N),
            ResidualBlock(2 * N, N),
        )

        self.enc = Seq(
            ResidualBlock(2 * N, 2 * N),
            ResidualBlock(2 * N, 2 * N),
            ResidualBlock(2 * N, 2 * N),
            ResidualBlock(2 * N, N),
        )

    def forward(self, x, entropy_params, I_min, I_max):
        identity = x

        f_latent = self.enc_latent(x)
        f_ent = self.enc_entropy_params(entropy_params)

        ret = self.enc(torch.cat([f_latent, f_ent], dim=1))

        ret += identity
        ret = torch.clamp(ret, min=I_min, max=I_max)

        return ret