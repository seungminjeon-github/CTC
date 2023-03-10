import torch
import torch.nn as nn
from torch.nn import Sequential as Seq

from .utils import *


class classifier_v003_7_sc(nn.Module):
    def __init__(self, N=192, **kwargs):
        super().__init__()
        self.enc_P = Seq(
            ResidualBlock(3 * N, 3 * N),
            ResidualBlock(3 * N, 3 * N),
            ResidualBlock(3 * N, N),
            ResidualBlock(N, N),
        )

        self.enc_q = Seq(
            ResidualBlock(3 * N, 3 * N),
            ResidualBlock(3 * N, 3 * N),
            ResidualBlock(3 * N, N),
            ResidualBlock(N, N),
        )

        self.enc_y = Seq(
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
        )

        self.enc_entropy_param = Seq(
            ResidualBlock(2 * N, 2 * N),
            ResidualBlock(2 * N, 2 * N),
            ResidualBlock(2 * N, N),
            ResidualBlock(N, N),
        )

        self.enc = Seq(
            ResidualBlock(4 * N, 4 * N),
            ResidualBlock(4 * N, 4 * N),
            ResidualBlock(4 * N, 4 * N),
            ResidualBlock(4 * N, 4 * N),
            ResidualBlock(4 * N, 4 * N),
            ResidualBlock(4 * N, 4 * N),
            conv1x1(4 * N, 4 * N),
            nn.LeakyReLU(inplace=True),
            conv1x1(4 * N, 4 * N),
            nn.LeakyReLU(inplace=True),
            conv1x1(4 * N, 4 * N),
            nn.LeakyReLU(inplace=True),
            conv1x1(4 * N, 4 * N),
        )

        self.softmax = nn.Softmax(dim=-1)
        self.sharpen_act = nn.Sigmoid()

    def forward(self, y_hat, P, q, scales_hat, means_hat, **kwargs):
        B, C, H, W, mode = P.shape

        identity = P

        P = torch.cat([P[:, :, :, :, i] for i in range(mode)], dim=1)
        q = torch.cat([q[:, :, :, :, i] for i in range(mode)], dim=1)

        P = self.enc_P(P)
        q = self.enc_q(q)
        y_hat = self.enc_y(y_hat)
        entropy_param = self.enc_entropy_param(torch.cat([scales_hat, means_hat], dim=1))

        out = torch.cat([P, y_hat, q, entropy_param], dim=1)

        out = self.enc(out)
        out0, out1, out2, sc = out.chunk(4, 1)
        out = torch.stack([out0, out1, out2], dim=-1)

        sc = self.sharpen_act(sc.unsqueeze(-1)) * 9.5 + 0.5

        out += identity
        out = out * sc
        out = self.softmax(out)

        return out