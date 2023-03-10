import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "ResidualBlock",
    "conv1x1",
    "post_processing_crr",
]

def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out


def post_processing_crr(i, device, p_len, l_ele, maxL,
                        scales_hat, means_hat,
                        pmfs_list, xpmfs_list, pmf_center_list,
                        pmfs_norm,
                        ):
    mode = 3
    prob_tensor = torch.zeros(list(scales_hat.shape) + [mode]).to(device)
    prob_tensor[:, :, :, :, 1] = 1

    Qp = list(map(lambda xp, p, c: xp.view(p.size(0), mode, p_len).sum(-1) / p.view(p.size(0), mode, p_len).sum(-1) - c, xpmfs_list[:i + 1], pmfs_list[:i + 1], pmf_center_list))
    q = torch.zeros_like(prob_tensor)

    for j in range(i + 1):
        prob_tensor[l_ele == maxL - j] = pmfs_norm[j]
        q[l_ele == maxL - j] = Qp[j]

    for j in range(i + 1, maxL):
        viewshape = pmfs_list[j].size(0), mode, pmfs_list[j].size(-1) // mode
        q[l_ele == maxL - j] = xpmfs_list[j].view(viewshape).sum(-1) / pmfs_list[j].view(viewshape).sum(-1) - pmf_center_list[j]

    q += means_hat.unsqueeze(-1)

    return prob_tensor, q
