# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from pytorch_msssim import ms_ssim

import compressai
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
from compressai.models.google import MeanScaleHyperprior

from ..utils_trit_plane import *
from utils import path2torch, pad, write_uints, read_uints, mkfulldir, listfulldir

opt_pnum = 5


class model_baseline(MeanScaleHyperprior):
    def __init__(self, N=192, **kwargs):
        super().__init__(N=N, M=N, **kwargs)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2), # GDN
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2), # GDN
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2), # GDN
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
            AttentionBlock(N),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.g_s = nn.Sequential(
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2), # IGDN
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2), # IGDN
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2), # IGDN
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(N * 2, N * 2, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N * 2, N * 2, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N * 2, N * 2, 1),
        )

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.conv1.weight"].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        gaussian_params = self.entropy_parameters(params)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)
        gaussian_params = self.entropy_parameters(params)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        gaussian_params = self.entropy_parameters(params)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def repr(self, x, **kwargs):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)
        gaussian_params = self.entropy_parameters(params)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        return y, z, z_strings, means_hat, scales_hat

    def encode_dpict(self, x):
        y, z, z_strings, means_hat, scales_hat = self.repr(x)

        device, maxL, l_ele, Nary_tensor = get_Nary_tensor(y, means_hat, scales_hat)
        pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list = make_pmf_table(scales_hat, device, maxL, l_ele)

        y_strings = [[] for _ in range(maxL)]

        for i in range(maxL):
            encoder = get_ans(type="enc")

            if i < maxL - opt_pnum:
                pmfs_norm = list(
                    map(
                        lambda p, idx:
                        (p * idx).view(p.size(0), mode, p.size(-1) // mode).sum(-1) / (p * idx).view(p.size(0), 1, p.size(-1)).sum(-1), pmfs_list[:i + 1], idx_ts_list[:i + 1]
                    )
                )

                TP_entropy_encoding(
                    i, device, maxL, l_ele, Nary_tensor,
                    pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list,
                    pmfs_norm,
                    encoder, y_strings
                )

            else:
                optim_tensor, pmfs_norm = get_transmission_tensor(
                    i, maxL,
                    pmfs_list, xpmfs_list, x2pmfs_list
                )

                TP_entropy_encoding_scalable(
                    i, device, maxL, l_ele, Nary_tensor,
                    pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list,
                    pmfs_norm, optim_tensor,
                    encoder, y_strings
                )

        return y_strings, z_strings, z.size()[-2:], y, z, means_hat, scales_hat

    def decode_dpict(self, args, x_in, y_strings, z_strings, z_shape, **kwargs):
        assert isinstance(y_strings, list)
        z_hat = self.entropy_bottleneck.decompress([z_strings], z_shape)

        params = self.h_s(z_hat)
        gaussian_params = self.entropy_parameters(params)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        scales_hat = torch.clamp(scales_hat, min=0.04)

        x_in_name = args.input_image.split("/")[-1].split(".")[0]
        mkfulldir(os.path.join(args.recons_path, x_in_name))

        x_hats_pathlist = [
            os.path.join(args.recons_path, x_in_name, string.split("/")[-1][:-3] + "png")
            for string in y_strings
        ]

        metrics = np.zeros([len(y_strings), 3])
        path_index = 0
        bpp = 8 * len(open("/".join(y_strings[0].split("/")[:-1]) + "/z.bin", "rb").read())
        num_pixels = x_in.shape[2] * x_in.shape[3]

        device, maxL, l_ele, _ = get_empty_Nary_tensor(scales_hat)
        pmf_l_tensor = torch.div((mode ** l_ele.reshape(-1)), 2, rounding_mode="floor")
        Nary_tensor = torch.zeros([pmf_l_tensor.size(0)] + [maxL]).int().to(device)
        pmf_center_list = [(mode ** (maxL - j)) // 2 for j in range(maxL)]

        pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list = make_pmf_table(scales_hat, device, maxL, l_ele)

        for i in range(maxL):
            if i < maxL - opt_pnum:
                decoder = get_ans(type="dec")
                bitstream = open(y_strings[i], "rb").read()
                bpp += 8 * len(bitstream)
                decoder.set_stream(bitstream)

                pmfs_norm = list(map(lambda p, idx: (p * idx).view(p.size(0), mode, p.size(-1) // mode).sum(-1) / (p * idx).view(p.size(0), 1, p.size(-1)).sum(-1), pmfs_list[:i + 1], idx_ts_list[:i + 1]))

                y_hat = TP_entropy_decoding(
                    i, device, maxL, l_ele, Nary_tensor,
                    pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list,
                    pmfs_norm,
                    decoder, means_hat, pmf_center_list
                )

                x_hat = self.g_s(y_hat).clamp_(0, 1)
                ToPILImage()(x_hat[0]).save(x_hats_pathlist[path_index])

                metrics[path_index, 0] = bpp / num_pixels
                metrics[path_index, 1] = -10 * math.log10(F.mse_loss(x_in, x_hat).item())
                metrics[path_index, 2] = ms_ssim(x_in, x_hat, data_range=1.0).item()
                path_index += 1

            else:
                optim_tensor, pmfs_norm = get_transmission_tensor(i, maxL, pmfs_list, xpmfs_list, x2pmfs_list)

                cond_cdf, total_symbols, cdf_lengths, offsets, sl, points_num = prepare_TPED_scalable(
                    i, device, maxL, l_ele, pmfs_norm, optim_tensor
                )

                decoded_rvs = []
                for point in range(points_num):
                    decoder = get_ans(type="dec")
                    for cp in y_strings:
                        if f"q{maxL - i - 1:02d}_{point + 1:03d}" in cp:
                            codepath = cp
                            break
                    bitstream = open(codepath, "rb").read()
                    bpp += 8 * len(bitstream)
                    decoder.set_stream(bitstream)
                    if point == points_num - 1:
                        y_hat = TPED_last_point(
                            i, device, maxL, l_ele, Nary_tensor,
                            pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list,
                            optim_tensor,
                            point, cond_cdf, total_symbols, cdf_lengths, offsets, sl,
                            decoder, decoded_rvs,
                            means_hat, pmf_center_list
                        )
                        x_hat = self.g_s(y_hat).clamp_(0, 1)
                        ToPILImage()(x_hat[0]).save(x_hats_pathlist[path_index])

                        metrics[path_index, 0] = bpp / num_pixels
                        metrics[path_index, 1] = -10 * math.log10(F.mse_loss(x_in, x_hat).item())
                        metrics[path_index, 2] = ms_ssim(x_in, x_hat, data_range=1.0).item()
                        path_index += 1
                        break

                    y_hat = TPED(
                        i, device, maxL, l_ele, Nary_tensor,
                        pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list,
                        optim_tensor,
                        point, cond_cdf, total_symbols, cdf_lengths, offsets, sl,
                        decoder, decoded_rvs,
                        means_hat, pmf_center_list
                    )
                    x_hat = self.g_s(y_hat).clamp_(0, 1)
                    ToPILImage()(x_hat[0]).save(x_hats_pathlist[path_index])

                    metrics[path_index, 0] = bpp / num_pixels
                    metrics[path_index, 1] = -10 * math.log10(F.mse_loss(x_in, x_hat).item())
                    metrics[path_index, 2] = ms_ssim(x_in, x_hat, data_range=1.0).item()
                    path_index += 1

            del x_hat, y_hat
            torch.cuda.empty_cache()

        return metrics

    def encode_and_save_bitstreams_dpict(self, args):
        compressai.set_entropy_coder('ans')

        x = path2torch(args.input_image).to(args.device)
        orig_x_size = x.shape[2:]
        image_name = args.input_image.split("/")[-1].split(".")[0]
        bit_path = os.path.join(args.bit_path, image_name)
        mkfulldir(bit_path)
        p = 64  # maximum 6 strides of 2
        x = pad(x, p)

        enc_start = time.time()
        y_strings, z_strings, z_shape, _, _, _, _ = self.encode_dpict(x)
        enc_time = time.time() - enc_start
        print(f"Enc {enc_time:.1f}sec, ", end="\t")

        with open(bit_path + f"/z.bin", "wb") as f:
            write_uints(f, (z_shape[0], z_shape[1]))
            f.write(z_strings[0])

        max_L = len(y_strings) - 1
        pre_indexing = 0

        for i, code in enumerate(y_strings):
            if len(code) == 1:
                with open(bit_path + f"/{pre_indexing:03d}_q{max_L - i:02d}.bin", "wb") as f:
                    f.write(code[0])
                pre_indexing += 1
            else:
                for j, subcode in enumerate(code):
                    with open(bit_path + f"/{pre_indexing:03d}_q{max_L - i:02d}_{j + 1:03d}.bin", "wb") as f:
                        f.write(subcode)
                    pre_indexing += 1

        return enc_time

    def evaluate_dpict(self, args):
        enc_time = self.encode_and_save_bitstreams_dpict(args)

        image_name = args.input_image.split("/")[-1].split(".")[0]
        bit_path = os.path.join(args.bit_path, image_name)
        y_strings_list = listfulldir(bit_path)

        for y_string in y_strings_list:
            if "z.bin" in y_string:
                y_strings_list.remove(y_string)
                break
        num_recons = len(y_strings_list)

        z_strings_list = open(os.path.join(bit_path, "z.bin"), "rb")
        z_shape = read_uints(z_strings_list, 2)
        z_strings = z_strings_list.read()

        x_in = path2torch(args.input_image).to(args.device)

        dec_time = time.time()
        metric_data = self.decode_dpict(args, x_in, y_strings_list, z_strings, z_shape)
        dec_time = time.time() - dec_time

        return metric_data, enc_time, dec_time, num_recons

    @staticmethod
    def _standardized_cumulative(inputs):
        half = float(0.5)
        const = float(-(2 ** -0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    @staticmethod
    def _pmf_to_cdf(pmf, tail_mass, pmf_length, max_length):
        cdf = torch.zeros(
            (len(pmf_length), max_length + 2), dtype=torch.int32, device=pmf.device
        )
        for i, p in enumerate(pmf):
            prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = _pmf_to_quantized_cdf(prob.tolist(), 16)
            _cdf = torch.IntTensor(_cdf)
            cdf[i, : _cdf.size(0)] = _cdf
        return cdf