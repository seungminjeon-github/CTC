from ..dpict.dpict import model_baseline

from ..utils_trit_plane import *

from .util import CDR_v000 as CDR_v0
from .util import classifier_v003_7_sc as CRR_v0
from .util import post_processing_crr

from utils import path2torch, pad, write_uints, read_uints, mkfulldir, listfulldir

import os
import time
import math
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from pytorch_msssim import ms_ssim

import compressai


class model_CTC(model_baseline):
    def __init__(self, N=192, **kwargs):
        super().__init__(N=N, **kwargs)

        self.CRR1 = CRR_v0(N=N)
        self.CRR2 = CRR_v0(N=N)
        self.CRR3 = CRR_v0(N=N)

        self.CDR1 = CDR_v0(N=N)
        self.CDR2 = CDR_v0(N=N)
        self.CDR3 = CDR_v0(N=N)

    def encode_ctc(self, x):
        mode = 3

        y, z, z_strings, means_hat, scales_hat = self.repr(x)
        scales_hat = torch.clamp(scales_hat, min=0.04)

        device, maxL, l_ele, Nary_tensor = get_Nary_tensor(y, means_hat, scales_hat)
        pmf_center_list = [(mode ** (maxL - j)) // 2 for j in range(maxL)]
        pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list = make_pmf_table(scales_hat, device, maxL, l_ele)

        y_strings = [[] for _ in range(maxL)]
        y_hat_target = means_hat.clone()

        for i in range(maxL):
            encoder = get_ans(type="enc")
            p_len = mode ** (maxL - 1 - i)

            if i < maxL - opt_pnum:
                pmfs_norm = list(
                    map(
                        lambda p, idx:
                        (p * idx).view(p.size(0), mode, p.size(-1) // mode).sum(-1) / (p * idx).view(p.size(0), 1, p.size(-1)).sum(-1), pmfs_list[:i + 1], idx_ts_list[:i + 1]
                    )
                )

                if i >= maxL - 6:
                    if i == maxL - 1:
                        crr = self.CRR1
                    elif i == maxL - 2:
                        crr = self.CRR2
                    else:
                        crr = self.CRR3

                    prob_tensor, q = post_processing_crr(
                        i, device, p_len, l_ele, maxL,
                        scales_hat, means_hat,
                        pmfs_list, xpmfs_list, pmf_center_list, pmfs_norm
                    )

                    Q = crr(y_hat=y_hat_target, P=prob_tensor, q=q, scales_hat=scales_hat, means_hat=means_hat)

                    for j in range(i + 1):
                        pmfs_norm[j] = Q[l_ele == maxL - j]

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

                if i >= maxL - 6:
                    if i == maxL - 1:
                        crr = self.CRR1
                    elif i == maxL - 2:
                        crr = self.CRR2
                    else:
                        crr = self.CRR3

                    prob_tensor, q = post_processing_crr(
                        i, device, p_len, l_ele, maxL,
                        scales_hat, means_hat,
                        pmfs_list, xpmfs_list, pmf_center_list, pmfs_norm
                    )

                    Q = crr(y_hat=y_hat_target, P=prob_tensor, q=q, scales_hat=scales_hat, means_hat=means_hat)

                    for j in range(i + 1):
                        pmfs_norm[j] = Q[l_ele == maxL - j]

                TP_entropy_encoding_scalable(
                    i, device, maxL, l_ele, Nary_tensor,
                    pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list,
                    pmfs_norm, optim_tensor,
                    encoder, y_strings
                )

            recon = list(map(lambda xp, p, l: (xp.sum(-1) / p.sum(-1)) - l, xpmfs_list, pmfs_list, pmf_center_list))
            y_hat_target = means_hat.clone()
            for j in range(i + 1):
                y_hat_target[l_ele == maxL - j] += recon[j]

        return y_strings, z_strings, z.size()[-2:], y, z, means_hat, scales_hat

    def decode_ctc(self, y_strings, z_strings, z_shape):
        mode = 3

        assert isinstance(y_strings, list)
        z_hat = self.entropy_bottleneck.decompress([z_strings], z_shape)
        bpp = 0

        params = self.h_s(z_hat)
        gaussian_params = self.entropy_parameters(params)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        scales_hat = torch.clamp(scales_hat, min=0.04)

        path_index = 0

        device, maxL, l_ele, _ = get_empty_Nary_tensor(scales_hat)
        pmf_l_tensor = torch.div((mode ** l_ele.reshape(-1)), 2, rounding_mode="floor")
        Nary_tensor = torch.zeros([pmf_l_tensor.size(0)] + [maxL]).int().to(device)
        pmf_center_list = [(mode ** (maxL - j)) // 2 for j in range(maxL)]

        pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list = make_pmf_table(scales_hat, device, maxL, l_ele)
        y_hat_target = means_hat.clone()

        for i in range(maxL):
            p_len = mode ** (maxL - 1 - i)

            if i < maxL - opt_pnum:
                decoder = get_ans(type="dec")
                bitstream = open(y_strings[path_index], "rb").read()
                bpp += 8 * len(bitstream)
                decoder.set_stream(bitstream)

                pmfs_norm = list(
                    map(
                        lambda p, idx:
                        (p * idx).view(p.size(0), mode, p.size(-1) // mode).sum(-1) / (p * idx).view(p.size(0), 1, p.size(-1)).sum(-1), pmfs_list[:i + 1], idx_ts_list[:i + 1]
                    )
                )

                if i >= maxL - 6:
                    if i == maxL - 1:
                        crr = self.CRR1
                    elif i == maxL - 2:
                        crr = self.CRR2
                    else:
                        crr = self.CRR3

                    prob_tensor, q = post_processing_crr(
                        i, device, p_len, l_ele, maxL,
                        scales_hat, means_hat,
                        pmfs_list, xpmfs_list, pmf_center_list, pmfs_norm
                    )

                    Q = crr(y_hat=y_hat_target, P=prob_tensor, q=q, scales_hat=scales_hat, means_hat=means_hat)

                    for j in range(i + 1):
                        pmfs_norm[j] = Q[l_ele == maxL - j]

                y_hat = TP_entropy_decoding(
                    i, device, maxL, l_ele, Nary_tensor,
                    pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list,
                    pmfs_norm,
                    decoder, means_hat, pmf_center_list, is_recon=path_index == len(y_strings) - 1
                )

                if path_index == len(y_strings) - 1:
                    if maxL - 6 <= i < maxL - 1:
                        if i == maxL - 2:
                            cdr = self.CDR1
                        elif i == maxL - 3:
                            cdr = self.CDR2
                        else:
                            cdr = self.CDR3

                        interval_min = means_hat.clone()
                        interval_max = means_hat.clone()

                        for j in range(maxL):
                            interval_min[l_ele == maxL - j] += torch.round(xpmfs_list[j] / pmfs_list[j]).int().min(dim=-1).values - pmf_center_list[j]
                            interval_max[l_ele == maxL - j] += torch.round(xpmfs_list[j] / pmfs_list[j]).int().max(dim=-1).values - pmf_center_list[j]

                        entropy_params = torch.cat([scales_hat, means_hat], dim=1)
                        y_hat = cdr(y_hat.view(scales_hat.shape), entropy_params, interval_min, interval_max)
                        return self.g_s(y_hat).clamp_(0, 1), bpp
                path_index += 1

            else:
                optim_tensor, pmfs_norm = get_transmission_tensor(i, maxL, pmfs_list, xpmfs_list, x2pmfs_list)

                if i >= maxL - 6:
                    if i == maxL - 1:
                        crr = self.CRR1
                    elif i == maxL - 2:
                        crr = self.CRR2
                    else:
                        crr = self.CRR3

                    prob_tensor, q = post_processing_crr(
                        i, device, p_len, l_ele, maxL,
                        scales_hat, means_hat,
                        pmfs_list, xpmfs_list, pmf_center_list, pmfs_norm
                    )

                    Q = crr(y_hat=y_hat_target, P=prob_tensor, q=q, scales_hat=scales_hat, means_hat=means_hat)

                    for j in range(i + 1):
                        pmfs_norm[j] = Q[l_ele == maxL - j]

                cond_cdf, total_symbols, cdf_lengths, offsets, sl, points_num = prepare_TPED_scalable(
                    i, device, maxL, l_ele, pmfs_norm, optim_tensor
                )

                decoded_rvs = []
                for point in range(points_num):
                    if path_index == len(y_strings) - 1:
                        a = 1
                    decoder = get_ans(type="dec")
                    codepath = y_strings[path_index]
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
                            means_hat, pmf_center_list, is_recon=path_index == len(y_strings) - 1
                        )
                        if path_index == len(y_strings) - 1:
                            if maxL - 6 <= i < maxL - 1:
                                if i == maxL - 2:
                                    cdr = self.CDR1
                                elif i == maxL - 3:
                                    cdr = self.CDR2
                                else:
                                    cdr = self.CDR3

                                interval_min = means_hat.clone()
                                interval_max = means_hat.clone()

                                for j in range(maxL):
                                    interval_min[l_ele == maxL - j] += torch.round(xpmfs_list[j] / pmfs_list[j]).int().min(dim=-1).values - pmf_center_list[j]
                                    interval_max[l_ele == maxL - j] += torch.round(xpmfs_list[j] / pmfs_list[j]).int().max(dim=-1).values - pmf_center_list[j]

                                entropy_params = torch.cat([scales_hat, means_hat], dim=1)
                                y_hat = cdr(y_hat.view(scales_hat.shape), entropy_params, interval_min, interval_max)
                            return self.g_s(y_hat).clamp_(0, 1), bpp

                        path_index += 1
                        break

                    y_hat = TPED(
                        i, device, maxL, l_ele, Nary_tensor,
                        pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list,
                        optim_tensor,
                        point, cond_cdf, total_symbols, cdf_lengths, offsets, sl,
                        decoder, decoded_rvs,
                        means_hat, pmf_center_list, is_recon=path_index == len(y_strings) - 1
                    )
                    if path_index == len(y_strings) - 1:
                        if maxL - 6 <= i < maxL - 1:
                            if i == maxL - 2:
                                cdr = self.CDR1
                            elif i == maxL - 3:
                                cdr = self.CDR2
                            else:
                                cdr = self.CDR3

                            interval_min = means_hat.clone()
                            interval_max = means_hat.clone()

                            for j in range(maxL):
                                interval_min[l_ele == maxL - j] += torch.round(xpmfs_list[j] / pmfs_list[j]).int().min(dim=-1).values - pmf_center_list[j]
                                interval_max[l_ele == maxL - j] += torch.round(xpmfs_list[j] / pmfs_list[j]).int().max(dim=-1).values - pmf_center_list[j]

                            entropy_params = torch.cat([scales_hat, means_hat], dim=1)
                            y_hat = cdr(y_hat.view(scales_hat.shape), entropy_params, interval_min, interval_max)
                        return self.g_s(y_hat).clamp_(0, 1), bpp

                    path_index += 1

            recon = list(map(lambda xp, p, l: (xp.sum(-1) / p.sum(-1)) - l, xpmfs_list, pmfs_list, pmf_center_list))
            y_hat_target = means_hat.clone()
            for j in range(i + 1):
                y_hat_target[l_ele == maxL - j] += recon[j]

        raise ValueError()

    def encode_and_save_bitstreams_ctc(self, x, save_path):
        compressai.set_entropy_coder('ans')

        p = 64  # maximum 6 strides of 2
        x = pad(x, p)

        enc_start = time.time()
        y_strings, z_strings, z_shape, _, _, _, _ = self.encode_ctc(x)
        enc_time = time.time() - enc_start
        print(f"Enc {enc_time:.1f}sec, ", end="\t")

        with open(save_path + f"/z.bin", "wb") as f:
            write_uints(f, (z_shape[0], z_shape[1]))
            f.write(z_strings[0])

        max_L = len(y_strings) - 1
        pre_indexing = 0

        for i, code in enumerate(y_strings):
            if len(code) == 1:
                with open(save_path + f"/{pre_indexing:03d}_q{max_L - i:02d}.bin", "wb") as f:
                    f.write(code[0])
                pre_indexing += 1
            else:
                for j, subcode in enumerate(code):
                    with open(save_path + f"/{pre_indexing:03d}_q{max_L - i:02d}_{j + 1:03d}.bin", "wb") as f:
                        f.write(subcode)
                    pre_indexing += 1

        return enc_time

    def reconstruct_ctc(self, args):
        bit_path = os.path.join(args.save_path, "bits")
        y_strings_list = listfulldir(bit_path)

        for y_string in y_strings_list:
            if "z.bin" in y_string:
                y_strings_list.remove(y_string)
                break

        recon_level = len(y_strings_list) - 161 + args.recon_level
        if args.recon_level < 0:
            raise ValueError(f"set [--recon-level] to larger integer value.")
        y_strings_list = y_strings_list[:recon_level]

        z_strings_list = open(os.path.join(bit_path, "z.bin"), "rb")
        z_shape = read_uints(z_strings_list, 2)
        z_strings = z_strings_list.read()

        dec_time = time.time()
        x_rec, bpp = self.decode_ctc(y_strings_list, z_strings, z_shape)
        bpp /= (x_rec.size(2) * x_rec.size(3))
        dec_time = time.time() - dec_time

        return dec_time, x_rec, bpp
