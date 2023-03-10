from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
from compressai.ans import BufferedRansEncoder, RansDecoder

import torch
import scipy
import math

multiplier = -scipy.stats.norm.ppf(1e-09 / 2)
mode = 3
opt_pnum = 5
pnum_btw_trit = 48
pnum_part = 1.0


__all__ = [
    "opt_pnum",
    "mode",
    "get_Nary_tensor",
    "make_pmf_table",
    "get_ans",
    "TP_entropy_encoding",
    "get_transmission_tensor",
    "TP_entropy_encoding_scalable",
    "get_empty_Nary_tensor",
    "TP_entropy_decoding",
    "TPED_last_point",
    "TPED",
    "prepare_TPED_scalable",
]


def get_ans(type):
    if type == "enc":
        return BufferedRansEncoder()
    elif type == "dec":
        return RansDecoder()
    else:
        raise ValueError(f"type must be 'enc' or 'dec'")


def _pmf_to_cdf(pmf, tail_mass, pmf_length, max_length):
    cdf = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32, device=pmf.device)
    for i, p in enumerate(pmf):
        prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
        _cdf = _pmf_to_quantized_cdf(prob.tolist(), 16)
        _cdf = torch.IntTensor(_cdf)
        cdf[i, : _cdf.size(0)] = _cdf
    return cdf


def _pmf_to_cdf_tensor(pmf, tail_mass):
    _cdf = torch.cat([pmf, tail_mass], dim=1).clamp_(min=2e-05)
    _cdf = torch.round((_cdf / _cdf.sum(dim=1, keepdim=True)).cumsum(dim=1) * (2 ** 16))
    _cdf = torch.cat([torch.zeros_like(tail_mass), _cdf], dim=1).int()
    return _cdf


def _standardized_cumulative(inputs):
    half = float(0.5)
    const = float(-(2 ** -0.5))
    # Using the complementary error function maximizes numerical precision.
    return half * torch.erfc(const * inputs)


def _pnum_part(i, max_L):
    if i == max_L - 0:
        pnum_part = 24 / 24
    elif i == max_L - 1:
        pnum_part = 24 / 24
    elif i == max_L - 2:
        pnum_part = 24 / 24
    elif i == max_L - 3:
        pnum_part = 16 / 24
    elif i == max_L - 4:
        pnum_part = 8 / 24
    elif i == max_L - 5:
        pnum_part = 8 / 24
    elif i == max_L - 6:
        pnum_part = 3 / 48
    else:
        pnum_part = 1 / 48
    return pnum_part


def get_empty_Nary_tensor(scales_hat):
    device = scales_hat.device
    scales_hat = torch.clamp(scales_hat, min=0.04)
    tail = scales_hat * multiplier * 2
    l_ele = torch.ceil(torch.log(tail) / torch.log(torch.Tensor([mode]).squeeze())).int()
    l_ele = torch.clamp(l_ele, 1, l_ele.max().item())
    maxL = l_ele.max().item()

    if torch.sum(l_ele == l_ele.max()) < 2:
        maxL = l_ele.max().item() - 1
        l_ele = torch.clamp(l_ele, 1, maxL)

    Nary_tensor = torch.zeros(list(scales_hat.shape) + [maxL]).int().to(device)
    return device, maxL, l_ele, Nary_tensor


def get_Nary_tensor(y, means_hat, scales_hat):
    device, maxL, l_ele, Nary_tensor = get_empty_Nary_tensor(scales_hat)

    symbol_tensor = torch.round(y - means_hat).int() + torch.div(mode ** l_ele, 2, rounding_mode="floor")
    symbol_tensor = torch.clamp(symbol_tensor, min=torch.zeros(y.shape).int().to(device), max=3 ** l_ele - 1)

    for i in range(1, maxL + 1):
        Nary_tensor[:, :, :, :, i - 1] = torch.div(symbol_tensor, (mode ** (maxL - i)), rounding_mode="floor")
        symbol_tensor = symbol_tensor % (mode ** (maxL - i))

    Nary_tensor = Nary_tensor.view(-1, maxL)
    del symbol_tensor
    torch.cuda.empty_cache()

    return device, maxL, l_ele, Nary_tensor


def make_pmf_table(scales_hat, device, maxL, l_ele):
    pmfs_list = []
    xpmfs_list = []
    x2pmfs_list = []
    idx_ts_list = []

    for i in range(1, maxL + 1):
        pmf_length = mode ** i
        pmf_center = pmf_length // 2
        samples = torch.abs(torch.arange(pmf_length, device=device).repeat((l_ele == i).sum(), 1) - pmf_center)
        upper = _standardized_cumulative((0.5 - samples) / scales_hat.reshape(-1, 1)[l_ele.reshape(-1) == i])
        lower = _standardized_cumulative((-0.5 - samples) / scales_hat.reshape(-1, 1)[l_ele.reshape(-1) == i])
        pmfs_ = upper - lower
        pmfs_ = (pmfs_ + 1e-10) / (pmfs_ + 1e-10).sum(dim=-1).unsqueeze(-1)
        pmfs_list.insert(0, pmfs_.clone())
        del upper, lower, samples
        torch.cuda.empty_cache()
        idx_tmp = torch.arange(mode ** i, device=device).repeat(pmfs_.size(0), 1)
        xpmfs_ = pmfs_ * idx_tmp
        xpmfs_list.insert(0, xpmfs_.clone())
        x2pmfs_ = pmfs_ * torch.pow(idx_tmp, 2)
        x2pmfs_list.insert(0, x2pmfs_.clone())
        idx_ts_list.insert(0, torch.ones_like(pmfs_list[0], device=device))
        del idx_tmp, pmfs_, xpmfs_, x2pmfs_
        torch.cuda.empty_cache()

    return pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list


def select_sub_interval(i, device, maxL, l_ele, Nary_tensor,
                        pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list):
    p_len = mode ** (maxL - 1 - i)

    for j in range(i + 1):
        num_pmf = pmfs_list[j].size(0)
        Nary_part = Nary_tensor[l_ele.reshape(-1) == maxL - j][:, i:i + 1]
        tmp_ = torch.arange(mode ** (maxL - i), device=device).repeat(Nary_part.size(0), 1).int()
        idx_ts_list[j] *= (torch.div((tmp_ % (p_len * mode)), p_len, rounding_mode="floor") == Nary_part)
        nz_idx = idx_ts_list[j].nonzero(as_tuple=True)
        pmfs_list[j] = pmfs_list[j][nz_idx].view(num_pmf, p_len)
        xpmfs_list[j] = xpmfs_list[j][nz_idx].view(num_pmf, p_len)
        x2pmfs_list[j] = x2pmfs_list[j][nz_idx].view(num_pmf, p_len)
        idx_ts_list[j] = idx_ts_list[j][nz_idx].view(num_pmf, p_len)


def get_transmission_tensor(i, maxL, pmfs_list, xpmfs_list, x2pmfs_list):
    p_len = mode ** (maxL - 1 - i)

    pmfs_list_l = pmfs_list[:i + 1]
    xpmfs_list_l = xpmfs_list[:i + 1]
    x2pmfs_list_l = x2pmfs_list[:i + 1]
    m_old = list(map(lambda x, y: x.sum(dim=-1) / y.sum(dim=-1), xpmfs_list_l, pmfs_list_l))
    D_old = list(map(lambda x2p, p, m: (x2p.sum(-1) - (m ** 2) * p.sum(-1)) / p.sum(-1), x2pmfs_list_l, pmfs_list_l, m_old))

    pmfs_cond_list_l = list(map(lambda x: x.view(x.size(0), mode, p_len).sum(-1), pmfs_list_l))
    xpmfs_cond_list_l = list(map(lambda xp: xp.view(xp.size(0), mode, p_len).sum(-1), xpmfs_list_l))
    x2pmfs_cond_list_l = list(map(lambda x: x.view(x.size(0), mode, p_len).sum(-1), x2pmfs_list_l))

    pmfs_norm = list(map(lambda p: p / p.sum(-1).view(-1, 1), pmfs_cond_list_l))

    m_new = list(map(lambda xp, p: xp / p, xpmfs_cond_list_l, pmfs_cond_list_l))
    D_new = list(map(lambda x2p, p, m, fullp: ((x2p - (m ** 2) * p) / fullp.sum(-1).view(-1, 1)).sum(-1),
                     x2pmfs_cond_list_l, pmfs_cond_list_l, m_new, pmfs_list_l))
    delta_D = list(map(lambda old, new: (new - old).clamp_(max=0), D_old, D_new))
    delta_R = list(map(lambda p: (-p * torch.log2(p)).sum(-1), pmfs_norm))
    delta_R = list(map(lambda h: h * (h >= 0), delta_R))

    optim_tensor = torch.cat(list(map(lambda D, R: -(D / R), delta_D, delta_R))).clamp_(min=0)

    return optim_tensor, pmfs_norm


def TP_entropy_encoding(i, device, maxL, l_ele, Nary_tensor,
                        pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list,
                        pmfs_norm,
                        encoder, y_strings):
    tail_mass = list(map(lambda p: torch.zeros([len(p), 1]).to(device) + 1e-09, pmfs_norm))
    cond_cdf = torch.cat(list(map(lambda p, tm: _pmf_to_cdf_tensor(p, tm), pmfs_norm, tail_mass)), dim=0).tolist()

    total_symbols_list = torch.cat([Nary_tensor[l_ele.reshape(-1) == maxL - j, i] - (mode // 2) for j in range(i + 1)]).tolist()
    indexes_list = list(range(len(total_symbols_list)))
    cdf_lengths = [mode + 2 for _ in range(len(total_symbols_list))]

    offsets = [-(mode // 2) for _ in range(len(total_symbols_list))]
    encoder.encode_with_indexes(
        total_symbols_list, indexes_list, cond_cdf, cdf_lengths, offsets
    )
    del pmfs_norm, tail_mass, cond_cdf
    torch.cuda.empty_cache()
    y_strings[i].append(encoder.flush())

    select_sub_interval(i, device, maxL, l_ele, Nary_tensor,
                        pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list)


def TP_entropy_encoding_scalable(i, device, maxL, l_ele, Nary_tensor,
                                 pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list,
                                 pmfs_norm, optim_tensor,
                                 encoder, y_strings):
    tail_mass = list(map(lambda p: torch.zeros([len(p), 1]).to(device) + 1e-09, pmfs_norm))
    cond_cdf = torch.cat(list(map(lambda p, tm: _pmf_to_cdf_tensor(p, tm), pmfs_norm, tail_mass)), dim=0)
    cond_cdf = cond_cdf[torch.argsort(optim_tensor, descending=True)].tolist()

    total_symbols_list = torch.cat([Nary_tensor[l_ele.reshape(-1) == maxL - j, i] - (mode // 2) for j in range(i + 1)])
    total_symbols_list = total_symbols_list[torch.argsort(optim_tensor, descending=True)].tolist()
    total_symbols = len(total_symbols_list)
    cdf_lengths = [mode + 2 for _ in range(total_symbols)]
    offsets = [-(mode // 2) for _ in range(total_symbols)]

    torch.cuda.empty_cache()

    # sl = total_symbols // pnum_btw_trit
    pnum_part = _pnum_part(i, maxL)
    points_num = math.ceil(pnum_btw_trit * pnum_part)

    sl = total_symbols // points_num

    for point in range(points_num):
        if point == points_num - 1:
            symbols_list = total_symbols_list[point * sl:]
            indexes_list = list(range(len(symbols_list)))
            encoder.encode_with_indexes(
                symbols_list,
                indexes_list,
                cond_cdf[point * sl:],
                cdf_lengths[point * sl:],
                offsets[point * sl:]
            )
            y_strings[i].append(encoder.flush())
            break

        symbols_list = total_symbols_list[point * sl:(point + 1) * sl]
        indexes_list = list(range(len(symbols_list)))
        encoder.encode_with_indexes(
            symbols_list,
            indexes_list,
            cond_cdf[point * sl:(point + 1) * sl],
            cdf_lengths[point * sl:(point + 1) * sl],
            offsets[point * sl:(point + 1) * sl]
        )
        y_strings[i].append(encoder.flush())

        encoder = BufferedRansEncoder()

    select_sub_interval(i, device, maxL, l_ele, Nary_tensor,
                        pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list)


def TP_entropy_decoding(i, device, maxL, l_ele, Nary_tensor,
                        pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list,
                        pmfs_norm,
                        decoder, means_hat, pmf_center_list, is_recon):
    tail_mass = list(map(lambda p: torch.zeros([len(p), 1]).to(device) + 1e-09, pmfs_norm))
    cond_cdf = torch.cat(list(map(lambda p, tm: _pmf_to_cdf_tensor(p, tm), pmfs_norm, tail_mass)), dim=0).tolist()

    symbols_num = (l_ele.reshape(-1) >= maxL - i).sum().item()
    indexes_list = list(range(symbols_num))
    cdf_lengths = [mode + 2 for _ in range(symbols_num)]
    offsets = [-(mode // 2) for _ in range(symbols_num)]
    rv = decoder.decode_stream(
        indexes_list, cond_cdf, cdf_lengths, offsets
    )
    rv = (torch.Tensor(rv) - torch.Tensor(offsets)).int().to(device)
    tmp_idx = 0
    for j in range(i + 1):
        if j == 0:
            tmp_idx += len(pmfs_list[j])
            Nary_tensor[l_ele.reshape(-1) == maxL - j, i] = rv[:tmp_idx]
        elif j == i:
            Nary_tensor[l_ele.reshape(-1) == maxL - j, i] = rv[tmp_idx:]
        else:
            Nary_tensor[l_ele.reshape(-1) == maxL - j, i] = rv[tmp_idx:tmp_idx + len(pmfs_list[j])]
            tmp_idx += len(pmfs_list[j])

    select_sub_interval(i, device, maxL, l_ele, Nary_tensor,
                        pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list)

    if is_recon:
        recon = list(map(lambda xp, p, l: (xp.sum(-1) / p.sum(-1)) - l, xpmfs_list, pmfs_list, pmf_center_list))
        y_hat = means_hat.clone().reshape(-1)
        for j in range(i + 1):
            y_hat[l_ele.reshape(-1) == maxL - j] += recon[j]
        y_hat = y_hat.reshape(means_hat.shape)
    else:
        y_hat = -1

    return y_hat


def prepare_TPED_scalable(i, device, maxL, l_ele,
                           pmfs_norm, optim_tensor):
    tail_mass = list(map(lambda p: torch.zeros([len(p), 1]).to(device) + 1e-09, pmfs_norm))
    cond_cdf = torch.cat(list(map(lambda p, tm: _pmf_to_cdf_tensor(p, tm), pmfs_norm, tail_mass)), dim=0)
    cond_cdf = cond_cdf[torch.argsort(optim_tensor, descending=True)].tolist()

    total_symbols = (l_ele.reshape(-1) >= maxL - i).sum().item()
    cdf_lengths = [mode + 2 for _ in range(total_symbols)]
    offsets = [-(mode // 2) for _ in range(total_symbols)]
    del tail_mass

    torch.cuda.empty_cache()

    pnum_part = _pnum_part(i, maxL)
    points_num = math.ceil(pnum_btw_trit * pnum_part)

    sl = total_symbols // points_num

    return cond_cdf, total_symbols, cdf_lengths, offsets, sl, points_num


def TPED_last_point(i, device, maxL, l_ele, Nary_tensor,
                     pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list,
                     optim_tensor,
                     point, cond_cdf, total_symbols, cdf_lengths, offsets, sl,
                     decoder, decoded_rvs,
                     means_hat, pmf_center_list, is_recon):
    p_len = mode ** (maxL - 1 - i)

    symbols_num_part = total_symbols - point * sl
    indexes_list = list(range(symbols_num_part))
    rv = decoder.decode_stream(
        indexes_list,
        cond_cdf[point * sl:],
        cdf_lengths[point * sl:],
        offsets[point * sl:]
    )
    rv = (torch.Tensor(rv) - torch.Tensor(offsets[point * sl:])).int().to(device)
    decoded_rvs.append(rv.clone())
    rv = torch.cat(decoded_rvs)
    Nary_tensor_tmp = rv[torch.argsort(torch.argsort(optim_tensor, descending=True), descending=False)].int()
    tmp_idx = 0
    for j in range(i + 1):
        if j == 0:
            tmp_idx += len(pmfs_list[j])
            Nary_tensor[l_ele.reshape(-1) == maxL - j, i] = Nary_tensor_tmp[:tmp_idx]
        elif j == i:
            Nary_tensor[l_ele.reshape(-1) == maxL - j, i] = Nary_tensor_tmp[tmp_idx:]
        else:
            Nary_tensor[l_ele.reshape(-1) == maxL - j, i] = Nary_tensor_tmp[tmp_idx:tmp_idx + len(pmfs_list[j])]
            tmp_idx += len(pmfs_list[j])

    for j in range(i + 1):
        Nary_part = Nary_tensor[l_ele.reshape(-1) == maxL - j][:, i:i + 1]
        tmp_ = torch.arange(mode ** (maxL - i), device=device).repeat(Nary_part.size(0), 1).int()
        idx_ts_list[j] *= (torch.div((tmp_ % (mode * p_len)), p_len, rounding_mode="floor") == Nary_part)
        nz_idx = idx_ts_list[j].nonzero(as_tuple=True)
        num_pmf = pmfs_list[j].size(0)
        size_pmf = pmfs_list[j].size(1) // mode
        pmfs_list[j] = pmfs_list[j][nz_idx].view(num_pmf, size_pmf)
        xpmfs_list[j] = xpmfs_list[j][nz_idx].view(num_pmf, size_pmf)
        x2pmfs_list[j] = x2pmfs_list[j][nz_idx].view(num_pmf, size_pmf)
        idx_ts_list[j] = idx_ts_list[j][nz_idx].view(num_pmf, size_pmf)

    if is_recon:
        recon = list(map(lambda xp, p, l: (xp.sum(-1) / p.sum(-1)) - l,
                         xpmfs_list, pmfs_list, pmf_center_list))
        y_hat = means_hat.clone().reshape(-1)
        for j in range(i + 1):
            y_hat[l_ele.reshape(-1) == maxL - j] += recon[j]
        y_hat = y_hat.reshape(means_hat.shape)
    else:
        y_hat = -1

    return y_hat


def TPED(i, device, maxL, l_ele, Nary_tensor,
          pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list,
          optim_tensor,
          point, cond_cdf, total_symbols, cdf_lengths, offsets, sl,
          decoder, decoded_rvs,
          means_hat, pmf_center_list, is_recon):
    p_len = mode ** (maxL - 1 - i)

    indexes_list = list(range(sl))
    rv = decoder.decode_stream(
        indexes_list,
        cond_cdf[point * sl:(point + 1) * sl],
        cdf_lengths[point * sl:(point + 1) * sl],
        offsets[point * sl:(point + 1) * sl]
    )
    rv = (torch.Tensor(rv) - torch.Tensor(offsets[point * sl:(point + 1) * sl])).int().to(device)
    decoded_rvs.append(rv.clone())

    pre_cat = torch.cat(decoded_rvs)
    post_cat = torch.zeros([total_symbols - (point + 1) * sl]).to(device) - 1
    rv = torch.cat([pre_cat, post_cat])

    Nary_tensor_tmp = rv[torch.argsort(torch.argsort(optim_tensor, descending=True), descending=False)].int()
    tmp_idx = 0
    for j in range(i + 1):
        if j == 0:
            tmp_idx += len(pmfs_list[j])
            Nary_tensor[l_ele.reshape(-1) == maxL - j, i] = Nary_tensor_tmp[:tmp_idx]
        elif j == i:
            Nary_tensor[l_ele.reshape(-1) == maxL - j, i] = Nary_tensor_tmp[tmp_idx:]
        else:
            Nary_tensor[l_ele.reshape(-1) == maxL - j, i] = \
                Nary_tensor_tmp[tmp_idx:tmp_idx + len(pmfs_list[j])]
            tmp_idx += len(pmfs_list[j])

    for j in range(i + 1):
        Nary_part = Nary_tensor[l_ele.reshape(-1) == maxL - j][:, i][Nary_tensor[l_ele.reshape(-1) == maxL - j][:, i] != -1]
        tmp_ = torch.arange(mode ** (maxL - i), device=device).repeat(Nary_part.size(0), 1).int()
        idx_ts_list[j][Nary_tensor[l_ele.reshape(-1) == maxL - j][:, i] != -1] *= (torch.div((tmp_ % (p_len * 3)), p_len, rounding_mode="floor") == Nary_part.view(-1, 1))
        pmfs_list[j][Nary_tensor[l_ele.reshape(-1) == maxL - j][:, i] != -1] *= idx_ts_list[j][Nary_tensor[l_ele.reshape(-1) == maxL - j][:, i] != -1]
        xpmfs_list[j][Nary_tensor[l_ele.reshape(-1) == maxL - j][:, i] != -1] *= idx_ts_list[j][Nary_tensor[l_ele.reshape(-1) == maxL - j][:, i] != -1]
        x2pmfs_list[j][Nary_tensor[l_ele.reshape(-1) == maxL - j][:, i] != -1] *= idx_ts_list[j][Nary_tensor[l_ele.reshape(-1) == maxL - j][:, i] != -1]
    Nary_tensor[Nary_tensor < 0] = 0

    if is_recon:
        recon = list(map(lambda xp, p, l: (xp.sum(-1) / p.sum(-1)) - l, xpmfs_list, pmfs_list, pmf_center_list))
        y_hat = means_hat.clone().reshape(-1)
        for j in range(i + 1):
            y_hat[l_ele.reshape(-1) == maxL - j] += recon[j]
        y_hat = y_hat.reshape(means_hat.shape)
    else:
        y_hat = -1

    return y_hat
