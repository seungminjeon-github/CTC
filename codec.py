import os
import sys
import argparse

import torch

from models import model_CTC
from utils import path2torch, torch2img, psnr

torch.autograd.set_detect_anomaly(True)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument("--mode", type=str, choices=["enc", "dec"], default="enc")
    parser.add_argument("--save-path", type=str, default="results")
    parser.add_argument("--input-file", type=str, default=None)
    parser.add_argument("--recon-level", type=int, choices=list(range(1, 161)), default=160)
    parser.add_argument("--cuda", action="store_true", default=False)

    args = parser.parse_args(argv)
    return args


def _enc(args, net):
    x = path2torch(args.input_file).to(args.device)
    save_path_enc = os.path.join(args.save_path, "bits")
    if not os.path.exists(save_path_enc): os.mkdir(save_path_enc)
    net.encode_and_save_bitstreams_ctc(x, save_path_enc)


def _dec(args, net):
    save_path_dec = os.path.join(args.save_path, "recon")
    if not os.path.exists(save_path_dec): os.mkdir(save_path_dec)
    dec_time, x_rec, bpp = net.reconstruct_ctc(args)
    torch2img(x_rec).save(f"{save_path_dec}/q{args.recon_level:04d}.png")

    print(f"dec time: {dec_time:.3f}, bpp: {bpp:.5f}", end=" ")

    if args.input_file is not None:
        x_in = path2torch(args.input_file).to(args.device)
        metric = psnr(x_in, x_rec)
        print(f", psnr: {metric:.4f}")


def main(argv):
    args = parse_args(argv)
    args.device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    net = model_CTC(N=192).to(args.device)
    ckpt = torch.load("ctc.pt")["state_dict"]
    net.load_state_dict(ckpt)
    net.update()

    if args.mode == "enc":
        _enc(args, net)

    elif args.mode == "dec":
        _dec(args, net)

    else:
        raise ValueError(f"{args.mode} error: choose 'enc' or 'dec'.")


if __name__ == "__main__":
    main(sys.argv[1:])