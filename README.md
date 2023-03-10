# CTC (CVPR 2023)
**"Context-based Trit-Plane Coding for Progressive Image Compression"** [[paper]](TBC)

Seungmin Jeon, Kwang Pyo Choi, Youngo Park, Chang-Su Kim (Corresponding author)

PyTorch-Based Official Code for CTC.

## Requirements
- pytorch 1.11.0 (cudatoolkit 11.0)
- torchvision 0.12.0
- [CompressAI](https://github.com/InterDigitalInc/CompressAI) 1.2.1
- Ubuntu 18.04 recommended

## Installation
Download [pre-trained model](https://drive.google.com/file/d/1q0IyOnOcl9E9Y07viYjmLmj3FkA3ZDMT/view?usp=sharing) parameters on the root path.

## Usage
### Encoding
```bash
  $ python codec.py --mode enc --save-path {path} --input-file {input image file} --cuda
```
"--cuda" is optional.

For exasmple, command below
```bash
  $ python codec.py --mode enc --save-path results --input-file sample.png --cuda
```
generates binary files in "results/bits".

### Decoding
```bash
  $ python codec.py --mode dec --save-path {path same with enc} --input-file {original image file} --recon-level {int} --cuda
```
"--input-file" is optional, used to calculate PSNR.

For example, command below
```bash
  $ python codec.py --mode dec --save-path results --input-file sample.png --recon-level 140 --cuda
```
prints metrics and saves reconstructed an image "results/recon/q0140.png".

## Citation
Please cite the following paper when you use this repository. Thanks!
```bibtex
    @inproceedings{2023_CVPR_jeon,
        author    = {Jeon, Seungmin and Choi, Kwang Pyo and Park, Youngo and Kim, Chang-Su}, 
        title     = {Context-Based Trit-Plane Coding for Progressive Image Compression}, 
        booktitle = {{Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition}},
        year      = {2023}
    }
```

### License
See [MIT License](https://github.com/seungminjeon-github/CTC/blob/master/LICENSE)
