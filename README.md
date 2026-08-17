# HyGenQ

HyGenQ is a post-training quantization implementation for hybrid iterative generative models. It targets MAR image generation, whose inference path combines autoregressive token refinement with diffusion-based token prediction. HyGenQ applies W8A8 quantization to the linear and matrix-multiplication operators used by both stages.

## Installation

```bash
conda env create -f environment.yaml
conda activate hygenq
```

## Local Prerequisites

The default command expects the following relative path:

- `pretrained_models/mar/mar_large/checkpoint-last.pth`

Set a different MAR checkpoint directory with `--resume`.

## Run Quantization

`run.sh` provides the default MAR-Large W8A8 calibration configuration:

```bash
bash run.sh
```

## Citation

```bibtex
@article{gao2026post,
  title={Post-training Quantization for Hybrid Iterative Generative Models},
  author={Gao, Jing and Wu, Junyi and Wang, Wei and Yan, Yan and Zhao, Yao},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2026},
  publisher={IEEE}
}
```

## Acknowledgments and License

HyGenQ builds on the [MAR](https://github.com/LTH14/mar) implementation by Tianhong Li et al. Its quantization design also draws on the following work:

- [PTQ4ViT: Post-training Quantization for Vision Transformers](https://arxiv.org/abs/2111.12293)
- [PTQ4DiT: Post-training Quantization for Diffusion Transformers](https://arxiv.org/abs/2405.16005)

The original MAR code remains copyright its respective authors and is released under the MIT License. HyGenQ and its modifications are also released under the MIT License; see [LICENSE](LICENSE).

When using the underlying MAR model or implementation, please also cite:

```bibtex
@article{li2024autoregressive,
  title={Autoregressive Image Generation without Vector Quantization},
  author={Li, Tianhong and Tian, Yonglong and Li, He and Deng, Mingyang and He, Kaiming},
  journal={arXiv preprint arXiv:2406.11838},
  year={2024}
}
```
