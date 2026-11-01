# Third-party sources

## CoMoE

- Repository: [UESTC-nnLab/CoMoE](https://github.com/UESTC-nnLab/CoMoE)
- Pinned commit: [`bef403e0ee2db33f98675dee080eccc68986ccc8`](https://github.com/UESTC-nnLab/CoMoE/tree/bef403e0ee2db33f98675dee080eccc68986ccc8)
- Paper: W. Duan, L. Ji, J. Huang, S. Zhu and M. Ye, “Cross-domain Joint Learning with Prototype-guided Mixture-of-Experts for Infrared Moving Small Target Detection,” *AAAI*, 2026.

KFGMoE uses the CoMoE encoder, prototype processing, expert transformations, temporal refinement, detection loss and supporting data/preprocessing routines. Its keyframe spatial-supervision and gate-correction modules are implemented in the `kfgmoe` package. The epoch-60 checkpoint retains the upstream parameter names through the `detector` wrapper.

`tools/prepare_upstream.py` prepares the pinned dependency under `third_party/CoMoE`. The CoMoE source is an external dependency and is excluded from the KFGMoE source archive. The preparation tool can use an existing local source tree or retrieve the pinned source from GitHub.

The dependency files are:

| Files | Role |
|---|---|
| `nets/darknet.py`, `nets/slowfastnet.py` | Shared encoder and detector |
| `nets/mix_fix1.py` | Prototype processing, shared experts and temporal refinement |
| `nets/yolo_training.py` | Detection loss, initialization, optimizer scheduling and EMA utilities |
| `utils/utils.py` | Image and training utilities |
| `nets/__init__.py`, `utils/__init__.py` | Package initialization |

These files are checked against the SHA256 values in `configs/kfgmoe.json` and are not rewritten by the preparation tool.

The pinned CoMoE checkout does not contain a `LICENSE` file. This repository does not assign a license to that code or its pretrained weights. Their rights remain with the original authors; contact them concerning redistribution permissions.

## Python dependencies

PyTorch, torchvision, NumPy, Pillow and pycocotools are installed separately through their distributions. Their respective licenses and notices apply. See [requirements.txt](requirements.txt) for the package versions used by this release.

## Datasets and weights

DAUB-H, IRDST-R and ITSDT-15K are obtained from the providers linked by the [CoMoE benchmark](https://github.com/UESTC-nnLab/CoMoE#datasets-bounding-box-based). Dataset files and model weights are not bundled with the source archive. Refer to the original providers for their usage terms.
