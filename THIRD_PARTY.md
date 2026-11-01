# Third-party sources

## CoMoE

- Repository: [UESTC-nnLab/CoMoE](https://github.com/UESTC-nnLab/CoMoE)
- Pinned commit: [`bef403e0ee2db33f98675dee080eccc68986ccc8`](https://github.com/UESTC-nnLab/CoMoE/tree/bef403e0ee2db33f98675dee080eccc68986ccc8)
- Paper: W. Duan, L. Ji, J. Huang, S. Zhu and M. Ye, “Cross-domain Joint Learning with Prototype-guided Mixture-of-Experts for Infrared Moving Small Target Detection,” *AAAI*, 2026.

KFGMoE uses the CoMoE encoder, prototype processing, expert transformations, temporal refinement, detection loss and supporting utilities. Its keyframe spatial-supervision and gate-correction modules are implemented in `kfgmoe`.

`tools/prepare_upstream.py` downloads the dependency to `third_party/CoMoE` and verifies the files listed in [configs/kfgmoe.json](configs/kfgmoe.json). To use an existing local checkout, run:

```bash
python tools/prepare_upstream.py --source /path/to/CoMoE
```

The pinned CoMoE checkout does not contain a `LICENSE` file. This repository does not assign a license to CoMoE code or pretrained weights; their rights remain with the original authors.

## Python dependencies

Python dependencies are installed separately under their respective licenses. Versions are listed in [requirements.txt](requirements.txt) and the [README](README.md#prerequisite).

## Datasets and weights

Datasets are obtained from the providers linked by the [CoMoE benchmark](https://github.com/UESTC-nnLab/CoMoE#datasets-bounding-box-based), under their usage terms. Datasets, model weights and the CoMoE source are not bundled with this repository.
