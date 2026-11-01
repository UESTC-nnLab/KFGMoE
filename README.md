# KFGMoE

**Keyframe-Guided Mixture of Experts for Multi-Domain Moving Infrared Small Target Detection**

Jinwu Lyu, Wenhao Li and Luping Ji

PyTorch implementation accompanying our manuscript, currently under review at *IEEE Transactions on Geoscience and Remote Sensing*.

![KFGMoE architecture](assets/architecture.png)

## Introduction

KFGMoE is a unified multi-domain detector that uses spatial evidence from the current keyframe to guide expert feature construction. The framework connects three stages: **prototype-guided keyframe enhancement**, **MoE-based feature construction** and **spatio-temporal feature refinement**. Bounding-box supervision teaches the enhanced keyframe representation to identify detection-relevant regions. A centered spatial descriptor then corrects the keyframe gate before expert selection and mixing. Historical-frame features are constructed with the enhanced keyframe as a reference and combined for current-frame detection.

The default model processes four historical frames and the current keyframe, with six shared experts and Top-2 selection. It jointly handles DAUB-H, IRDST-R and ITSDT-15K using one model.

## Prerequisite

- Python 3.10
- PyTorch 2.9 and the matching torchvision release
- NumPy, Pillow and pycocotools; see [requirements.txt](requirements.txt)

Install PyTorch and torchvision for your CUDA environment, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Prepare the pinned CoMoE dependency:

```bash
python tools/prepare_upstream.py
```

If you already have the matching CoMoE source locally:

```bash
python tools/prepare_upstream.py --source /path/to/CoMoE
```

The dependency is pinned to commit `bef403e0ee2db33f98675dee080eccc68986ccc8`. Source attribution and dependency details are in [THIRD_PARTY.md](THIRD_PARTY.md).

## Datasets

We use the bounding-box versions of **DAUB-H**, **IRDST-R** and **ITSDT-15K** provided by the [CoMoE benchmark](https://github.com/UESTC-nnLab/CoMoE#datasets-bounding-box-based). Download the datasets from their original providers and keep each video's frame sequence intact.

Annotation lists use JSON Lines, with one keyframe per line:

```json
{"path":"DAUB-H/train/data1/12.bmp","domain":1,"boxes":[[100,80,106,86,0]]}
{"path":"DAUB-H/train/data1/13.bmp","domain":1,"boxes":[]}
```

`path` is relative to `--data-root`. Each box is `[x1, y1, x2, y2, class_id]` in original-image pixels; the single target class has ID `0`. An empty list represents an image without a target. Domain IDs are fixed as follows:

| Dataset | Domain ID |
|---|---:|
| DAUB-H | 1 |
| IRDST-R | 2 |
| ITSDT-15K | 3 |

Frames within each video have consecutive numeric BMP filenames starting at `0.bmp`. The loader uses the current frame and its four preceding frames; at the start of a video, it repeats the first frame.

Frame indices must match the benchmark annotations. If the original files start at `001.bmp`, prepare a separate zero-based copy with the corresponding benchmark labels; keep the original dataset unchanged.

```text
datasets/
├── DAUB-H/train/data1/0.bmp
├── DAUB-H/train/data1/1.bmp
├── IRDST-R/train/data1/0.bmp
└── ITSDT-15K/train/1/0.bmp
annotations/
├── train.jsonl
└── test.jsonl
```

Convert an existing CoMoE-style annotation file (`image_path x1,y1,x2,y2,0 ...`) with:

```bash
python tools/prepare_annotations.py --input original_train.txt --output annotations/train.jsonl --data-root /path/to/datasets
```

For annotation files created on another machine, add `--strip-prefix /old/dataset/root`. For a single dataset whose path does not contain its name, specify its ID with `--domain`.

## Usage

### Train

Download the [initialization weights provided by CoMoE](https://github.com/UESTC-nnLab/CoMoE/blob/bef403e0ee2db33f98675dee080eccc68986ccc8/model_data/pretrain.pth), save them as `weights/pretrained.pth` and provide the training annotation list:

```bash
python train.py --config configs/kfgmoe.json --train-list annotations/train.jsonl --data-root /path/to/datasets --initial-weights weights/pretrained.pth --output outputs/kfgmoe
```

The default recipe uses five frames, six experts, Top-2 selection, a gate-correction bound of `0.5` and a spatial-supervision weight of `0.1`. Training uses FP32, batch size 4 and seed 3047. Settings are provided in [configs/kfgmoe.json](configs/kfgmoe.json).

Check the data and configuration before training with `--dry-run`. Resume an interrupted run with:

```bash
python train.py --config configs/kfgmoe.json --train-list annotations/train.jsonl --data-root /path/to/datasets --resume outputs/kfgmoe/latest.pt --output outputs/kfgmoe
```

### Test

The test entry point evaluates bounding-box detections for each domain and reports their equally weighted macro averages:

```bash
python test.py --config configs/kfgmoe.json --weights weights/kfgmoe_epoch060_ema.pth --annotations annotations/test.jsonl --data-root /path/to/datasets --output runs/test --device cuda
```

### Visualization

The prediction entry point uses a causal five-frame window and produces detections for the current frame:

```bash
python predict.py --weights weights/kfgmoe_epoch060_ema.pth --sequence /path/to/video/frames --domain DAUB-H --output runs/demo --device cuda
```

![KFGMoE detection examples](assets/detection_examples.png)

Examples from the released checkpoint on DAUB-H, IRDST-R and ITSDT-15K. Cyan boxes indicate ground truth and orange boxes indicate predictions with confidence at least 0.5. The lower row enlarges the first annotated target in each image. Sequence names and frame indices are shown in the figure.

### Smoke check

```bash
python smoke.py --weights weights/kfgmoe_epoch060_ema.pth --device cpu
```

Omit `--weights` to check the model with synthetic input before downloading the checkpoint.

## Results

Results for the epoch-60 EMA on the complete three-domain test set:

| Dataset | mAP50 (%) | Pr (%) | Rc (%) | F1 (%) | mAP75 (%) | mAP50:95 (%) | Fa ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| DAUB-H | 53.7530 | 70.4940 | 77.6877 | 73.9162 | 38.6289 | 33.8694 | 0.151322 |
| IRDST-R | 74.2099 | 83.2310 | 90.6922 | 86.8015 | 0.0126 | 14.4536 | 0.064314 |
| ITSDT-15K | 84.0296 | 90.2328 | 94.1169 | 92.1339 | 11.6903 | 31.3551 | 0.301000 |
| **Macro average** | **70.6642** | **81.3193** | **87.4989** | **84.2839** | **16.7773** | **26.5594** | **0.172212** |

Pr, Rc and F1 follow the benchmark's PR-curve summary at IoU 0.5: Rc is maximum recall, Pr averages the interpolated precision over that recall range and F1 is their harmonic mean. Fa is false detections per frame at confidence 0.5 and matching IoU 0.5. Each macro metric is the arithmetic mean of its three domain values. Evaluation uses 512 × 512 stretch resizing, score floor 0.001, NMS IoU 0.65 and at most 100 detections per image.

### Model weights

Download the [trained KFGMoE checkpoint](https://github.com/UESTC-nnLab/KFGMoE/releases/download/v1.0.0/kfgmoe_epoch060_ema.pth) and save it as `weights/kfgmoe_epoch060_ema.pth` (86,857,987 bytes), with SHA256:

```text
f54c475b6d9644a782f96d00bef9fa219b3840855cb3dc8f47c18d82442cc2e8
```

## Acknowledgements

Our implementation builds on [CoMoE](https://github.com/UESTC-nnLab/CoMoE) for the shared encoder, prototype processing, expert transformations and temporal refinement. We thank its authors and the dataset providers for making their work available.

## Contact

For questions, please contact Jinwu Lyu at [jinwulyu@outlook.com](mailto:jinwulyu@outlook.com).

## Citation

Citation metadata will be added when the manuscript becomes publicly available.
