# Plant Leaf Disease Classification — Swin Transformer (timm)

Fine-tunes a pretrained **Swin Transformer** ([`timm`](https://github.com/huggingface/pytorch-image-models))
on your own leaf-image dataset. The training recipe, stochastic-depth rates, and
augmentation choices are not invented defaults — they're copied from the papers,
official configs, and open repositories listed in **[References](#references)** below,
each tied to the specific line of code it justifies.

Built to generalize beyond a single lab dataset: multi-GPU (DDP) training,
gradient accumulation, checkpoint resume, and EMA weights are included for
scaling to larger or field-collected datasets, not just PlantVillage-sized ones.

## Repository structure

```
.
├── train.py                 # fine-tune + evaluate (single-GPU, multi-GPU via torchrun, or CPU)
├── evaluate.py               # evaluate an existing checkpoint on any folder
├── predict.py                 # predict on one image or a folder of images
├── gradcam.py                # Grad-CAM heat-map (adapted for Swin's NHWC feature maps)
├── plantdisease/
│   ├── data.py                  # dataset loading, splitting, transforms, DistributedSampler
│   ├── model.py                  # Swin creation, per-size stochastic depth, layer-wise LR decay
│   ├── engine.py                  # train/eval loops: Mixup/CutMix, EMA, grad clip, accumulation
│   ├── scheduler.py                # cosine LR with linear warmup
│   ├── metrics.py                   # accuracy/F1/kappa/MCC/ROC-AUC, reports, confusion matrix
│   ├── plots.py                      # training curves, confusion-matrix plots
│   └── utils.py                       # seeding, device, DDP setup/teardown
├── scripts/split_dataset.py    # group-aware stratified train/val/test split
├── tests/test_smoke.py          # end-to-end tests (verified to pass — see below)
├── requirements.txt / requirements-dev.txt / LICENSE
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

GPU is strongly recommended. Swin-Tiny (28M params) is manageable on a single
modern GPU; Swin-Base/Large or large datasets benefit from multi-GPU (below).

## 1. Prepare your dataset

Same two layouts as before — pre-split `train/`, `val/`(or `valid`/`validation`),
`test/`(or `testing`) folders, or a single folder with one subfolder per class
(auto-split 70/15/15). If multiple photos share a leaf or plant, split by group
first so duplicates don't leak across splits:

```bash
python scripts/split_dataset.py --src raw_data --dst data/split --group_regex "^(leaf\d+)_"
```

## 2. Train

**Single GPU / CPU:**
```bash
python train.py --data_dir data --output_dir runs/exp1 --epochs 50
```

**Multi-GPU (DDP) — for bigger datasets:**
```bash
torchrun --nproc_per_node=4 train.py --data_dir data --output_dir runs/exp1 \
    --epochs 50 --batch_size 32     # 32/GPU × 4 GPUs = 128 effective batch
```

**Resume an interrupted run** (checkpoints the optimizer state and history):
```bash
python train.py --data_dir data --output_dir runs/exp1 --resume
```

Key flags and where each comes from:

| Flag | Default | Source |
|---|---|---|
| `--model` | `swin_tiny_patch4_window7_224` | any timm `swin_*`/`swinv2_*` |
| `--lr` | 5e-4 | timm "sw" recipe range ([ref 3](#references)) |
| `--weight_decay` | 0.05 | Swin paper / official config ([ref 1](#references), [ref 2](#references)) |
| `--drop_path_rate` | per-size (0.2/0.3/0.5 for T/S/B) | official config ([ref 2](#references)) |
| `--label_smoothing` | 0.1 | Swin paper ([ref 1](#references)) |
| `--mixup`, `--cutmix` | 0 (off) | Swin paper default is 0.8 / 1.0 ([ref 1](#references), [ref 2](#references)) — off here by default; see [note](#why-mixupcutmix-defaults-to-off) |
| `--grad_clip` | 5.0 | official config `CLIP_GRAD` ([ref 2](#references)) |
| `--ema` / `--ema_decay` | off / 0.9998 | timm "sw" recipe ([ref 3](#references)) |
| `--warmup_epochs` | 5 | Swin paper cosine+warmup schedule ([ref 1](#references)) |
| `--layer_decay` | 1.0 (off) | 0.7–0.9 matches MixMAE/FastMIM Swin-B/L fine-tuning ([ref 9](#references), [ref 10](#references)) |
| `--accum_steps` | 1 | gradient accumulation for a larger effective batch without more VRAM |
| `--freeze_epochs` | 2 | epochs training only the classifier head before unfreezing |
| `--class_weights` | off | for imbalanced datasets |

Outputs in `--output_dir`: `best_model.pth`, `last_model.pth` (for `--resume`),
`training_history.csv`/`training_curves.png`, `test_metrics.json`,
`test_classification_report.txt`, confusion matrices (raw/normalized/csv),
`test_per_class_metrics.csv`, `test_predictions.csv`, `class_names.json`, `args.json`.

Reported test metrics: accuracy, precision/recall/F1 (macro & weighted), Cohen's
kappa, MCC, one-vs-rest macro ROC-AUC, parameter count, checkpoint size, and
CPU/GPU inference latency per image.

## 3. Evaluate on another folder

```bash
python evaluate.py --checkpoint runs/exp1/best_model.pth --test_dir data/external_test
```
Add `--no_ema` to evaluate the raw (non-averaged) weights instead of the EMA copy,
if `--ema` was used during training.

## 4. Predict

```bash
python predict.py --checkpoint runs/exp1/best_model.pth --input leaf.jpg
python predict.py --checkpoint runs/exp1/best_model.pth --input new_leaves/ --csv preds.csv
```

## 5. Explain a prediction (Grad-CAM)

```bash
python gradcam.py --checkpoint runs/exp1/best_model.pth --image leaf.jpg --output cam.png
```
Swin's internal feature maps are `(B, H, W, C)` rather than a CNN's `(B, C, H, W)`
— confirmed directly against timm's `swin_tiny_patch4_window7_224` output shape
during development (`(2, 7, 7, 768)` for a 224px input at `layers[-1]`) — so
`gradcam.py` converts the layout automatically before computing the CAM.

## Why Mixup/CutMix defaults to off

The Swin paper's Mixup=0.8/CutMix=1.0 ([ref 1](#references)) was tuned for
1000-way ImageNet classification with visually distinct object categories.
Plant-disease classes are often fine-grained and visually close (e.g. several
*Tomato* diseases differ only in subtle lesion texture); blending two such
images can blur exactly the signal the model needs to learn. Pass `--mixup 0.8
--cutmix 1.0` to use the original recipe — both tested and working here (see
`tests/test_smoke.py::test_train_with_mixup_and_resume`) — but validate on your
own data before trusting it over the default.

## What's actually been verified

Everything below was run in this environment with real `torch`/`timm` installed,
not just written and assumed to work:

- `pytest tests/test_smoke.py` — **passes** (2/2): trains Swin-Tiny end-to-end on
  tiny synthetic data, exercises `--ema`, confirms every output file is written
  correctly, and separately verifies the `--mixup`/`--cutmix` + `--resume` path.
- `gradcam.py` — run end-to-end on a trained checkpoint; the `(B,H,W,C)` →
  `(B,C,H,W)` conversion was checked directly against timm's actual output shape
  before being written into the heuristic.
- `--layer_decay 0.8` — confirmed the resulting parameter groups carry
  monotonically increasing learning rates from the earliest Swin stage toward
  the classification head, as the layer-decay fine-tuning strategy intends.

Multi-GPU (`torchrun`) training was *not* tested end-to-end (no multi-GPU
hardware available here) — the DDP wiring follows the standard PyTorch pattern
([ref 11](#references)) but test it on your own hardware before a long run.

## Run the tests yourself

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Generalizing to a bigger dataset

- **Multi-GPU**: `torchrun --nproc_per_node=N train.py ...` shards the training
  set across GPUs with a `DistributedSampler`; validation/test still run on a
  single process to keep metric computation simple and exact.
- **Gradient accumulation** (`--accum_steps`): raise the effective batch size
  without more VRAM.
- **Resume** (`--resume`): safely continue an interrupted multi-day run from
  `last_model.pth`, including optimizer state and history.
- **EMA** (`--ema`): stabilizes evaluation on noisier, more diverse (e.g.
  field-collected) data — part of timm's own Swin recipe ([ref 3](#references)).
- For datasets too large to fit on local disk, swap `plantdisease/data.py`'s
  `ImageFolder`-based loading for a streaming format (e.g. WebDataset) — the
  rest of the pipeline (transforms, training loop, metrics) is independent of
  how samples are loaded.

## References

1. Liu, Z. et al. *"Swin Transformer: Hierarchical Vision Transformer using
   Shifted Windows."* ICCV 2021. https://arxiv.org/abs/2103.14030 — architecture;
   fine-tuning recipe (AdamW, cosine+warmup, label smoothing 0.1, Mixup 0.8/CutMix 1.0).
2. Official Swin-Transformer config (Microsoft): https://github.com/microsoft/Swin-Transformer
   (mirror used during development: https://openi.pcl.ac.cn/Goodman2023/SwinTransformer/src/branch/master/config.py)
   — exact per-size `DROP_PATH_RATE`, `CLIP_GRAD=5.0`, AdamW betas, weight decay.
3. timm's official **"sw" (Swin) training recipe tag**:
   https://huggingface.co/docs/timm/hparams — "AdamW with gradient clipping,
   EMA | Cosine with warmup."
4. timm model card, `swin_tiny_patch4_window7_224`:
   https://huggingface.co/timm/swin_tiny_patch4_window7_224.ms_in22k_ft_in1k —
   correct `timm.create_model` / data-config usage.
5. *"Domain-Specific Self-Supervised Pre-training for Agricultural Disease
   Classification: A Hierarchical Vision Transformer Study."*
   https://arxiv.org/pdf/2601.11612 — Swin-style HVT evaluated on PlantVillage
   (96.3%), PlantDoc (87.1%), Cotton Leaf Disease (90.24%); informed the choice
   to keep Mixup/CutMix optional rather than on-by-default for fine-grained
   disease classes.
6. *"Explainable AI-based two-stage Swin transformer for grapevine leaf-based
   variety and disease classification."* Politecnico di Torino.
   https://iris.polito.it/retrieve/handle/11583/3013827/1002699 — Swin-Tiny
   backbone + Grad-CAM-style explainability for disease diagnosis; the
   `gradcam.py` approach here follows the same idea applied to Swin's last
   stage.
7. PlantDoc-Predictor model zoo — Swin-Tiny/Swin-Base fine-tuned on the
   PlantVillage 38-class dataset, 99.1% accuracy.
   https://pypi.org/project/plantdoc-predictor/
8. HaritaX — hybrid VGG16/ResNet50/ViT/Swin ensemble for plant disease
   detection (Swin component: 98.80% on a 15-class subset).
   https://github.com/aayush010904/HaritaX
9. Liu, X. et al. *"MixMAE: Mixed and Masked Autoencoder for Efficient
   Pretraining of Hierarchical Vision Transformers."* CVPR 2023.
   https://arxiv.org/pdf/2205.13137 — layer-wise LR decay (0.85/0.9) for
   fine-tuning Swin-B/L.
10. *"FastMIM: Expediting Masked Image Modeling Pre-training for Vision."*
    https://arxiv.org/pdf/2212.06593 — grid-searched layer-wise decay ratios
    for fine-tuning Swin-B/L.
11. PyTorch DDP tutorial: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
    — standard `torchrun` / `DistributedDataParallel` pattern used in `train.py`.

## A note carried over from the PlantVillage/InceptionV3 discussion

The same caution applies here: published 99%+ numbers (including the Swin
figures in refs 5–8 above) are usually on PlantVillage-style lab datasets, and
can be inflated by non-grouped splits where near-duplicate images of the same
leaf land in both train and test. Use `scripts/split_dataset.py --group_regex`
and evaluate on an independently collected test set before trusting a number
like that for your own paper.
