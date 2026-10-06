"""Fine-tune a timm Swin Transformer on a plant-leaf disease dataset.

Single GPU / CPU:
    python train.py --data_dir data/my_leaves --epochs 50 --output_dir runs/exp1

Multi-GPU (DDP), e.g. 4 GPUs on one machine - for bigger datasets:
    torchrun --nproc_per_node=4 train.py --data_dir data/my_leaves --epochs 50 \
        --output_dir runs/exp1 --batch_size 32   # 32 per GPU = 128 effective

Resume an interrupted run:
    python train.py --data_dir data/my_leaves --output_dir runs/exp1 --resume

Training recipe (AdamW, cosine+warmup, gradient clip, label smoothing,
Mixup/CutMix, per-size stochastic depth, optional EMA) follows:
  Liu et al., "Swin Transformer: Hierarchical Vision Transformer using
  Shifted Windows", ICCV 2021.      https://arxiv.org/abs/2103.14030
  Official Swin-Transformer config defaults (Microsoft):
      https://github.com/microsoft/Swin-Transformer
  timm "sw" (Swin) recipe tag:      https://huggingface.co/docs/timm/hparams
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from plantdisease.data import build_datasets, build_transforms, make_loaders
from plantdisease.engine import (build_ema, build_mixup, measure_latency,
                                 predict_loader, train_one_epoch)
from plantdisease.metrics import compute_metrics, save_evaluation
from plantdisease.model import (build_layer_decay_groups, create_model, data_config,
                                save_checkpoint, set_backbone_frozen)
from plantdisease.plots import plot_history
from plantdisease.scheduler import CosineWarmupScheduler
from plantdisease.utils import (cleanup_ddp, get_device, is_main_process, set_seed,
                                setup_ddp)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_dir", default="runs/exp1")
    ap.add_argument("--model", default="swin_tiny_patch4_window7_224",
                    help="any timm swin_*/swinv2_* model; also accepts other timm models")
    ap.add_argument("--no_pretrained", action="store_true", help="random init (offline / tests)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32, help="per-process (per-GPU) batch size")
    ap.add_argument("--accum_steps", type=int, default=1,
                    help="gradient accumulation; effective batch = batch_size * accum_steps * world_size")
    ap.add_argument("--lr", type=float, default=5e-4, help="base LR (timm sw recipe default range)")
    ap.add_argument("--layer_decay", type=float, default=1.0,
                    help="layer-wise LR decay across Swin stages; 1.0 = off (uniform LR, timm sw default). "
                         "0.7-0.9 matches MixMAE/FastMIM fine-tuning recipes for larger Swin variants.")
    ap.add_argument("--weight_decay", type=float, default=0.05, help="Swin paper / timm sw default")
    ap.add_argument("--drop_path_rate", type=float, default=None,
                    help="default: official per-size value (tiny=0.2, small=0.3, base=0.5)")
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--mixup", type=float, default=0.0, help="Swin paper default is 0.8; off by default "
                    "for small/fine-grained fine-tuning datasets (see README)")
    ap.add_argument("--cutmix", type=float, default=0.0, help="Swin paper default is 1.0")
    ap.add_argument("--grad_clip", type=float, default=5.0, help="official Swin config CLIP_GRAD")
    ap.add_argument("--ema", action="store_true", help="keep an EMA of the weights (timm sw recipe)")
    ap.add_argument("--ema_decay", type=float, default=0.9998)
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--img_size", type=int, default=None, help="default: model native size (224)")
    ap.add_argument("--freeze_epochs", type=int, default=2,
                    help="epochs training only the classifier head before unfreezing")
    ap.add_argument("--patience", type=int, default=10, help="early-stopping patience (val macro-F1)")
    ap.add_argument("--class_weights", action="store_true", help="weight loss for imbalanced data")
    ap.add_argument("--val_size", type=float, default=0.15, help="used if no val folder exists")
    ap.add_argument("--test_size", type=float, default=0.15, help="used for single-folder data")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="resume from --output_dir/last_model.pth")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rank, local_rank, world_size = setup_ddp()
    distributed = world_size > 1
    set_seed(args.seed, rank)
    device = get_device(local_rank)
    out = Path(args.output_dir)
    if is_main_process():
        out.mkdir(parents=True, exist_ok=True)
        (out / "args.json").write_text(json.dumps(vars(args), indent=2))
        print(f"Device: {device} | world_size: {world_size}")

    # ---- data ----------------------------------------------------------------------
    bundle = build_datasets(args.data_dir, args.val_size, args.test_size, args.seed)
    class_names = bundle.class_names
    num_classes = len(class_names)
    if is_main_process():
        print(f"Classes ({num_classes}): {class_names}")
        print(f"Train/Val/Test: {len(bundle.train)}/{len(bundle.val)}/{len(bundle.test)}")
        (out / "class_names.json").write_text(json.dumps(class_names, indent=2))

    # ---- model ---------------------------------------------------------------------
    model = create_model(args.model, num_classes, pretrained=not args.no_pretrained,
                         drop_path_rate=args.drop_path_rate).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if is_main_process():
        print(f"Model: {args.model} | parameters: {n_params/1e6:.2f} M")

    cfg = data_config(model, args.img_size)
    train_tf, eval_tf = build_transforms(cfg)
    bundle.set_transforms(train_tf, eval_tf)
    train_loader, val_loader, test_loader, train_sampler = make_loaders(
        bundle, args.batch_size, args.workers, device.type == "cuda",
        distributed=distributed, world_size=world_size, rank=rank)

    mixup_fn, soft_criterion = build_mixup(args.mixup, args.cutmix, args.label_smoothing, num_classes)

    raw_model = model  # unwrapped reference, used for freezing/EMA/saving even under DDP
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if device.type == "cuda" else None)

    ema = build_ema(raw_model, decay=args.ema_decay) if args.ema else None

    # ---- optimisation --------------------------------------------------------------
    weight = None
    if args.class_weights:
        counts = np.bincount(bundle.train_labels, minlength=num_classes)
        w = counts.sum() / (num_classes * np.maximum(counts, 1))
        weight = torch.tensor(w, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)

    if args.layer_decay < 1.0:
        param_groups = build_layer_decay_groups(raw_model, args.lr, args.weight_decay, args.layer_decay)
        optimizer = torch.optim.AdamW(param_groups)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineWarmupScheduler(optimizer, args.warmup_epochs, args.epochs, len(train_loader))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "val_f1"]}
    best_f1, bad_epochs, start_epoch = -1.0, 0, 1
    best_path, last_path = out / "best_model.pth", out / "last_model.pth"

    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        raw_model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["extra"]["optimizer"])
        start_epoch = ckpt["extra"]["epoch"] + 1
        best_f1 = ckpt["extra"]["best_f1"]
        hist = ckpt["extra"]["hist"]
        if ema is not None and "ema_state_dict" in ckpt:
            ema.module.load_state_dict(ckpt["ema_state_dict"])
        if is_main_process():
            print(f"Resumed from epoch {start_epoch - 1}, best F1 so far {best_f1:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        frozen = epoch <= args.freeze_epochs
        set_backbone_frozen(raw_model, frozen)
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp,
            mixup_fn=mixup_fn, soft_criterion=soft_criterion, ema=ema,
            grad_clip=args.grad_clip, accum_steps=args.accum_steps,
            sampler=train_sampler, epoch=epoch)
        for _ in range(len(train_loader) // max(1, args.accum_steps)):
            scheduler.step()

        if is_main_process():
            eval_model = ema.module if ema is not None else raw_model
            probs, labels, val_loss = predict_loader(eval_model, val_loader, device, criterion)
            vm = compute_metrics(labels, probs, num_classes)
            for k, v in zip(hist, [tr_loss, tr_acc, val_loss, vm["accuracy"], vm["f1_macro"]]):
                hist[k].append(v)
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:03d}/{args.epochs} | lr {lr_now:.2e} | train loss {tr_loss:.4f} "
                  f"acc {tr_acc:.4f} | val loss {val_loss:.4f} acc {vm['accuracy']:.4f} "
                  f"F1 {vm['f1_macro']:.4f}{'  [head only]' if frozen else ''}")

            save_checkpoint(last_path, raw_model, args.model, class_names, cfg,
                            extra={"epoch": epoch, "best_f1": max(best_f1, vm["f1_macro"]),
                                   "hist": hist, "optimizer": optimizer.state_dict()}, ema=ema)
            if vm["f1_macro"] > best_f1:
                best_f1, bad_epochs = vm["f1_macro"], 0
                save_checkpoint(best_path, raw_model, args.model, class_names, cfg,
                                extra={"epoch": epoch, "val_f1_macro": best_f1}, ema=ema)
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print(f"Early stopping: no val macro-F1 improvement for {args.patience} epochs.")
                    break

    if is_main_process():
        pd.DataFrame(hist).to_csv(out / "training_history.csv", index=False)
        plot_history(hist, out / "training_curves.png")

        eval_model, ckpt = raw_model, torch.load(best_path, map_location=device)
        state = ckpt.get("ema_state_dict", ckpt["state_dict"])
        eval_model.load_state_dict(state)
        probs, labels, _ = predict_loader(eval_model, test_loader, device)
        extra = {
            "best_epoch": int(ckpt["extra"]["epoch"]),
            "params_millions": n_params / 1e6,
            "model_size_mb": best_path.stat().st_size / 1e6,
            "world_size": world_size,
            "latency_ms_per_image_cpu": measure_latency(eval_model, torch.device("cpu"),
                                                         cfg["img_size"], runs=30),
        }
        if device.type == "cuda":
            extra["latency_ms_per_image_gpu"] = measure_latency(eval_model, device, cfg["img_size"])
        save_evaluation(out, labels, probs, class_names, tag="test",
                        paths=[p for p, _ in bundle.test.samples], extra=extra)
        print(f"All outputs saved in: {out.resolve()}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
