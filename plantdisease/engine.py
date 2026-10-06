"""Training / inference loops.

Implements the timm "sw" (Swin) recipe building blocks - AdamW (built by the
caller) + gradient clipping + EMA, cosine LR with warmup (scheduler.py) - and
the official Swin-Transformer Mixup/CutMix augmentation:
  timm hparams "sw" tag: https://huggingface.co/docs/timm/hparams
  Swin paper (Mixup=0.8, CutMix=1.0, label smoothing=0.1), ICCV 2021 A1:
  https://arxiv.org/abs/2103.14030
Mixup/CutMix implementation itself is timm's `timm.data.Mixup`
(https://github.com/huggingface/pytorch-image-models), which turns hard
labels into soft targets - hence the SoftTargetCrossEntropy loss used below
only in that case (matches timm's own training script behaviour).
"""
import copy
import time

import torch
import torch.nn as nn
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV2
from tqdm import tqdm

from .utils import is_main_process, reduce_mean


def build_mixup(mixup_alpha, cutmix_alpha, label_smoothing, num_classes):
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return None, None
    from timm.data import Mixup
    mixup_fn = Mixup(mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha,
                     prob=1.0, switch_prob=0.5, mode="batch",
                     label_smoothing=label_smoothing, num_classes=num_classes)
    return mixup_fn, SoftTargetCrossEntropy()


def build_ema(model, decay=0.9998):
    return ModelEmaV2(model, decay=decay)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp,
                    mixup_fn=None, soft_criterion=None, ema=None,
                    grad_clip=5.0, accum_steps=1, sampler=None, epoch=0):
    """`grad_clip=5.0` matches the official Swin-Transformer config's CLIP_GRAD.
    `accum_steps>1` simulates a larger effective batch size without more VRAM -
    useful for scaling to bigger datasets on limited hardware.
    """
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)  # required for correct DDP shuffling each epoch
    model.train()
    loss_sum, correct, seen = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    for step, (x, y) in enumerate(tqdm(loader, desc="train", leave=False, disable=not is_main_process())):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        y_hard = y
        if mixup_fn is not None and x.size(0) % 2 == 0:
            x, y = mixup_fn(x, y)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss_fn = soft_criterion if (mixup_fn is not None and x.size(0) % 2 == 0) else criterion
            loss = loss_fn(logits, y) / accum_steps
        scaler.scale(loss).backward()
        if (step + 1) % accum_steps == 0:
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
        loss_sum += loss.item() * accum_steps * x.size(0)
        correct += (logits.argmax(1) == y_hard).sum().item()
        seen += x.size(0)
    device_for_reduce = device
    return (reduce_mean(loss_sum / seen, device_for_reduce),
            reduce_mean(correct / seen, device_for_reduce))


@torch.no_grad()
def predict_loader(model, loader, device, criterion=None):
    """Returns (probabilities [N,C], labels [N], mean loss or None).
    Intentionally single-process (see data.make_loaders) - run only on rank 0.
    """
    model.eval()
    probs, labels, loss_sum, n = [], [], 0.0, 0
    for x, y in tqdm(loader, desc="eval", leave=False, disable=not is_main_process()):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        if criterion is not None:
            loss_sum += criterion(logits, y).item() * x.size(0)
            n += x.size(0)
        probs.append(torch.softmax(logits.float(), dim=1).cpu())
        labels.append(y.cpu())
    return (torch.cat(probs).numpy(), torch.cat(labels).numpy(),
            loss_sum / n if n else None)


@torch.no_grad()
def measure_latency(model, device, img_size, runs=50, warmup=10):
    model = copy.deepcopy(model).to(device).eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(runs):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / runs * 1000
