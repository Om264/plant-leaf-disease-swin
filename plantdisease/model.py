"""Swin Transformer creation, checkpoint I/O, and EMA.

Default backbone: timm's `swin_tiny_patch4_window7_224` (ImageNet-22k->1k
pretrained). Architecture and defaults follow:
  Liu et al., "Swin Transformer: Hierarchical Vision Transformer using
  Shifted Windows", ICCV 2021. https://arxiv.org/abs/2103.14030

Stochastic-depth (drop_path) defaults below are copied from the official
Swin-Transformer repo's per-model config
(DROP_PATH_RATE: tiny=0.2, small=0.3, base=0.5):
  https://github.com/microsoft/Swin-Transformer
  (mirror of config.py: https://openi.pcl.ac.cn/Goodman2023/SwinTransformer/src/branch/master/config.py)
"""
from typing import Optional

import timm
import torch
from timm.data import resolve_data_config

# Official per-size stochastic-depth rates (Swin-Transformer config.py defaults).
DEFAULT_DROP_PATH = {
    "swin_tiny_patch4_window7_224": 0.2,
    "swin_small_patch4_window7_224": 0.3,
    "swin_base_patch4_window7_224": 0.5,
    "swin_base_patch4_window12_384": 0.5,
    "swin_large_patch4_window7_224": 0.5,
    "swin_large_patch4_window12_384": 0.5,
}


def create_model(name: str, num_classes: int, pretrained: bool = True,
                 drop_path_rate: Optional[float] = None, dropout: float = 0.0):
    if drop_path_rate is None:
        drop_path_rate = DEFAULT_DROP_PATH.get(name, 0.2)
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes,
                             drop_path_rate=drop_path_rate, drop_rate=dropout)


def data_config(model, img_size: Optional[int] = None) -> dict:
    cfg = resolve_data_config({}, model=model)
    return {
        "img_size": int(img_size or cfg["input_size"][1]),
        "mean": [float(x) for x in cfg["mean"]],
        "std": [float(x) for x in cfg["std"]],
    }


def set_backbone_frozen(model, frozen: bool) -> None:
    """Freeze everything except the classification head during the warm-up epochs."""
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    for p in model.parameters():
        p.requires_grad = (not frozen) or (id(p) in head_ids)


def build_layer_decay_groups(model, base_lr: float, weight_decay: float, layer_decay: float):
    """Layer-wise LR decay across Swin's 4 stages, assigning progressively smaller
    LRs to earlier (more generic) stages. Follows the layer-wise decay fine-tuning
    strategy used for Swin-B/L in:
      MixMAE (CVPR 2023), App. A.1: https://arxiv.org/pdf/2205.13137
      FastMIM, App. A.1 (grid-searched decay for Swin-B/L): https://arxiv.org/pdf/2212.06593
    Optional - plain uniform LR (layer_decay=1.0) is the timm "sw" recipe default
    for standard supervised fine-tuning: https://huggingface.co/docs/timm/hparams
    """
    # timm SwinTransformer: patch_embed -> layers[0..3] (stages) -> norm -> head
    num_stages = len(model.layers) if hasattr(model, "layers") else 4
    num_layers = num_stages + 1  # + stem

    def layer_id(name: str) -> int:
        if name.startswith("patch_embed"):
            return 0
        if name.startswith("layers."):
            return int(name.split(".")[1]) + 1
        return num_layers  # norm / head: finetune at full LR

    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lid = layer_id(name)
        scale = layer_decay ** (num_layers - lid)
        wd = 0.0 if (p.ndim == 1 or name.endswith(".bias")) else weight_decay
        key = (lid, wd)
        if key not in groups:
            groups[key] = {"params": [], "lr": base_lr * scale, "weight_decay": wd}
        groups[key]["params"].append(p)
    return list(groups.values())


def save_checkpoint(path, model, model_name, class_names, cfg, extra=None, ema=None):
    payload = {
        "model_name": model_name,
        "state_dict": model.state_dict(),
        "class_names": list(class_names),
        "data_cfg": cfg,
        "extra": extra or {},
    }
    if ema is not None:
        payload["ema_state_dict"] = ema.module.state_dict()
    torch.save(payload, path)


def load_checkpoint(path, device, use_ema=True):
    ckpt = torch.load(path, map_location=device)
    model = create_model(ckpt["model_name"], len(ckpt["class_names"]), pretrained=False)
    state = ckpt.get("ema_state_dict") if (use_ema and "ema_state_dict" in ckpt) else ckpt["state_dict"]
    model.load_state_dict(state)
    return model.to(device).eval(), ckpt
