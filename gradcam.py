"""Grad-CAM heat-map: which part of the leaf drove the prediction.

Targets the output of Swin's last stage (model.layers[-1]) by default - the
last spatial feature map before global pooling. Swin's internal feature maps
are channel-last (B, H, W, C) in timm, unlike a CNN's (B, C, H, W), so this
converts automatically before computing the CAM. Grad-CAM itself is the
standard method of Selvaraju et al., ICCV 2017 (https://arxiv.org/abs/1610.02391),
applied here to a transformer's last spatial stage rather than a conv layer -
the same approach used for the Swin backbone in the two-stage Swin-Tiny
grapevine-disease study (Politecnico di Torino):
https://iris.polito.it/retrieve/handle/11583/3013827/1002699

Example:
    python gradcam.py --checkpoint runs/exp1/best_model.pth --image leaf.jpg --output cam.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

from plantdisease.data import build_eval_transform
from plantdisease.model import load_checkpoint
from plantdisease.utils import get_device


def _to_nchw(t: torch.Tensor) -> torch.Tensor:
    """Swin stage outputs can be (B,H,W,C), (B,L,C) or already (B,C,H,W)."""
    if t.dim() == 4:
        # Heuristic: in (B,H,W,C) the channel dim (last) is usually >> H or W.
        if t.shape[-1] > t.shape[1] and t.shape[-1] > t.shape[2]:
            return t.permute(0, 3, 1, 2).contiguous()
        return t
    if t.dim() == 3:  # (B, L, C) -> square spatial grid
        b, L, c = t.shape
        side = int(L ** 0.5)
        if side * side != L:
            raise ValueError(f"Can't reshape {L} tokens into a square grid; pass --target_layer.")
        return t.transpose(1, 2).reshape(b, c, side, side)
    raise ValueError(f"Unsupported feature shape {tuple(t.shape)} for Grad-CAM.")


class GradCAM:
    def __init__(self, model, target_layer):
        self.model, self.acts, self.grads = model, None, None
        target_layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        out = out[0] if isinstance(out, (tuple, list)) else out
        self.acts = _to_nchw(out)
        out.register_hook(lambda g: setattr(self, "grads", _to_nchw(g)))

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        with torch.enable_grad():
            logits = self.model(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            probs = torch.softmax(logits, dim=1)[0].detach().cpu()
            idx = int(logits.argmax(1)) if class_idx is None else class_idx
            logits[0, idx].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy(), idx, probs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="runs/exp1/best_model.pth")
    ap.add_argument("--image", required=True)
    ap.add_argument("--output", default="gradcam.png")
    ap.add_argument("--class_idx", type=int, default=None, help="default: predicted class")
    ap.add_argument("--target_layer", default=None,
                    help="module attribute path, default: layers[-1]")
    args = ap.parse_args(argv)

    device = get_device()
    model, ckpt = load_checkpoint(args.checkpoint, device)
    class_names, cfg = ckpt["class_names"], ckpt["data_cfg"]

    if args.target_layer:
        layer = model.get_submodule(args.target_layer)
    elif hasattr(model, "layers"):
        layer = model.layers[-1]
    else:
        raise AttributeError("Could not find a default target layer; pass --target_layer.")
    cam_fn = GradCAM(model, layer)

    img = Image.open(args.image).convert("RGB")
    x = build_eval_transform(cfg)(img).unsqueeze(0).to(device)
    cam, idx, probs = cam_fn(x, args.class_idx)

    size = cfg["img_size"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
    ax[0].imshow(img.resize((size, size)))
    ax[0].set_title("Input")
    ax[1].imshow(img.resize((size, size)))
    ax[1].imshow(cam, cmap="jet", alpha=0.45)
    ax[1].set_title(f"{class_names[idx]}\n({probs[idx]:.1%})")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"Saved {args.output} | predicted: {class_names[idx]} ({probs[idx]:.4f})")


if __name__ == "__main__":
    main()
