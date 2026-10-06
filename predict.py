"""Predict the disease class of one image or every image in a folder.

Examples:
    python predict.py --checkpoint runs/exp1/best_model.pth --input leaf.jpg
    python predict.py --checkpoint runs/exp1/best_model.pth --input new_leaves/ --csv preds.csv
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from plantdisease.data import build_eval_transform
from plantdisease.model import load_checkpoint
from plantdisease.utils import get_device

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="runs/exp1/best_model.pth")
    ap.add_argument("--input", required=True, help="image file or folder")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--csv", default=None, help="optional: save predictions to CSV")
    ap.add_argument("--no_ema", action="store_true", help="use raw weights instead of EMA, if present")
    args = ap.parse_args(argv)

    device = get_device()
    model, ckpt = load_checkpoint(args.checkpoint, device, use_ema=not args.no_ema)
    class_names = ckpt["class_names"]
    tf = build_eval_transform(ckpt["data_cfg"])

    src = Path(args.input)
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"No images found in {src}")

    rows, k = [], min(args.topk, len(class_names))
    with torch.no_grad():
        for f in files:
            x = tf(Image.open(f).convert("RGB")).unsqueeze(0).to(device)
            probs = torch.softmax(model(x), dim=1)[0].cpu()
            top_p, top_i = probs.topk(k)
            print(f"\n{f}")
            for p, i in zip(top_p, top_i):
                print(f"  {class_names[i]:40s} {p.item():.4f}")
            rows.append({"path": str(f), "prediction": class_names[top_i[0]],
                         "confidence": top_p[0].item()})
    if args.csv:
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"\nSaved {len(rows)} predictions to {args.csv}")


if __name__ == "__main__":
    main()
