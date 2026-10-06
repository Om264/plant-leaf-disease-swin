"""Evaluate a trained checkpoint on any ImageFolder-style directory.

Example:
    python evaluate.py --checkpoint runs/exp1/best_model.pth --test_dir data/external_test
"""
import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from plantdisease.data import build_eval_transform, make_eval_dataset
from plantdisease.engine import predict_loader
from plantdisease.metrics import save_evaluation
from plantdisease.model import load_checkpoint
from plantdisease.utils import get_device


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="runs/exp1/best_model.pth")
    ap.add_argument("--test_dir", required=True, help="folder with one sub-folder per class")
    ap.add_argument("--output_dir", default=None, help="default: checkpoint's folder")
    ap.add_argument("--tag", default="eval", help="prefix for the output files")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no_ema", action="store_true", help="use raw weights instead of EMA, if present")
    args = ap.parse_args(argv)

    device = get_device()
    model, ckpt = load_checkpoint(args.checkpoint, device, use_ema=not args.no_ema)
    class_names = ckpt["class_names"]
    ds = make_eval_dataset(args.test_dir, class_names, build_eval_transform(ckpt["data_cfg"]))
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)
    probs, labels, _ = predict_loader(model, loader, device)
    out_dir = args.output_dir or str(Path(args.checkpoint).parent)
    save_evaluation(out_dir, labels, probs, class_names, tag=args.tag,
                    paths=[p for p, _ in ds.samples])


if __name__ == "__main__":
    main()
