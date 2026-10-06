"""End-to-end smoke test on tiny synthetic data (no internet, random-init model)."""
import json

import numpy as np
from PIL import Image


def _make_dataset(root, n_classes=3, n_per_class=16, size=224):
    rng = np.random.default_rng(0)
    for c in range(n_classes):
        d = root / f"class_{c}"
        d.mkdir(parents=True)
        for i in range(n_per_class):
            arr = rng.integers(0, 60, (size, size, 3), dtype=np.uint8)
            arr[..., c] += 150
            Image.fromarray(arr).save(d / f"img_{i}.png")


def test_train_predict_evaluate(tmp_path):
    from evaluate import main as evaluate_main
    from predict import main as predict_main
    from train import main as train_main

    data, out = tmp_path / "data", tmp_path / "out"
    _make_dataset(data)
    train_main(["--data_dir", str(data), "--output_dir", str(out),
                "--model", "swin_tiny_patch4_window7_224", "--no_pretrained",
                "--epochs", "2", "--batch_size", "4", "--img_size", "224",
                "--workers", "0", "--freeze_epochs", "1", "--warmup_epochs", "1",
                "--ema"])

    for name in ["best_model.pth", "test_metrics.json", "test_classification_report.txt",
                 "test_confusion_matrix.png", "test_per_class_metrics.csv", "training_curves.png"]:
        assert (out / name).exists(), name
    metrics = json.loads((out / "test_metrics.json").read_text())
    assert 0.0 <= metrics["accuracy"] <= 1.0

    img = next((data / "class_0").glob("*.png"))
    predict_main(["--checkpoint", str(out / "best_model.pth"), "--input", str(img)])
    evaluate_main(["--checkpoint", str(out / "best_model.pth"), "--test_dir", str(data),
                   "--workers", "0"])
    assert (out / "eval_metrics.json").exists()


def test_train_with_mixup_and_resume(tmp_path):
    from train import main as train_main

    data, out = tmp_path / "data", tmp_path / "out"
    _make_dataset(data)
    common = ["--data_dir", str(data), "--output_dir", str(out),
              "--model", "swin_tiny_patch4_window7_224", "--no_pretrained",
              "--batch_size", "4", "--img_size", "224", "--workers", "0",
              "--freeze_epochs", "0", "--warmup_epochs", "1",
              "--mixup", "0.8", "--cutmix", "1.0"]
    train_main(common + ["--epochs", "1"])
    assert (out / "last_model.pth").exists()
    # resume for one more epoch
    train_main(common + ["--epochs", "2", "--resume"])
    assert (out / "best_model.pth").exists()
