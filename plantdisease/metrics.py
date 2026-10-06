"""Metrics + report writing."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score,
                             confusion_matrix, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)

from .plots import plot_confusion


def compute_metrics(y_true, probs, num_classes) -> dict:
    y_pred = probs.argmax(1)
    labels = list(range(num_classes))
    out = {"accuracy": accuracy_score(y_true, y_pred)}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=avg, zero_division=0)
        out.update({f"precision_{avg}": p, f"recall_{avg}": r, f"f1_{avg}": f})
    out["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)
    out["mcc"] = matthews_corrcoef(y_true, y_pred)
    try:
        out["roc_auc_ovr_macro"] = roc_auc_score(
            y_true, probs, multi_class="ovr", average="macro", labels=labels)
    except ValueError:
        out["roc_auc_ovr_macro"] = None
    return {k: (None if v is None else float(v)) for k, v in out.items()}


def save_evaluation(out_dir, y_true, probs, class_names, tag="test", paths=None, extra=None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(class_names)
    y_pred = probs.argmax(1)

    metrics = compute_metrics(y_true, probs, n)
    if extra:
        metrics.update(extra)
    (out_dir / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2))

    report = classification_report(y_true, y_pred, labels=list(range(n)),
                                   target_names=class_names, digits=4, zero_division=0)
    (out_dir / f"{tag}_classification_report.txt").write_text(report)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        out_dir / f"{tag}_confusion_matrix.csv")
    plot_confusion(cm, class_names, out_dir / f"{tag}_confusion_matrix.png")
    plot_confusion(cm, class_names, out_dir / f"{tag}_confusion_matrix_normalized.png", normalize=True)

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n)), zero_division=0)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    tn = cm.sum() - cm.sum(0) - cm.sum(1) + tp
    spec = tn / np.maximum(tn + fp, 1)
    pd.DataFrame({"class": class_names, "precision": p, "recall": r, "f1": f,
                  "specificity": spec, "support": s}).to_csv(
        out_dir / f"{tag}_per_class_metrics.csv", index=False)

    pred_df = pd.DataFrame({
        "true": [class_names[i] for i in y_true],
        "pred": [class_names[i] for i in y_pred],
        "confidence": probs.max(1),
        "correct": y_true == y_pred,
    })
    if paths is not None:
        pred_df.insert(0, "path", list(paths))
    pred_df.to_csv(out_dir / f"{tag}_predictions.csv", index=False)

    print(f"\n===== {tag.upper()} RESULTS =====")
    for k, v in metrics.items():
        print(f"{k:24s}: {v:.4f}" if isinstance(v, float) else f"{k:24s}: {v}")
    print("\n" + report)
    return metrics
