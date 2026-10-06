import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def plot_history(hist: dict, out_path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["train_loss"], label="train")
    ax[0].plot(hist["val_loss"], label="val")
    ax[0].set(title="Loss", xlabel="epoch")
    ax[0].legend()
    ax[1].plot(hist["train_acc"], label="train acc")
    ax[1].plot(hist["val_acc"], label="val acc")
    ax[1].plot(hist["val_f1"], label="val macro-F1")
    ax[1].set(title="Accuracy / F1", xlabel="epoch")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_confusion(cm, class_names, out_path, normalize=False):
    cm = cm.astype(float)
    if normalize:
        cm = cm / cm.sum(1, keepdims=True).clip(min=1)
    n = len(class_names)
    side = max(6, n * 0.5)
    plt.figure(figsize=(side, side * 0.85))
    sns.heatmap(cm, annot=n <= 25, fmt=".2f" if normalize else ".0f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title("Normalized confusion matrix" if normalize else "Confusion matrix")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
