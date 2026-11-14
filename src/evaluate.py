"""Metrics and figures shared by all models."""
import json
from pathlib import Path

import numpy as np

GREEN, MUTED, PAPER, INK = "#1D3F04", "#92998E", "#F6F4EF", "#111111"


def summarize(y_true, y_pred, classes, extra=None) -> dict:
    """Accuracy, macro precision/recall/F1, per-class scores and confusion matrix."""
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)

    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(len(classes)), zero_division=0)
    out = {
        "test_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(p.mean()), 4),
        "recall_macro": round(float(r.mean()), 4),
        "macro_f1": round(float(f.mean()), 4),
        "per_class": {c: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                          "f1": round(float(f[i]), 4)} for i, c in enumerate(classes)},
        "confusion": confusion_matrix(y_true, y_pred, labels=range(len(classes))).tolist(),
    }
    if extra:
        out.update(extra)
    return out


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))


SHORT = {"LAYING": "Laying", "SITTING": "Sitting", "STANDING": "Standing", "WALKING": "Walking",
         "WALKING_DOWNSTAIRS": "Downstairs", "WALKING_UPSTAIRS": "Upstairs"}


def plot_confusion(cm, classes, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cm = np.asarray(cm)
    norm = cm / cm.sum(1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("g", [PAPER, GREEN])
    labels = [SHORT.get(c, c) for c in classes]
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")
    ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=10,
                    color="white" if norm[i, j] > 0.5 else INK)
    ax.set_xticks(range(len(classes)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(classes)), labels)
    ax.set_xlabel("Predicted activity"); ax.set_ylabel("True activity")
    ax.set_title(title, loc="left", fontsize=12)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_curves(histories: dict, path):
    """histories: {name: {"accuracy": [...], "val_accuracy": [...], "best_epoch": int}}"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(histories), figsize=(6 * len(histories), 4.2), facecolor="white", squeeze=False)
    for ax, (name, h) in zip(axes[0], histories.items()):
        ep = range(1, len(h["accuracy"]) + 1)
        ax.plot(ep, h["accuracy"], color=MUTED, lw=1.8, label="Train")
        ax.plot(ep, h["val_accuracy"], color=GREEN, lw=2.2, label="Validation")
        if h.get("best_epoch"):
            ax.axvline(h["best_epoch"], color=INK, ls=":", lw=1, label=f"Best epoch {h['best_epoch']}")
        ax.set_title(name, loc="left"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
        ax.grid(alpha=0.25); ax.legend(frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_comparison(results: dict, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    acc = [results[n]["test_accuracy"] * 100 for n in names]
    f1 = [results[n]["macro_f1"] * 100 for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")
    ax.bar(x - 0.18, acc, 0.36, color=GREEN, label="Test accuracy (%)")
    ax.bar(x + 0.18, f1, 0.36, color=MUTED, label="Macro F1 (×100)")
    for i, (a, b) in enumerate(zip(acc, f1)):
        ax.text(i - 0.18, a + 0.3, f"{a:.1f}", ha="center", fontsize=9)
        ax.text(i + 0.18, b + 0.3, f"{b:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x, names); ax.set_ylim(80, 100); ax.legend(frameon=False, loc="upper right", ncol=2)
    ax.set_title("Model comparison on the 9 unseen test subjects", loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
