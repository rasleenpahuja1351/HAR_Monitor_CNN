"""Predict activities for a CSV of UCI HAR feature rows (no TensorFlow needed).

Examples:
    python predict.py data/test.csv
    python predict.py data/test.csv --out results/predictions.csv
    python predict.py my_windows.csv --model model/har_mlp.npz

The CSV must contain the 561 feature columns. A header row is optional; if it
has 'subject' and 'Activity' columns they are ignored for prediction, and if
'Activity' is present the accuracy is printed.
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from src.inference import HARModel

ROOT = Path(__file__).resolve().parent


def read_csv(path, n_features):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError("the file is empty")
    header = rows[0]
    has_header = any(not _is_number(v) for v in header if v.strip())
    labels = None
    if has_header:
        names = [h.strip() for h in header]
        lower = [n.lower() for n in names]
        label_idx = lower.index("activity") if "activity" in lower else None
        feat_idx = [i for i, n in enumerate(lower) if n not in ("activity", "subject")]
        body = rows[1:]
        if label_idx is not None:
            labels = [r[label_idx].strip() for r in body]
    else:
        feat_idx = list(range(n_features))
        body = rows
    if len(feat_idx) != n_features:
        raise ValueError(f"expected {n_features} feature columns, found {len(feat_idx)}")
    X = np.array([[float(r[i]) for i in feat_idx] for r in body], dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError("the file contains missing or non-numeric values")
    return X, labels


def _is_number(v):
    try:
        float(v)
        return True
    except ValueError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="CSV file with 561 feature columns")
    ap.add_argument("--model", default=str(ROOT / "model" / "har_mlp.npz"))
    ap.add_argument("--out", default=None, help="write predictions to this CSV")
    ap.add_argument("--threshold", type=float, default=0.7, help="flag predictions below this confidence")
    args = ap.parse_args()

    try:
        model = HARModel(args.model)
        X, labels = read_csv(args.csv, model.n_features)
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    preds, conf = model.predict(X)
    print(f"Classified {len(preds):,} windows")
    for c, n in Counter(preds).most_common():
        print(f"  {c:<20} {n:>6}")
    print(f"Low confidence (< {args.threshold:.0%}): {int((conf < args.threshold).sum())}")
    if labels:
        acc = np.mean([p == t for p, t in zip(preds, labels)])
        print(f"Accuracy against the Activity column: {acc * 100:.2f}%")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row", "predicted", "confidence", "low_confidence"] + (["true_activity", "correct"] if labels else []))
            for i, (p, c) in enumerate(zip(preds, conf)):
                extra = [labels[i], p == labels[i]] if labels else []
                w.writerow([i + 1, p, f"{c:.4f}", bool(c < args.threshold)] + extra)
        print("Wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
