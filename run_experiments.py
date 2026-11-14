"""Run the full experiment: Random Forest, Dense NN and 1D CNN on one subject-wise split.

Examples (from the project root):
    python run_experiments.py                      # train all three models (several minutes on CPU)
    python run_experiments.py --evaluate-only      # reuse saved checkpoints, recompute results + figures
    python run_experiments.py --models dense cnn   # train only some models
    python run_experiments.py --export             # also refresh model/ for the web app
    python run_experiments.py --quick --out /tmp/smoke   # 2-epoch smoke test, writes elsewhere

Outputs (results/):
    metrics_all.json        accuracy, precision, recall, F1, confusion matrix per model
    summary.md              comparison table
    figures/*.png           confusion matrices, training curves, model comparison
"""
import argparse
import json
import sys
import time
from pathlib import Path

from src.config import load_config, resolve, with_overrides
from src.evaluate import plot_comparison, plot_confusion, plot_curves, save_json

NAMES = {"dense": "Dense NN", "rf": "Random Forest", "cnn": "1D CNN"}


def write_summary(results, path):
    lines = ["# Results (test set: 2,947 windows from 9 unseen subjects)", "",
             "| Model | Test accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Validation accuracy | Size |",
             "|---|---|---|---|---|---|---|"]
    for k, r in results.items():
        lines.append(f"| {NAMES[k]} | {r['test_accuracy'] * 100:.1f}% | {r['precision_macro']:.3f} | "
                     f"{r['recall_macro']:.3f} | {r['macro_f1']:.3f} | {r['val_accuracy'] * 100:.1f}% | {r.get('size', '')} |")
    lines += ["", "## Per-class F1", "", "| Activity | " + " | ".join(NAMES[k] for k in results) + " |",
              "|---|" + "---|" * len(results)]
    first = next(iter(results.values()))
    for c in first["per_class"]:
        lines.append(f"| {c} | " + " | ".join(f"{results[k]['per_class'][c]['f1']:.3f}" for k in results) + " |")
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="folder with train.csv and test.csv")
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="+", choices=["dense", "rf", "cnn"], default=["dense", "rf", "cnn"])
    ap.add_argument("--evaluate-only", action="store_true", help="do not train; evaluate saved checkpoints")
    ap.add_argument("--export", action="store_true", help="export the Dense model to model/ for the API")
    ap.add_argument("--quick", action="store_true", help="2 epochs, 20 trees (smoke test only)")
    ap.add_argument("--out", default=None, help="results folder (default: results/)")
    ap.add_argument("--checkpoints", default=None, help="checkpoint folder (default: checkpoints/)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.quick:
        cfg = with_overrides(cfg, epochs=2, rf_trees=20)
    if args.checkpoints:
        cfg["paths"]["checkpoints"] = args.checkpoints
    if args.out:
        cfg["paths"]["results"] = args.out
    data_dir = args.data or resolve(cfg, "data_dir")
    ckpt, out = resolve(cfg, "checkpoints"), resolve(cfg, "results")
    ckpt.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    from src import train as T

    t0 = time.time()
    D = T.prepare(cfg, data_dir)
    print(f"Data: train {len(D['tr'])} / val {len(D['va'])} / test {len(D['yt'])} windows; "
          f"{len(set(D['s'][D['tr']]))}/{len(set(D['s'][D['va']]))}/{len(set(D['st']))} subjects")

    results = {}
    for m in [k for k in ["dense", "rf", "cnn"] if k in args.models]:
        print(f"\n=== {NAMES[m]} ({'evaluate' if args.evaluate_only else 'train'}) ===")
        if args.evaluate_only:
            r = {"dense": lambda: T.evaluate_dense(D, ckpt), "rf": lambda: T.evaluate_rf(cfg, D, ckpt),
                 "cnn": lambda: T.evaluate_cnn(D, ckpt)}[m]()
        else:
            r = {"dense": T.train_dense, "rf": T.train_rf, "cnn": T.train_cnn}[m](cfg, D, ckpt)
        results[m] = r
        print(f"{NAMES[m]}: test accuracy {r['test_accuracy'] * 100:.2f}% | macro F1 {r['macro_f1']:.4f} | "
              f"val {r['val_accuracy'] * 100:.2f}%")
        plot_confusion(r["confusion"], D["classes"], f"{NAMES[m]} · test confusion matrix (acc {r['test_accuracy'] * 100:.1f}%)",
                       out / "figures" / f"confusion_{m}.png")

    hist = {NAMES[k]: {**r["history"], "best_epoch": r.get("best_epoch")} for k, r in results.items() if r.get("history")}
    if hist:
        plot_curves(hist, out / "figures" / "training_curves.png")
    if len(results) > 1:
        plot_comparison({NAMES[k]: r for k, r in results.items()}, out / "figures" / "model_comparison.png")

    slim = {k: {kk: vv for kk, vv in r.items() if kk != "history"} for k, r in results.items()}
    save_json({"classes": D["classes"], "split": {"train": len(D["tr"]), "val": len(D["va"]), "test": len(D["yt"])},
               "models": slim}, out / "metrics_all.json")
    write_summary(slim, out / "summary.md")
    print(f"\nWrote {out / 'metrics_all.json'}, {out / 'summary.md'} and figures/ in {time.time() - t0:.0f} s")

    if args.export:
        from src.export_model import export
        export(cfg, data_dir, None, slim)
    return 0


if __name__ == "__main__":
    sys.exit(main())
