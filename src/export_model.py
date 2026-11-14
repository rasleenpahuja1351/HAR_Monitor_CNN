"""Export the trained Keras Dense network to NumPy weights for the web API.

TensorFlow is far too large for a Vercel function. The deployed network is only
Dense + BatchNorm + ReLU layers, so BatchNorm and the StandardScaler are folded
into the Dense weights:

    BatchNorm:  k = gamma / sqrt(var + eps),  W' = W * k,  b' = (b - mean) * k + beta
    Scaler:     (x - mu) / s @ W = x @ (W / s) - (mu / s) @ W

Usage (from the project root, after training):
    python -m src.export_model --data data
Writes model/har_mlp.npz, model/metrics.json and model/samples.npz.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from .config import load_config, resolve
from .data import load_har, subject_split
from .inference import HARModel


def fold_model(model, scaler):
    from tensorflow import keras

    layers_out, pending = [], None
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense):
            w = layer.kernel.numpy().astype(np.float64)
            b = layer.bias.numpy().astype(np.float64) if layer.use_bias else np.zeros(w.shape[1])
            pending = [w, b]
            layers_out.append(pending)
        elif isinstance(layer, keras.layers.BatchNormalization):
            gamma, beta = layer.gamma.numpy(), layer.beta.numpy()
            mean, var = layer.moving_mean.numpy(), layer.moving_variance.numpy()
            k = gamma / np.sqrt(var + layer.epsilon)
            pending[0] = pending[0] * k
            pending[1] = (pending[1] - mean) * k + beta
    w0, b0 = layers_out[0]
    mu, s = scaler.mean_, scaler.scale_
    layers_out[0] = [w0 / s[:, None], b0 - (mu / s) @ w0]
    return [(w.astype(np.float32), b.astype(np.float32)) for w, b in layers_out]


def export(cfg, data_dir, out_dir=None, results=None):
    import joblib
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from tensorflow import keras

    ckpt = resolve(cfg, "checkpoints")
    out = Path(out_dir) if out_dir else resolve(cfg, "model")
    out.mkdir(parents=True, exist_ok=True)

    model = keras.models.load_model(ckpt / "best_har_mlp.keras")
    pre = joblib.load(ckpt / "har_mlp_preprocess.joblib")
    scaler, le = pre["scaler"], pre["label_encoder"]
    X, y, s, Xt, yt, st, _ = load_har(data_dir)
    layers = fold_model(model, scaler)

    arrays = {}
    for i, (w, b) in enumerate(layers):
        arrays[f"W{i}"], arrays[f"b{i}"] = w, b
    np.savez_compressed(out / "har_mlp.npz", n_layers=len(layers),
                        classes=np.array([str(c) for c in le.classes_], dtype=str),
                        features=np.array(list(X.columns), dtype=str), **arrays)

    # Parity check: the NumPy model must make the same predictions as Keras
    np_model = HARModel(out / "har_mlp.npz")
    keras_p = model.predict(scaler.transform(Xt).astype("float32"), verbose=0)
    np_p = np_model.predict_proba(Xt.to_numpy())
    agree = float((keras_p.argmax(1) == np_p.argmax(1)).mean())
    assert agree > 0.999, f"NumPy export does not match Keras (agreement {agree:.4f})"
    print(f"Parity with Keras: {agree * 100:.2f}% identical predictions, "
          f"max probability difference {float(np.abs(keras_p - np_p).max()):.2e}")

    y_nn = np_p.argmax(1)
    tr, va = subject_split(X, y, s, cfg["split"]["val_size"], cfg["seed"])
    val_nn = np_model.predict_proba(X.iloc[va].to_numpy()).argmax(1)
    metrics = {
        "group_id": cfg.get("group_id"),
        "classes": list(le.classes_),
        "dataset": {
            "train_windows": int(len(X)), "test_windows": int(len(Xt)), "features": int(X.shape[1]),
            "train_subjects": int(len(set(s))), "test_subjects": int(len(set(st))),
            "train_counts": np.bincount(y, minlength=len(le.classes_)).tolist(),
        },
        "keras": {
            "test_accuracy": float(accuracy_score(yt, y_nn)),
            "val_accuracy": float(accuracy_score(y[va], val_nn)),
            "macro_f1": float(f1_score(yt, y_nn, average="macro")),
            "f1": f1_score(yt, y_nn, average=None).round(4).tolist(),
            "confusion": confusion_matrix(yt, y_nn).tolist(),
            "params": int(model.count_params()),
            "architecture": "561 → 512 → 256 → 128 → 6 (Dense + BatchNorm + ReLU + Dropout)",
        },
    }
    hist_path = ckpt / "history_dense.json"
    if hist_path.exists():
        h = json.loads(hist_path.read_text())
        metrics["keras"]["history"] = h
        metrics["keras"]["best_epoch"] = int(np.argmin(h["val_loss"])) + 1
    rf_path = ckpt / "rf_baseline.joblib"
    if rf_path.exists():
        rf = joblib.load(rf_path)["model"]
        y_rf = rf.predict(Xt)
        metrics["random_forest"] = {
            "test_accuracy": float(accuracy_score(yt, y_rf)),
            "macro_f1": float(f1_score(yt, y_rf, average="macro")),
            "f1": f1_score(yt, y_rf, average=None).round(4).tolist(),
        }
    if results and "cnn" in results:
        c = results["cnn"]
        metrics["cnn"] = {"test_accuracy": c["test_accuracy"], "macro_f1": c["macro_f1"],
                          "params": c.get("params"), "val_accuracy": c.get("val_accuracy")}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))

    # Real held-out test windows for the live demo (stratified)
    k = cfg["deployment"]["live_samples_per_class"]
    rng = np.random.default_rng(cfg["seed"])
    idx = np.concatenate([rng.choice(np.where(yt == c)[0], k, replace=False) for c in range(len(le.classes_))])
    rng.shuffle(idx)
    np.savez_compressed(out / "samples.npz", X=Xt.to_numpy()[idx].astype(np.float32),
                        y=yt[idx].astype(np.int8), subject=st[idx].astype(np.int16))
    print("Wrote", sorted(p.name for p in out.iterdir()), "to", out)
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="folder with train.csv and test.csv (default: config paths.data_dir)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="output folder (default: model/)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    res_file = resolve(cfg, "results") / "metrics_all.json"
    results = json.loads(res_file.read_text()).get("models") if res_file.exists() else None
    export(cfg, args.data or resolve(cfg, "data_dir"), args.out, results)


if __name__ == "__main__":
    main()
