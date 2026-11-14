"""Training and evaluation of the three models on the same subject-wise split.

Called by run_experiments.py. Each function returns a metrics dict (see
src.evaluate.summarize) and writes its trained model to the checkpoints folder.
"""
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from .data import load_har, subject_split
from .evaluate import summarize
from .models import build_cnn, build_dense, make_rf, training_callbacks


def prepare(cfg, data_dir):
    """Load data, make the subject-wise split and fit the scaler on the training part only."""
    from sklearn.preprocessing import StandardScaler

    X, y, s, Xt, yt, st, le = load_har(data_dir)
    tr, va = subject_split(X, y, s, cfg["split"]["val_size"], cfg["seed"])
    scaler = StandardScaler().fit(X.iloc[tr])
    f = lambda A: scaler.transform(A).astype("float32")
    return {
        "X": X, "y": y, "s": s, "Xt": Xt, "yt": yt, "st": st, "le": le, "classes": list(le.classes_),
        "tr": tr, "va": va, "scaler": scaler,
        "X_tr": f(X.iloc[tr]), "X_va": f(X.iloc[va]), "X_te": f(Xt), "y_tr": y[tr], "y_va": y[va],
    }


def _history(h):
    return {k: [float(v) for v in h[k]] for k in ["accuracy", "val_accuracy", "loss", "val_loss"]}


# ----------------------------------------------------------------------------- Random Forest
def train_rf(cfg, D, ckpt: Path):
    import joblib
    from sklearn.metrics import accuracy_score

    rf = make_rf(cfg, cfg["seed"]).fit(D["X"].iloc[D["tr"]], D["y_tr"])            # validation run
    val_acc = accuracy_score(D["y_va"], rf.predict(D["X"].iloc[D["va"]]))
    rf = make_rf(cfg, cfg["seed"]).fit(D["X"], D["y"])                               # final fit on all training people
    y_pred = rf.predict(D["Xt"])
    joblib.dump({"model": rf, "label_encoder": D["le"]}, ckpt / "rf_baseline.joblib", compress=3)
    return summarize(D["yt"], y_pred, D["classes"], {
        "val_accuracy": round(float(val_acc), 4), "size": f"{cfg['random_forest']['n_estimators']} trees"})


# ----------------------------------------------------------------------------- Dense NN
def train_dense(cfg, D, ckpt: Path):
    import joblib
    from tensorflow import keras

    keras.utils.set_random_seed(cfg["seed"])
    c = cfg["dense"]
    model = build_dense(D["X_tr"].shape[1], len(D["classes"]), cfg)
    path = ckpt / "best_har_mlp.keras"
    hist = model.fit(D["X_tr"], D["y_tr"], validation_data=(D["X_va"], D["y_va"]), epochs=c["epochs"],
                     batch_size=c["batch_size"], callbacks=training_callbacks(c, path), verbose=2)
    h = _history(hist.history)
    (ckpt / "history_dense.json").write_text(json.dumps(h))
    joblib.dump({"scaler": D["scaler"], "label_encoder": D["le"]}, ckpt / "har_mlp_preprocess.joblib")
    return evaluate_dense(D, ckpt, h)


def evaluate_dense(D, ckpt: Path, history=None):
    from sklearn.metrics import accuracy_score
    from tensorflow import keras

    model = keras.models.load_model(ckpt / "best_har_mlp.keras")
    y_pred = model.predict(D["X_te"], verbose=0).argmax(1)
    val_acc = accuracy_score(D["y_va"], model.predict(D["X_va"], verbose=0).argmax(1))
    history = history or _load_history(ckpt / "history_dense.json")
    return summarize(D["yt"], y_pred, D["classes"], {
        "val_accuracy": round(float(val_acc), 4), "params": int(model.count_params()),
        "size": f"{model.count_params():,} params",
        **({"best_epoch": int(np.argmin(history["val_loss"])) + 1, "epochs_run": len(history["loss"]),
            "history": history} if history else {})})


# ----------------------------------------------------------------------------- 1D CNN
def train_cnn(cfg, D, ckpt: Path):
    from tensorflow import keras

    keras.utils.set_random_seed(cfg["seed"])
    c = cfg["cnn"]
    model = build_cnn(D["X_tr"].shape[1], len(D["classes"]), cfg)
    hist = model.fit(D["X_tr"][..., None], D["y_tr"], validation_data=(D["X_va"][..., None], D["y_va"]),
                     epochs=c["epochs"], batch_size=c["batch_size"], callbacks=training_callbacks(c), verbose=2)
    model.save(ckpt / "best_har_cnn1d.keras")        # EarlyStopping already restored the best weights
    h = _history(hist.history)
    (ckpt / "history_cnn.json").write_text(json.dumps(h))
    return evaluate_cnn(D, ckpt, h)


def evaluate_cnn(D, ckpt: Path, history=None):
    from sklearn.metrics import accuracy_score
    from tensorflow import keras

    model = keras.models.load_model(ckpt / "best_har_cnn1d.keras")
    y_pred = model.predict(D["X_te"][..., None], verbose=0).argmax(1)
    val_acc = accuracy_score(D["y_va"], model.predict(D["X_va"][..., None], verbose=0).argmax(1))
    history = history or _load_history(ckpt / "history_cnn.json")
    return summarize(D["yt"], y_pred, D["classes"], {
        "val_accuracy": round(float(val_acc), 4), "params": int(model.count_params()),
        "size": f"{model.count_params():,} params",
        **({"best_epoch": int(np.argmin(history["val_loss"])) + 1, "epochs_run": len(history["loss"]),
            "history": history} if history else {})})


def evaluate_rf(cfg, D, ckpt: Path):
    """Evaluate the saved Random Forest; validation accuracy is recomputed with a fit on the training people only."""
    import joblib
    from sklearn.metrics import accuracy_score

    rf = joblib.load(ckpt / "rf_baseline.joblib")["model"]
    y_pred = rf.predict(D["Xt"])
    rf_val = make_rf(cfg, cfg["seed"]).fit(D["X"].iloc[D["tr"]], D["y_tr"])
    val_acc = accuracy_score(D["y_va"], rf_val.predict(D["X"].iloc[D["va"]]))
    return summarize(D["yt"], y_pred, D["classes"], {
        "val_accuracy": round(float(val_acc), 4), "size": f"{rf.n_estimators} trees"})


def _load_history(path: Path):
    return json.loads(path.read_text()) if path.exists() else None
