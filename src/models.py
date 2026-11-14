"""Model definitions: Random Forest baseline, Dense neural network and 1D CNN."""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def make_rf(cfg, seed=42):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(n_estimators=cfg["random_forest"]["n_estimators"],
                                  n_jobs=-1, random_state=seed)


def build_dense(n_features, n_classes, cfg):
    """561 -> 512 -> 256 -> 128 -> 6, each hidden layer Dense + BatchNorm + ReLU + Dropout."""
    from tensorflow import keras
    from tensorflow.keras import layers

    c = cfg["dense"]
    inputs = keras.Input(shape=(n_features,))
    x = inputs
    for units, drop in zip(c["units"], c["dropout"]):
        x = layers.Dense(units, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(drop)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="har_dense")
    model.compile(optimizer=keras.optimizers.Adam(c["learning_rate"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_cnn(n_features, n_classes, cfg):
    """Three Conv1D blocks (32/64/128 filters, kernels 7/5/3) -> GAP -> Dense 128 -> softmax."""
    from tensorflow import keras
    from tensorflow.keras import layers

    c = cfg["cnn"]
    inp = keras.Input(shape=(n_features, 1))
    x = inp
    for filters, k in zip(c["filters"], c["kernels"]):
        x = layers.Conv1D(filters, k, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling1D(c["pool_size"])(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(c["dense_units"], activation="relu")(x)
    x = layers.Dropout(c["dropout"])(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inp, out, name="har_cnn1d")
    model.compile(optimizer=keras.optimizers.Adam(c["learning_rate"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def training_callbacks(c, checkpoint_path=None):
    """EarlyStopping (restore best weights) + ReduceLROnPlateau (+ optional checkpoint)."""
    from tensorflow import keras

    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=c["early_stopping_patience"],
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=c["lr_factor"],
                                          patience=c["lr_patience"], min_lr=c["min_lr"]),
    ]
    if checkpoint_path:
        cbs.append(keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor="val_loss",
                                                   save_best_only=True))
    return cbs
