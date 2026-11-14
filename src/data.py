"""Load the UCI HAR dataset (Kaggle CSV version) and build a subject-wise split.

Usage:  python -m src.data data      -> prints shapes, missing values, class counts
"""
import sys
from pathlib import Path


def load_har(data_dir):
    """Return X_train, y_train, subj_train, X_test, y_test, subj_test, label_encoder."""
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    data_dir = Path(data_dir)
    for name in ("train.csv", "test.csv"):
        if not (data_dir / name).exists():
            raise FileNotFoundError(
                f"{data_dir / name} not found. Put train.csv and test.csv in '{data_dir}' "
                "(see data/README.md or the 04_Dataset folder).")
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    # Label-encode Activity (fit on train only, reuse on test)
    le = LabelEncoder()
    y_train = le.fit_transform(train["Activity"])
    y_test = le.transform(test["Activity"])

    # Keep subject separately: it is used only for splitting, never as a feature
    subj_train = train["subject"].to_numpy()
    subj_test = test["subject"].to_numpy()

    X_train = train.drop(columns=["subject", "Activity"])
    X_test = test.drop(columns=["subject", "Activity"])
    return X_train, y_train, subj_train, X_test, y_test, subj_test, le


def subject_split(X, y, groups, val_size=0.2, seed=42):
    """Validation split in which no subject appears in both train and validation."""
    from sklearn.model_selection import GroupShuffleSplit

    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr_idx, va_idx = next(gss.split(X, y, groups=groups))
    return tr_idx, va_idx


def main(data_dir="data"):
    import pandas as pd

    X_train, y_train, s_train, X_test, y_test, s_test, le = load_har(data_dir)
    print("X_train:", X_train.shape, "| X_test:", X_test.shape)
    print("Missing values (train/test):", int(X_train.isna().sum().sum()), "/", int(X_test.isna().sum().sum()))
    print("Duplicate rows (train/test):", int(X_train.duplicated().sum()), "/", int(X_test.duplicated().sum()))
    print("Feature range:", float(X_train.min().min()), "to", float(X_train.max().max()))
    print("\nLabel mapping:")
    for i, name in enumerate(le.classes_):
        print(f"  {i} = {name}")
    print("\nClass counts (train):")
    print(pd.Series(le.inverse_transform(y_train)).value_counts().to_string())
    print("\nSubjects: train", len(set(s_train)), "| test", len(set(s_test)),
          "| overlap", len(set(s_train) & set(s_test)))
    tr, va = subject_split(X_train, y_train, s_train)
    print(f"Subject-wise split: train={len(tr)} windows ({len(set(s_train[tr]))} people), "
          f"val={len(va)} windows ({len(set(s_train[va]))} people), "
          f"overlap={len(set(s_train[tr]) & set(s_train[va]))}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
