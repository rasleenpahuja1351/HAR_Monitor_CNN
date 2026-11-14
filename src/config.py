"""Load configs/config.json and resolve project paths."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "config.json"


def load_config(path=None) -> dict:
    cfg = json.loads(Path(path or DEFAULT_CONFIG).read_text())
    return cfg


def resolve(cfg: dict, key: str) -> Path:
    """Absolute path for cfg['paths'][key] (relative paths are relative to the project root)."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p


def with_overrides(cfg: dict, epochs=None, rf_trees=None) -> dict:
    """Copy of cfg with quick-run overrides (used for smoke tests)."""
    cfg = copy.deepcopy(cfg)
    if epochs:
        cfg["dense"]["epochs"] = cfg["cnn"]["epochs"] = int(epochs)
    if rf_trees:
        cfg["random_forest"]["n_estimators"] = int(rf_trees)
    return cfg
