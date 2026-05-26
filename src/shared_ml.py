from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_class(model_py: Path):
    """Load CNNTransformerClassifier from a Python file."""
    spec = importlib.util.spec_from_file_location("legacy_cnn_transformer", model_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load model from {model_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    if not hasattr(module, "CNNTransformerClassifier"):
        raise AttributeError("CNNTransformerClassifier not found in model file")
    return module.CNNTransformerClassifier


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute unweighted macro F1 over labels present in y_true/y_pred."""
    if len(y_true) == 0:
        return 0.0
    labels = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    for cls in labels:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        f1s.append(float(f1))
    return float(np.mean(f1s)) if f1s else 0.0


def get_window(var_ds: dict[str, np.ndarray], index: int) -> np.ndarray:
    """Extract a variable-length window [T, C] from packed representation."""
    s = int(var_ds["offsets"][index])
    e = int(var_ds["offsets"][index + 1])
    return np.asarray(var_ds["emg_concat"][s:e], dtype=np.float32)

