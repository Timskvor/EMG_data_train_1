from __future__ import annotations

from pathlib import Path

import numpy as np


def load_varlen_npz_dataset(npz_path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate a variable-length dataset stored in NPZ format."""
    data = np.load(str(npz_path), allow_pickle=False)
    required = {"emg_concat", "offsets", "lengths", "target", "subject"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"Missing required keys in {npz_path}: {sorted(missing)}")

    out = {
        "emg_concat": np.asarray(data["emg_concat"], dtype=np.float32),
        "offsets": np.asarray(data["offsets"], dtype=np.int64),
        "lengths": np.asarray(data["lengths"], dtype=np.int32),
        "target": np.asarray(data["target"]).reshape(-1).astype(np.int64),
        "subject": np.char.strip(np.asarray(data["subject"]).reshape(-1).astype(str)),
    }
    if "channels" in data.files:
        out["channels"] = np.asarray(data["channels"]).reshape(-1).astype(str)
    if "session" in data.files:
        out["session"] = np.char.strip(np.asarray(data["session"]).reshape(-1).astype(str))

    n = len(out["target"])
    if len(out["subject"]) != n or len(out["lengths"]) != n or len(out["offsets"]) != n + 1:
        raise ValueError(
            f"Inconsistent arrays in {npz_path}: "
            f"target={len(out['target'])}, subject={len(out['subject'])}, "
            f"lengths={len(out['lengths'])}, offsets={len(out['offsets'])}"
        )
    if "session" in out and len(out["session"]) != n:
        raise ValueError(
            f"Inconsistent arrays in {npz_path}: target={n}, session={len(out['session'])}"
        )
    if out["offsets"][0] != 0:
        raise ValueError("offsets[0] must be 0")
    if out["offsets"][-1] != len(out["emg_concat"]):
        raise ValueError("offsets[-1] must equal len(emg_concat)")

    return out

