from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample


@dataclass(frozen=True)
class SessionFiles:
    subject: str
    session: str
    events_tsv: Path
    bdf_file: Path
    channels_tsv: Path | None
    sidecar_json: Path | None


def discover_sessions(dataset_root: str | Path) -> list[SessionFiles]:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    sessions: list[SessionFiles] = []
    pattern = "sub-*/ses-*/emg/*_task-handwriting_events.tsv"

    for events_tsv in sorted(root.glob(pattern)):
        base = events_tsv.name.replace("_task-handwriting_events.tsv", "")
        bdf_file = events_tsv.with_name(f"{base}_task-handwriting_emg.bdf")
        channels_tsv = events_tsv.with_name(f"{base}_task-handwriting_channels.tsv")
        sidecar_json = events_tsv.with_name(f"{base}_task-handwriting_emg.json")

        if not bdf_file.exists():
            raise FileNotFoundError(f"BDF file is missing for session: {events_tsv}")

        subject = events_tsv.parts[-4]
        session = events_tsv.parts[-3]
        sessions.append(
            SessionFiles(
                subject=subject,
                session=session,
                events_tsv=events_tsv,
                bdf_file=bdf_file,
                channels_tsv=channels_tsv if channels_tsv.exists() else None,
                sidecar_json=sidecar_json if sidecar_json.exists() else None,
            )
        )

    if not sessions:
        raise ValueError(f"No sessions discovered in: {root}")

    return sessions


def read_events(events_tsv: str | Path) -> list[dict[str, Any]]:
    path = Path(events_tsv)
    if not path.exists():
        raise FileNotFoundError(f"Events file not found: {path}")

    events: list[dict[str, Any]] = []

    def _safe_float(v: str, default: float = np.nan) -> float:
        try:
            return float(v)
        except Exception:
            return default

    def _safe_int(v: str, default: int = -1) -> int:
        try:
            return int(float(v))
        except Exception:
            return default

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"onset", "duration", "sample", "value", "prompt_text"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Events file {path} is missing columns: {sorted(missing)}")

        for row in reader:
            events.append(
                {
                    "onset": _safe_float(row.get("onset", "")),
                    "duration": _safe_float(row.get("duration", "")),
                    "sample": _safe_int(row.get("sample", "")),
                    "value": row.get("value", ""),
                    "prompt_text": row.get("prompt_text", ""),
                    "stage_name": row.get("stage_name", ""),
                    "posture": row.get("posture", ""),
                }
            )

    return events


def filter_digit_prompts(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for event in events:
        value = str(event.get("value", "")).strip()
        prompt = str(event.get("prompt_text", "")).strip()
        if value != "handwriting_prompt":
            continue
        if len(prompt) != 1 or prompt not in "0123456789":
            continue
        out.append(event)
    return out


def _is_git_annex_stub(file_path: Path) -> bool:
    try:
        size = file_path.stat().st_size
    except FileNotFoundError:
        return False

    if size > 4096:
        return False

    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return False

    return content.startswith("/annex/objects/")


def _raise_stub_error(file_path: Path) -> None:
    raise RuntimeError(
        "BDF file is a git-annex stub (pointer), not real EMG data:\n"
        f"  {file_path}\n\n"
        "Please download actual annex objects first and rerun preprocessing."
    )


def _load_full_recording_mne(bdf_path: Path) -> tuple[np.ndarray, float, list[str]]:
    import mne

    raw = mne.io.read_raw_bdf(str(bdf_path), preload=True, verbose="ERROR")
    emg_picks = [i for i, name in enumerate(raw.ch_names) if name.upper().startswith("EMG")]
    if not emg_picks:
        raise ValueError(f"No EMG channels found in file: {bdf_path}")

    emg_names = [raw.ch_names[i] for i in emg_picks]
    raw_data = np.asarray(raw.get_data(picks=emg_picks), dtype=np.float32)
    if raw_data.ndim != 2:
        raise ValueError(f"Unexpected MNE data shape for {bdf_path}: {raw_data.shape}")

    data = np.transpose(raw_data, (1, 0)).astype(np.float32, copy=False)
    sfreq = float(raw.info["sfreq"])
    return data, sfreq, emg_names


def _load_full_recording_pyedflib(bdf_path: Path) -> tuple[np.ndarray, float, list[str]]:
    import pyedflib

    reader = pyedflib.EdfReader(str(bdf_path))
    try:
        signal_labels = [str(x) for x in reader.getSignalLabels()]
        emg_picks = [i for i, name in enumerate(signal_labels) if name.upper().startswith("EMG")]
        if not emg_picks:
            raise ValueError(f"No EMG channels found in file: {bdf_path}")

        sfreqs = [float(reader.getSampleFrequency(i)) for i in emg_picks]
        if not np.allclose(sfreqs, sfreqs[0]):
            raise ValueError(f"Inconsistent EMG sample rates in file: {bdf_path}")

        signals = [np.asarray(reader.readSignal(i), dtype=np.float32).reshape(-1) for i in emg_picks]
        data = np.stack(signals, axis=1).astype(np.float32, copy=False)
        sfreq = float(sfreqs[0])
        emg_names = [signal_labels[i] for i in emg_picks]
        return data, sfreq, emg_names
    finally:
        reader.close()


def _load_full_recording(
    bdf_path: str | Path,
    *,
    backend: str = "auto",
) -> tuple[np.ndarray, float, list[str]]:
    path = Path(bdf_path)
    if not path.exists():
        raise FileNotFoundError(f"BDF file not found: {path}")
    if _is_git_annex_stub(path):
        _raise_stub_error(path)

    if backend not in {"auto", "mne", "pyedflib"}:
        raise ValueError("backend must be one of {'auto', 'mne', 'pyedflib'}")

    errors: list[Exception] = []
    candidates = ["mne", "pyedflib"] if backend == "auto" else [backend]

    for candidate in candidates:
        try:
            if candidate == "mne":
                return _load_full_recording_mne(path)
            return _load_full_recording_pyedflib(path)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    missing_mods = [e for e in errors if isinstance(e, ModuleNotFoundError)]
    if missing_mods and len(missing_mods) == len(errors):
        missing_names = sorted({str(getattr(e, "name", "unknown")) for e in missing_mods})
        raise RuntimeError(
            "No available backend to read BDF. Missing modules: "
            + ", ".join(missing_names)
            + ". Install one of: `pip install mne` or `pip install pyedflib`."
        )

    joined = "\n".join(f"- {type(e).__name__}: {e}" for e in errors)
    raise RuntimeError(f"Failed to read BDF file {path} with backend={backend}\n{joined}")


def load_emg_segment(
    bdf_path: str | Path,
    onset_sample: int,
    duration_samples: int,
    *,
    target_num_samples: int,
    channel_indices: Iterable[int] | None = None,
    backend: str = "auto",
    recording_data: np.ndarray | None = None,
) -> np.ndarray:
    if recording_data is None:
        recording_data, _, _ = _load_full_recording(bdf_path, backend=backend)

    start = int(onset_sample)
    stop = start + int(duration_samples)
    if start < 0 or stop > len(recording_data):
        raise ValueError(
            f"Segment [{start}:{stop}] is out of bounds for recording length {len(recording_data)}"
        )

    segment = recording_data[start:stop]
    if channel_indices is not None:
        idx = np.asarray(list(channel_indices), dtype=np.int64)
        if idx.size == 0:
            raise ValueError("channel_indices must not be empty")
        if idx.min() < 0 or idx.max() >= segment.shape[1]:
            raise ValueError(f"channel_indices out of range for C={segment.shape[1]}: {idx.tolist()}")
        segment = segment[:, idx]

    target_len = int(target_num_samples)
    if target_len <= 0:
        raise ValueError(f"target_num_samples must be positive, got {target_len}")

    if len(segment) != target_len:
        resampled = resample(segment, target_len, axis=0)
        if isinstance(resampled, tuple):
            segment = np.asarray(resampled[0], dtype=np.float32)
        else:
            segment = np.asarray(resampled, dtype=np.float32)

    return np.asarray(segment, dtype=np.float32)


def summarize_lengths(lengths: np.ndarray) -> dict[str, float]:
    arr = np.asarray(lengths, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {
            "n": 0.0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "q01": 0.0,
            "q05": 0.0,
            "q95": 0.0,
            "q99": 0.0,
        }
    return {
        "n": float(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "q01": float(np.quantile(arr, 0.01)),
        "q05": float(np.quantile(arr, 0.05)),
        "q95": float(np.quantile(arr, 0.95)),
        "q99": float(np.quantile(arr, 0.99)),
    }


def pack_variable_length_trials(
    emg_list: Iterable[np.ndarray],
    target: np.ndarray,
    subject: np.ndarray,
    *,
    session: np.ndarray | None = None,
    channels: list[str] | None = None,
) -> dict[str, np.ndarray]:
    seqs = [np.asarray(x, dtype=np.float32) for x in emg_list]
    if not seqs:
        raise ValueError("emg_list is empty")
    if any(x.ndim != 2 for x in seqs):
        bad = [i for i, x in enumerate(seqs) if x.ndim != 2][:5]
        raise ValueError(f"All trials must be 2D (T,C). Bad indices: {bad}")

    n_channels = int(seqs[0].shape[1])
    if any(int(x.shape[1]) != n_channels for x in seqs):
        raise ValueError("All trials must have the same channel count")

    target_arr = np.asarray(target, dtype=np.int16).reshape(-1)
    subject_arr = np.asarray(subject).reshape(-1).astype("U16")
    session_arr = None if session is None else np.asarray(session).reshape(-1).astype("U32")
    n = len(seqs)
    if n != len(target_arr) or n != len(subject_arr):
        raise ValueError(
            f"Length mismatch: trials={n}, target={len(target_arr)}, subject={len(subject_arr)}"
        )
    if session_arr is not None and n != len(session_arr):
        raise ValueError(
            f"Length mismatch: trials={n}, session={len(session_arr)}"
        )

    lengths = np.asarray([int(x.shape[0]) for x in seqs], dtype=np.int32)
    offsets = np.zeros(n + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths, dtype=np.int64)
    emg_concat = np.concatenate(seqs, axis=0).astype(np.float32, copy=False)

    if channels is None:
        channels_arr = np.asarray([f"EMG{i}" for i in range(n_channels)], dtype="U32")
    else:
        if len(channels) != n_channels:
            raise ValueError(f"channels length {len(channels)} does not match C={n_channels}")
        channels_arr = np.asarray(channels, dtype="U32")

    out = {
        "emg_concat": emg_concat,
        "offsets": offsets,
        "lengths": lengths,
        "target": target_arr,
        "subject": subject_arr,
        "channels": channels_arr,
    }
    if session_arr is not None:
        out["session"] = session_arr
    return out


def save_variable_length_npz(
    output_path: str | Path,
    packed: dict[str, np.ndarray],
    *,
    metadata: dict[str, Any] | None = None,
    metadata_json_path: str | Path | None = None,
) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **packed)

    if metadata is None:
        return

    if metadata_json_path is None:
        metadata_json_path = out.with_suffix(".metadata.json")
    meta_path = Path(metadata_json_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def _iter_packed_windows(packed: dict[str, np.ndarray]) -> Iterable[np.ndarray]:
    emg_concat = np.asarray(packed["emg_concat"], dtype=np.float32)
    offsets = np.asarray(packed["offsets"], dtype=np.int64)
    for i in range(len(offsets) - 1):
        s = int(offsets[i])
        e = int(offsets[i + 1])
        yield emg_concat[s:e]


def _apply_channel_variant_window(
    x: np.ndarray,
    variant: str,
    *,
    selected_channels: np.ndarray | None = None,
) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("window must have shape (T,C)")
    c = int(x.shape[1])

    if variant == "8-first":
        if c < 8:
            raise ValueError(f"8-first requires >=8 channels, got {c}")
        return x[:, :8]

    if variant == "8-selected":
        if c < 8:
            raise ValueError(f"8-selected requires >=8 channels, got {c}")
        if selected_channels is None:
            if c == 8:
                return x[:, :8]
            raise ValueError("selected_channels is required for 8-selected when channels > 8")
        idx = np.asarray(selected_channels, dtype=np.int64).reshape(-1)
        if idx.size != 8:
            raise ValueError(f"8-selected requires exactly 8 indices, got {idx.size}")
        return x[:, idx]

    if variant == "16-channel":
        if c == 16:
            return x
        if c > 16:
            return x[:, :16]
        pad = np.zeros((x.shape[0], 16 - c), dtype=x.dtype)
        return np.concatenate([x, pad], axis=1)

    raise ValueError(f"Unknown variant: {variant}")


def apply_channel_variant_packed(
    packed: dict[str, np.ndarray],
    variant: str,
    *,
    selected_channels: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    emg_list = [
        _apply_channel_variant_window(x, variant, selected_channels=selected_channels)
        for x in _iter_packed_windows(packed)
    ]

    in_channels = int(np.asarray(packed["emg_concat"]).shape[1])
    if variant == "8-first":
        info = {"selected_channels": list(range(8))}
    elif variant == "8-selected":
        if selected_channels is None and in_channels == 8:
            selected_channels = np.arange(8, dtype=np.int64)
        if selected_channels is None:
            raise ValueError("selected_channels is required for 8-selected")
        info = {"selected_channels": np.asarray(selected_channels, dtype=np.int64).tolist()}
    elif variant == "16-channel":
        info = {"selected_channels": list(range(min(in_channels, 16)))}
    else:
        raise ValueError(f"Unknown variant: {variant}")

    out_channels = int(emg_list[0].shape[1])
    channels = [f"EMG{i}" for i in range(out_channels)]
    out = pack_variable_length_trials(
        emg_list,
        np.asarray(packed["target"]),
        np.asarray(packed["subject"]),
        session=np.asarray(packed["session"]) if "session" in packed else None,
        channels=channels,
    )
    return out, info


def build_nm_variable_windows(
    dataset_root: str | Path,
    *,
    source_sfreq_hz: int = 2000,
    target_sfreq_hz: int = 1000,
    duration_q_low: float = 0.01,
    duration_q_high: float = 0.99,
    channel_indices: Iterable[int] | None = None,
    max_sessions: int | None = None,
    backend: str = "auto",
    skip_stub_sessions: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    sessions = discover_sessions(dataset_root)
    if max_sessions is not None:
        sessions = sessions[:max_sessions]

    all_durations: list[float] = []
    for session in sessions:
        events = filter_digit_prompts(read_events(session.events_tsv))
        for ev in events:
            d = float(ev.get("duration", np.nan))
            if np.isfinite(d) and d > 0:
                all_durations.append(d)

    if not all_durations:
        raise ValueError("No valid digit durations found in nm dataset")

    dur_arr = np.asarray(all_durations, dtype=np.float64)
    q_low = float(np.quantile(dur_arr, duration_q_low))
    q_high = float(np.quantile(dur_arr, duration_q_high))
    if q_low <= 0:
        q_low = float(np.min(dur_arr[dur_arr > 0]))
    if q_high < q_low:
        q_high = q_low

    emg_list: list[np.ndarray] = []
    targets: list[int] = []
    subjects: list[str] = []
    session_ids: list[str] = []

    processed_events = 0
    kept_events = 0
    skipped_oob = 0
    skipped_stub_sessions = 0
    skipped_invalid_duration = 0
    clipped_low = 0
    clipped_high = 0
    selected_channel_names: list[str] | None = None

    for idx, session in enumerate(sessions, start=1):
        events = filter_digit_prompts(read_events(session.events_tsv))
        processed_events += len(events)
        if not events:
            continue

        try:
            recording, sfreq, ch_names = _load_full_recording(session.bdf_file, backend=backend)
        except RuntimeError as exc:
            if skip_stub_sessions and "git-annex stub" in str(exc):
                skipped_stub_sessions += 1
                if verbose:
                    print(
                        f"[{idx:04d}/{len(sessions):04d}] {session.subject} {session.session} "
                        "skipped: BDF not downloaded yet (stub)"
                    )
                continue
            raise

        if int(round(sfreq)) != source_sfreq_hz:
            raise ValueError(
                f"Unexpected source sample rate in {session.bdf_file}: {sfreq} Hz "
                f"(expected {source_sfreq_hz} Hz)"
            )

        if channel_indices is None:
            selected_channel_names = ch_names
        else:
            idx_arr = np.asarray(list(channel_indices), dtype=np.int64)
            selected_channel_names = [ch_names[i] for i in idx_arr]

        if verbose:
            print(
                f"[{idx:04d}/{len(sessions):04d}] {session.subject} {session.session} "
                f"| digit events: {len(events)}"
            )

        for ev in events:
            raw_dur = float(ev.get("duration", np.nan))
            if not np.isfinite(raw_dur) or raw_dur <= 0:
                skipped_invalid_duration += 1
                continue

            dur = raw_dur
            if dur < q_low:
                dur = q_low
                clipped_low += 1
            elif dur > q_high:
                dur = q_high
                clipped_high += 1

            src_samples = max(1, int(round(dur * source_sfreq_hz)))
            tgt_samples = max(1, int(round(dur * target_sfreq_hz)))
            onset = int(ev["sample"])

            try:
                seg = load_emg_segment(
                    session.bdf_file,
                    onset_sample=onset,
                    duration_samples=src_samples,
                    target_num_samples=tgt_samples,
                    channel_indices=channel_indices,
                    backend=backend,
                    recording_data=recording,
                )
            except ValueError:
                skipped_oob += 1
                continue

            emg_list.append(seg)
            targets.append(int(ev["prompt_text"]))
            subjects.append(session.subject)
            session_ids.append(session.session)
            kept_events += 1

    if not emg_list:
        raise ValueError("No nm variable-length windows were built")

    lengths = np.asarray([x.shape[0] for x in emg_list], dtype=np.int32)
    return {
        "emg_list": emg_list,
        "target": np.asarray(targets, dtype=np.int16),
        "subject": np.asarray(subjects, dtype="U16"),
        "session": np.asarray(session_ids, dtype="U32"),
        "channels": selected_channel_names or [],
        "meta": {
            "source": "nm000106",
            "n_sessions": len(sessions),
            "processed_digit_events": processed_events,
            "kept_windows": kept_events,
            "skipped_out_of_bounds": skipped_oob,
            "skipped_stub_sessions": skipped_stub_sessions,
            "skipped_invalid_duration": skipped_invalid_duration,
            "clipped_low_count": clipped_low,
            "clipped_high_count": clipped_high,
            "source_sfreq_hz": source_sfreq_hz,
            "target_sfreq_hz": target_sfreq_hz,
            "duration_q_low": duration_q_low,
            "duration_q_high": duration_q_high,
            "duration_clip_sec": {"low": q_low, "high": q_high},
            "length_stats": summarize_lengths(lengths),
        },
    }


def build_old_variable_windows(
    recordings_root: str | Path,
    metadata_json_path: str | Path,
    *,
    sfreq_hz: int = 1000,
    min_samples: int = 5,
    channel_indices: Iterable[int] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    root = Path(recordings_root)
    meta_path = Path(metadata_json_path)
    if not root.exists():
        raise FileNotFoundError(f"Recordings root not found: {root}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata json not found: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    recs = meta.get("recordings", [])
    if not recs:
        raise ValueError(f"No recordings in metadata: {meta_path}")

    emg_list: list[np.ndarray] = []
    targets: list[int] = []
    subjects: list[str] = []

    processed_trials = 0
    skipped_no_onpaper = 0
    skipped_too_short = 0
    skipped_bad_target = 0

    for ridx, rec in enumerate(recs, start=1):
        filename = str(rec.get("filename", ""))
        subject = str(rec.get("subject", ""))
        if not filename:
            continue

        file_path = root / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing recording file: {file_path}")

        mat = loadmat(str(file_path), squeeze_me=True, struct_as_record=True)
        try:
            emg_full = np.asarray(mat["export_EMG"], dtype=np.float32)
            onpaper = np.asarray(mat["export_OnPaper"]) == 1
            trial = np.asarray(mat["export_Trial"])
            types = np.asarray(mat["export_Type"])
        except KeyError as exc:
            raise KeyError(f"Missing expected key in {file_path}: {exc}") from exc

        starts = np.r_[0, np.flatnonzero(trial[1:] != trial[:-1]) + 1]
        ends = np.r_[starts[1:], len(trial)]
        processed_trials += len(starts)

        if verbose:
            print(f"[{ridx:03d}/{len(recs):03d}] {filename} ({subject}) | trials: {len(starts)}")

        for s, e in zip(starts, ends):
            paper_idx = np.flatnonzero(onpaper[s:e])
            if paper_idx.size == 0:
                skipped_no_onpaper += 1
                continue

            # Strict OnPaper-only crop: no pre/post expansion.
            t_start = max(0, int(s + paper_idx[0]))
            t_end = min(len(trial), int(s + paper_idx[-1] + 1))
            if t_end <= t_start:
                skipped_too_short += 1
                continue

            seg = emg_full[t_start:t_end, :]
            if channel_indices is not None:
                idx_arr = np.asarray(list(channel_indices), dtype=np.int64)
                seg = seg[:, idx_arr]

            if seg.shape[0] < min_samples:
                skipped_too_short += 1
                continue

            t_raw = np.asarray(types[t_start:t_end]).astype(np.int64, copy=False)
            if t_raw.size == 0:
                skipped_bad_target += 1
                continue

            t_val = int(np.bincount(t_raw).argmax() - 1)
            if "15Nov07" in filename and t_val > 3:
                t_val += 1
            if t_val < 0 or t_val > 9:
                skipped_bad_target += 1
                continue

            emg_list.append(np.asarray(seg, dtype=np.float32))
            targets.append(t_val)
            subjects.append(subject)

    if not emg_list:
        raise ValueError("No old variable-length windows were built")

    c = int(emg_list[0].shape[1])
    channels = [f"EMG{i}" for i in range(c)]
    lengths = np.asarray([x.shape[0] for x in emg_list], dtype=np.int32)
    return {
        "emg_list": emg_list,
        "target": np.asarray(targets, dtype=np.int16),
        "subject": np.asarray(subjects, dtype="U16"),
        "channels": channels,
        "meta": {
            "source": "old_dataset",
            "recordings_root": str(root),
            "metadata_json_path": str(meta_path),
            "n_recordings": len(recs),
            "processed_trials": processed_trials,
            "kept_windows": int(len(emg_list)),
            "skipped_no_onpaper": skipped_no_onpaper,
            "skipped_too_short": skipped_too_short,
            "skipped_bad_target": skipped_bad_target,
            "sfreq_hz": sfreq_hz,
            "crop_rule": "onpaper_only",
            "length_stats": summarize_lengths(lengths),
        },
    }
