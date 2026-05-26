from __future__ import annotations

from pathlib import Path
import copy
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from shared_ml import get_window, load_model_class, macro_f1_score, set_seed


def make_model(ModelClass: Any, *, in_ch: int, n_classes: int, model_cfg: dict[str, Any]) -> nn.Module:
    return ModelClass(in_ch=in_ch, n_classes=n_classes, **model_cfg, variable_length=True)


def _to_numeric_digit_targets(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(y)

    if arr.dtype.kind in 'iu':
        keep = np.isin(arr.astype(np.int64), np.arange(10, dtype=np.int64))
        return arr.astype(np.int64), keep

    ys = arr.astype(str)
    mapped = np.full(len(ys), fill_value=-1, dtype=np.int64)
    keep = np.zeros(len(ys), dtype=bool)
    for i, token in enumerate(ys):
        t = token.strip().lower()
        if t.isdigit() and int(t) in range(10):
            mapped[i] = int(t)
            keep[i] = True
    return mapped, keep


def filter_digits_only(var_ds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    y_num, keep = _to_numeric_digit_targets(var_ds['target'])
    idx = np.where(keep)[0].astype(np.int64)
    if len(idx) == 0:
        raise RuntimeError('No digit samples left after filtering')

    windows = [get_window(var_ds, int(i)) for i in idx]
    offsets = [0]
    for w in windows:
        offsets.append(offsets[-1] + int(w.shape[0]))

    emg_concat = np.concatenate(windows, axis=0).astype(np.float32, copy=False)
    out = {
        'emg_concat': emg_concat,
        'offsets': np.asarray(offsets, dtype=np.int64),
        'lengths': np.asarray([w.shape[0] for w in windows], dtype=np.int32),
        'target': y_num[idx].astype(np.int64),
        'subject': np.asarray(var_ds.get('subject', np.array(['global'] * len(idx))), dtype=str)[idx],
        'session': np.asarray(var_ds.get('session', np.array([f'trial_{i:06d}' for i in range(len(var_ds['target']))], dtype=str)), dtype=str)[idx],
        'channels': np.asarray(var_ds.get('channels', np.array([], dtype=str))),
    }
    return out


def load_old_varlen_npz(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as d:
        ds = {k: d[k] for k in d.files}
    if 'session' not in ds:
        ds['session'] = np.asarray([f'trial_{i:06d}' for i in range(len(ds['target']))], dtype=str)
    return filter_digits_only(ds)


def load_nm_varlen_npz(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as d:
        ds = {k: d[k] for k in d.files}
    if 'session' not in ds:
        ds['session'] = np.asarray([f'trial_{i:06d}' for i in range(len(ds['target']))], dtype=str)
    return filter_digits_only(ds)


def load_hcmyo_postcrop_varlen(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as d:
        X = np.asarray(d['X'], dtype=np.float32)  # [S,N,C,T]
        y = np.asarray(d['y'])
        subjects = np.asarray(d['subjects']).astype(str)
        lens = np.asarray(d['lens'], dtype=np.int64)

    y_num, keep_any = _to_numeric_digit_targets(y.reshape(-1))
    if not np.any(keep_any):
        raise RuntimeError('No digit labels in HCMYO dataset')

    windows: list[np.ndarray] = []
    targets: list[int] = []
    subj_arr: list[str] = []
    sess_arr: list[str] = []
    lengths: list[int] = []

    n_subj, n_trials, _n_ch, max_t = X.shape
    flat_i = 0
    for si in range(n_subj):
        subj = str(subjects[si])
        for ti in range(n_trials):
            lbl = y_num[flat_i]
            keep = keep_any[flat_i]
            flat_i += 1
            if not keep:
                continue
            real_len = int(lens[si, ti])
            real_len = max(1, min(real_len, int(max_t)))
            x_ct = X[si, ti, :, :real_len]
            x_tc = np.transpose(x_ct, (1, 0))
            windows.append(x_tc.astype(np.float32, copy=False))
            targets.append(int(lbl))
            subj_arr.append(subj)
            sess_arr.append(f'{subj}_trial_{ti:05d}')
            lengths.append(real_len)

    offsets = [0]
    for w in windows:
        offsets.append(offsets[-1] + int(w.shape[0]))

    emg_concat = np.concatenate(windows, axis=0).astype(np.float32, copy=False)
    return {
        'emg_concat': emg_concat,
        'offsets': np.asarray(offsets, dtype=np.int64),
        'lengths': np.asarray(lengths, dtype=np.int32),
        'target': np.asarray(targets, dtype=np.int64),
        'subject': np.asarray(subj_arr, dtype=str),
        'session': np.asarray(sess_arr, dtype=str),
        'channels': np.asarray([f'ch{i}' for i in range(X.shape[2])], dtype=str),
    }


def stratified_split(y: np.ndarray, *, train_ratio: float, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.int64)
    rng = np.random.default_rng(int(seed))

    tr, va, te = [], [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0].astype(np.int64)
        idx = idx.copy()
        rng.shuffle(idx)

        n = len(idx)
        if n < 3:
            tr.extend(idx.tolist())
            continue

        n_tr = int(round(n * float(train_ratio)))
        n_tr = max(1, min(n - 2, n_tr))

        rem = n - n_tr
        val_ratio_cond = float(val_ratio) / max(1.0 - float(train_ratio), 1e-9)
        n_va = int(round(rem * val_ratio_cond))
        n_va = max(1, min(rem - 1, n_va))

        tr.extend(idx[:n_tr].tolist())
        va.extend(idx[n_tr:n_tr + n_va].tolist())
        te.extend(idx[n_tr + n_va:].tolist())

    return (
        np.asarray(sorted(tr), dtype=np.int64),
        np.asarray(sorted(va), dtype=np.int64),
        np.asarray(sorted(te), dtype=np.int64),
    )


def select_topk_pca_channels(
    var_ds: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    k: int,
    seed: int,
    max_windows: int,
    time_subsample: int,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    rng = np.random.default_rng(int(seed))
    if len(idx) > int(max_windows):
        idx = np.asarray(sorted(rng.choice(idx, size=int(max_windows), replace=False).tolist()), dtype=np.int64)

    parts = []
    step = max(1, int(time_subsample))
    for i in idx:
        x = get_window(var_ds, int(i))
        if len(x) >= 2:
            parts.append(x[::step])

    if not parts:
        raise RuntimeError('No windows for PCA channel selection')

    flat = np.concatenate(parts, axis=0).astype(np.float64, copy=False)
    flat = flat - flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    flat = flat / std

    _u, s, vh = np.linalg.svd(flat, full_matrices=False)
    n_comp = min(int(k), int(vh.shape[0]))
    comps = vh[:n_comp]
    ev = s[:n_comp] ** 2
    evr = ev / max(ev.sum(), 1e-12)
    scores = (np.abs(comps) * evr[:, None]).sum(axis=0)

    ch = np.argsort(scores)[-int(k):]
    return np.asarray(sorted(ch.tolist()), dtype=np.int64), scores.astype(np.float64)


def fit_norm_stats_global(
    var_ds: dict[str, np.ndarray],
    indices: np.ndarray,
    selected_channels: np.ndarray,
    *,
    clip: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    ch = np.asarray(selected_channels, dtype=np.int64)

    parts = []
    for i in idx:
        x = np.clip(get_window(var_ds, int(i))[:, ch], -clip, clip)
        parts.append(x)
    flat = np.concatenate(parts, axis=0).astype(np.float32, copy=False)

    center = np.median(flat, axis=0)
    scale = 1.4826 * np.median(np.abs(flat - center), axis=0)
    scale = np.maximum(scale, eps)
    return center.astype(np.float32), scale.astype(np.float32)


def build_cache(
    var_ds: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    selected_channels: np.ndarray,
    max_seq_len: int,
    clip: float,
    eps: float,
    norm_center: np.ndarray,
    norm_scale: np.ndarray,
) -> dict[str, Any]:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    y_all = np.asarray(var_ds['target'], dtype=np.int64)

    windows = []
    ys = []
    for i in idx:
        x = get_window(var_ds, int(i))[:, selected_channels]
        if x.shape[0] > int(max_seq_len):
            x = x[: int(max_seq_len)]
        x = np.clip(x, -clip, clip)
        x = ((x - norm_center[None, :]) / (norm_scale[None, :] + eps)).astype(np.float32, copy=False)
        windows.append(np.transpose(x, (1, 0)).astype(np.float32, copy=False))
        ys.append(int(y_all[int(i)]))

    return {
        'windows': windows,
        'y': np.asarray(ys, dtype=np.int64),
        'global_idx': idx,
        'in_ch': int(windows[0].shape[0]),
    }


def pad_windows(windows: list[np.ndarray], indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    idx = np.asarray(indices, dtype=np.int64)
    selected = [windows[int(i)] for i in idx]
    bsz = len(selected)
    ch = int(selected[0].shape[0])
    max_len = max(int(w.shape[1]) for w in selected)

    x_pad = torch.zeros((bsz, ch, max_len), dtype=torch.float32)
    pad_mask = torch.ones((bsz, max_len), dtype=torch.bool)
    for i, w in enumerate(selected):
        t = int(w.shape[1])
        x_pad[i, :, :t] = torch.from_numpy(w)
        pad_mask[i, :t] = False

    return x_pad, pad_mask


def iter_batches(n: int, batch_size: int, *, shuffle: bool, rng: np.random.Generator) -> list[np.ndarray]:
    idx = np.arange(n, dtype=np.int64)
    if shuffle:
        rng.shuffle(idx)
    return [idx[i:i + batch_size] for i in range(0, n, batch_size)]


def model_forward_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, pad_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = model(x, y, padding_mask=pad_mask)
    if isinstance(out, (tuple, list)) and len(out) == 3:
        logits, y_main, mix_info = out
        if mix_info is None:
            return logits, F.cross_entropy(logits, y_main)
        y2, lam = mix_info
        return logits, lam * F.cross_entropy(logits, y_main) + (1.0 - lam) * F.cross_entropy(logits, y2)

    logits = out[0] if isinstance(out, (tuple, list)) else out
    return logits, F.cross_entropy(logits, y)


@torch.no_grad()
def evaluate_cache(model: nn.Module, cache: dict[str, Any], *, device: str, batch_size: int) -> dict[str, float]:
    model.eval()
    n = len(cache['y'])
    rng = np.random.default_rng(0)

    total_loss = 0.0
    yt_all, yp_all = [], []

    for b in iter_batches(n, batch_size, shuffle=False, rng=rng):
        x, m = pad_windows(cache['windows'], b)
        x = x.to(device)
        m = m.to(device)
        y = torch.as_tensor(cache['y'][b], dtype=torch.long, device=device)

        out = model(x, padding_mask=m)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=1)

        total_loss += float(loss.item()) * len(b)
        yt_all.append(y.detach().cpu().numpy())
        yp_all.append(pred.detach().cpu().numpy())

    yt = np.concatenate(yt_all) if yt_all else np.array([], dtype=np.int64)
    yp = np.concatenate(yp_all) if yp_all else np.array([], dtype=np.int64)

    return {
        'loss': total_loss / max(n, 1),
        'acc': float(np.mean(yt == yp)) if len(yt) else float('nan'),
        'f1_macro': macro_f1_score(yt, yp) if len(yt) else float('nan'),
    }


def fit_ce(
    model: nn.Module,
    *,
    train_cache: dict[str, Any],
    val_cache: dict[str, Any],
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    batch_size: int,
    seed: int,
    grad_clip: float = 1.0,
    verbose: bool = False,
    progress_prefix: str = '',
) -> tuple[nn.Module, dict[str, Any]]:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -float('inf')
    best_epoch = 0
    bad = 0
    history = []

    rng = np.random.default_rng(int(seed))
    n_train = len(train_cache['y'])

    for epoch in range(1, int(epochs) + 1):
        model.train()
        tr_loss, tr_corr, tr_total = 0.0, 0, 0

        for b in iter_batches(n_train, int(batch_size), shuffle=True, rng=rng):
            x, m = pad_windows(train_cache['windows'], b)
            x = x.to(device)
            m = m.to(device)
            y = torch.as_tensor(train_cache['y'][b], dtype=torch.long, device=device)

            opt.zero_grad(set_to_none=True)
            logits, loss = model_forward_loss(model, x, y, m)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()

            pred = logits.argmax(dim=1)
            tr_corr += int((pred == y).sum().item())
            tr_total += int(len(b))
            tr_loss += float(loss.item()) * int(len(b))

        val = evaluate_cache(model, val_cache, device=device, batch_size=int(batch_size))
        row = {
            'epoch': int(epoch),
            'train_loss': float(tr_loss / max(tr_total, 1)),
            'train_acc': float(tr_corr / max(tr_total, 1)),
            'val_loss': float(val['loss']),
            'val_acc': float(val['acc']),
            'val_f1_macro': float(val['f1_macro']),
        }
        history.append(row)

        if bool(verbose):
            prefix = f"{progress_prefix} " if progress_prefix else ""
            print(
                f"{prefix}epoch {epoch:03d}/{int(epochs):03d} | "
                f"train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.4f} | "
                f"val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.4f} val_f1={row['val_f1_macro']:.4f} | "
                f"best_f1={best_f1 if np.isfinite(best_f1) else float('nan'):.4f}",
                flush=True,
            )

        cur = float(row['val_f1_macro'])
        if np.isfinite(cur) and cur > best_f1:
            best_f1 = cur
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= int(patience):
                break

    model.load_state_dict(best_state)
    return model, {
        'best_epoch': int(best_epoch),
        'best_val_f1_macro': float(best_f1),
        'history': history,
    }


def prepare_dataset_splits_and_cache(
    ds: dict[str, np.ndarray],
    *,
    seed: int,
    pca_k: int,
    max_pca_windows: int,
    pca_time_subsample: int,
    norm_clip: float,
    norm_eps: float,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, Any]:
    y = np.asarray(ds['target'], dtype=np.int64)
    tr_idx, va_idx, te_idx = stratified_split(y, train_ratio=float(train_ratio), val_ratio=float(val_ratio), seed=int(seed))

    n_ch = int(ds['emg_concat'].shape[1])
    if n_ch == 16:
        selected_ch, pca_scores = select_topk_pca_channels(
            ds,
            tr_idx,
            k=int(pca_k),
            seed=int(seed),
            max_windows=int(max_pca_windows),
            time_subsample=int(pca_time_subsample),
        )
    else:
        selected_ch = np.arange(min(8, n_ch), dtype=np.int64)
        pca_scores = np.full((n_ch,), np.nan, dtype=np.float64)

    center, scale = fit_norm_stats_global(
        ds,
        tr_idx,
        selected_ch,
        clip=float(norm_clip),
        eps=float(norm_eps),
    )

    len_cap = int(round(float(np.asarray(ds['lengths'][tr_idx], dtype=np.float64).mean())))
    len_cap = max(1, len_cap)

    tr_cache = build_cache(
        ds,
        tr_idx,
        selected_channels=selected_ch,
        max_seq_len=len_cap,
        clip=float(norm_clip),
        eps=float(norm_eps),
        norm_center=center,
        norm_scale=scale,
    )
    va_cache = build_cache(
        ds,
        va_idx,
        selected_channels=selected_ch,
        max_seq_len=len_cap,
        clip=float(norm_clip),
        eps=float(norm_eps),
        norm_center=center,
        norm_scale=scale,
    )
    te_cache = build_cache(
        ds,
        te_idx,
        selected_channels=selected_ch,
        max_seq_len=len_cap,
        clip=float(norm_clip),
        eps=float(norm_eps),
        norm_center=center,
        norm_scale=scale,
    )

    return {
        'train_idx': tr_idx,
        'val_idx': va_idx,
        'test_idx': te_idx,
        'selected_channels': selected_ch,
        'pca_scores': pca_scores,
        'norm_center': center,
        'norm_scale': scale,
        'len_cap': int(len_cap),
        'train_cache': tr_cache,
        'val_cache': va_cache,
        'test_cache': te_cache,
    }


def experiment_grid(dataset_names: list[str]) -> list[dict[str, str]]:
    names = list(dataset_names)
    out = []
    for src in names:
        for ft in names:
            if ft == src:
                continue
            unseen = [x for x in names if x not in {src, ft}]
            if len(unseen) != 1:
                continue
            out.append({'pretrain_ds': src, 'finetune_ds': ft, 'unseen_ds': unseen[0]})
    return out


def mstd(series: pd.Series) -> dict[str, float]:
    arr = np.asarray(series, dtype=np.float64)
    return {'mean': float(np.nanmean(arr)), 'std': float(np.nanstd(arr))}
