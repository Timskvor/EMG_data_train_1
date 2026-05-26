from __future__ import annotations

from pathlib import Path
import json
import copy
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from shared_ml import get_window, load_model_class, macro_f1_score, set_seed


class SubjectVarlenNormalizer:
    def __init__(self, *, clip: float = 6.0, method: str = 'robust', eps: float = 1e-6):
        if method not in {'robust', 'zscore'}:
            raise ValueError("method must be one of {'robust', 'zscore'}")
        self.clip = float(clip)
        self.method = method
        self.eps = float(eps)
        self.subject_stats_: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.global_stats_: tuple[np.ndarray, np.ndarray] | None = None

    def _estimate_from_indices(self, var_ds: dict[str, np.ndarray], indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        parts: list[np.ndarray] = []
        for i in idx:
            x = np.clip(get_window(var_ds, int(i)), -self.clip, self.clip)
            parts.append(x)
        if not parts:
            raise ValueError('No windows to estimate stats')
        flat = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        if self.method == 'zscore':
            center = flat.mean(axis=0)
            scale = flat.std(axis=0)
        else:
            center = np.median(flat, axis=0)
            scale = 1.4826 * np.median(np.abs(flat - center), axis=0)
        scale = np.maximum(scale, self.eps)
        return center.astype(np.float32), scale.astype(np.float32)

    def fit(self, var_ds: dict[str, np.ndarray], indices: np.ndarray) -> 'SubjectVarlenNormalizer':
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        subjects = np.asarray(var_ds['subject'])[idx].astype(str)
        self.subject_stats_.clear()
        for subj in np.unique(subjects):
            subj_idx = idx[subjects == subj]
            self.subject_stats_[str(subj)] = self._estimate_from_indices(var_ds, subj_idx)
        self.global_stats_ = self._estimate_from_indices(var_ds, idx)
        return self

    def get_stats_for_subject(
        self,
        subj: str,
        *,
        var_ds: dict[str, np.ndarray] | None = None,
        indices_for_subject: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        key = str(subj)
        if key in self.subject_stats_:
            return self.subject_stats_[key]
        if var_ds is not None and indices_for_subject is not None and len(indices_for_subject) > 0:
            return self._estimate_from_indices(var_ds, indices_for_subject)
        if self.global_stats_ is None:
            raise RuntimeError('Normalizer not fitted')
        return self.global_stats_

    def transform_window(self, x: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        x = np.clip(x, -self.clip, self.clip)
        return ((x - center[None, :]) / (scale[None, :] + self.eps)).astype(np.float32, copy=False)


def build_varlen_cache(
    var_ds: dict[str, np.ndarray],
    indices: np.ndarray,
    normalizer: SubjectVarlenNormalizer,
    *,
    fit_unknown: bool,
    max_seq_len: int | None = None,
    subject_stats_override: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    subjects_full = np.asarray(var_ds['subject']).astype(str)
    y_full = np.asarray(var_ds['target']).astype(np.int64)
    sessions_full = np.asarray(var_ds['session']).astype(str) if 'session' in var_ds else np.asarray([''] * len(y_full), dtype=str)

    override = subject_stats_override or {}
    unknown_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    used_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    used_stats_source: dict[str, str] = {}

    windows: list[np.ndarray] = []
    ys: list[int] = []
    subs: list[str] = []
    sess: list[str] = []

    local_subjects = subjects_full[idx]

    for global_i in idx:
        subj = str(subjects_full[global_i])
        if subj in override:
            center, scale = override[subj]
            stats_source = 'override'
        elif subj in normalizer.subject_stats_:
            center, scale = normalizer.subject_stats_[subj]
            stats_source = 'train_subject'
        else:
            if not fit_unknown:
                center, scale = normalizer.get_stats_for_subject(subj)
                stats_source = 'global_fallback'
            else:
                if subj not in unknown_stats:
                    subj_idx = idx[local_subjects == subj]
                    unknown_stats[subj] = normalizer.get_stats_for_subject(
                        subj,
                        var_ds=var_ds,
                        indices_for_subject=subj_idx,
                    )
                center, scale = unknown_stats[subj]
                stats_source = 'fit_unknown_local'

        used_stats[subj] = (center, scale)
        used_stats_source[subj] = stats_source

        x = get_window(var_ds, int(global_i))
        if max_seq_len is not None and int(max_seq_len) > 0 and x.shape[0] > int(max_seq_len):
            x = x[: int(max_seq_len)]
        x = normalizer.transform_window(x, center, scale)

        windows.append(np.transpose(x, (1, 0)).astype(np.float32, copy=False))  # [C, T]
        ys.append(int(y_full[global_i]))
        subs.append(subj)
        sess.append(str(sessions_full[global_i]))

    if not windows:
        raise RuntimeError('No windows in cache')

    return {
        'windows': windows,
        'y': np.asarray(ys, dtype=np.int64),
        'subjects': np.asarray(subs, dtype=str),
        'sessions': np.asarray(sess, dtype=str),
        'global_idx': idx,
        'in_ch': int(windows[0].shape[0]),
        'max_seq_len': None if max_seq_len is None else int(max_seq_len),
        'subject_stats': used_stats,
        'subject_stats_source': used_stats_source,
    }


class SessionFold:
    def __init__(
        self,
        *,
        fold_id: int,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
        train_subjects: list[str],
        val_subjects: list[str],
        test_subjects: list[str],
    ) -> None:
        self.fold_id = int(fold_id)
        self.train_idx = np.asarray(train_idx, dtype=np.int64)
        self.val_idx = np.asarray(val_idx, dtype=np.int64)
        self.test_idx = np.asarray(test_idx, dtype=np.int64)
        self.train_subjects = [str(s) for s in train_subjects]
        self.val_subjects = [str(s) for s in val_subjects]
        self.test_subjects = [str(s) for s in test_subjects]


def get_subject_session_counts(subjects: np.ndarray, sessions: np.ndarray) -> dict[str, int]:
    subj = np.asarray(subjects).reshape(-1).astype(str)
    sess = np.asarray(sessions).reshape(-1).astype(str)
    out: dict[str, int] = {}
    for s in sorted(np.unique(subj).tolist()):
        out[s] = int(len(np.unique(sess[subj == s])))
    return out


def make_session_aware_subject_folds(
    subjects: np.ndarray,
    sessions: np.ndarray,
    *,
    n_folds: int,
    n_val_subjects: int,
    seed: int,
    min_eval_sessions: int,
    test_subject_exclude_prefixes: list[str] | None = None,
) -> tuple[list[Any], dict[str, int], list[str]]:
    subj = np.asarray(subjects).reshape(-1).astype(str)
    sess = np.asarray(sessions).reshape(-1).astype(str)

    uniq_all = sorted(np.unique(subj).tolist())
    sess_counts = get_subject_session_counts(subj, sess)

    eligible_eval = sorted([s for s in uniq_all if int(sess_counts.get(s, 0)) >= int(min_eval_sessions)])
    if len(eligible_eval) < int(n_val_subjects) + 1:
        raise RuntimeError(
            f'Not enough subjects with >= {min_eval_sessions} sessions for val/test: {len(eligible_eval)} found'
        )

    excluded_prefixes = [str(p) for p in (test_subject_exclude_prefixes or []) if str(p)]
    test_candidates = [s for s in eligible_eval if not any(s.startswith(p) for p in excluded_prefixes)]
    if len(test_candidates) < int(n_folds):
        raise RuntimeError(
            f'Not enough eligible test subjects: need={n_folds}, available={len(test_candidates)}'
        )

    rng = np.random.default_rng(int(seed))
    test_candidates = list(test_candidates)
    rng.shuffle(test_candidates)
    chosen_tests = test_candidates[: int(n_folds)]

    folds: list[Any] = []
    for fold_id, test_subj in enumerate(chosen_tests, start=1):
        remaining_eval = [s for s in eligible_eval if s != test_subj]
        remaining_eval = list(remaining_eval)
        rng.shuffle(remaining_eval)

        val_subjects = sorted(remaining_eval[: int(n_val_subjects)])
        train_subjects = sorted([s for s in uniq_all if s not in set(val_subjects) and s != test_subj])

        train_idx = np.where(np.isin(subj, train_subjects))[0].astype(np.int64)
        val_idx = np.where(np.isin(subj, val_subjects))[0].astype(np.int64)
        test_idx = np.where(subj == test_subj)[0].astype(np.int64)

        folds.append(
            SessionFold(
                fold_id=int(fold_id),
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                train_subjects=train_subjects,
                val_subjects=val_subjects,
                test_subjects=[str(test_subj)],
            )
        )

    return folds, sess_counts, eligible_eval


def split_subject_adapt_eval_by_sessions(
    sessions: np.ndarray,
    indices: np.ndarray,
    *,
    seed: int,
    ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    sess_all = np.asarray(sessions).reshape(-1).astype(str)
    if len(idx) == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64), {
            'n_sessions_total': 0,
            'n_sessions_adapt': 0,
            'n_sessions_eval': 0,
            'adapt_sessions': [],
            'eval_sessions': [],
        }

    subj_sessions = sess_all[idx]
    uniq = sorted(np.unique(subj_sessions).tolist())
    if len(uniq) < 2:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64), {
            'n_sessions_total': int(len(uniq)),
            'n_sessions_adapt': 0,
            'n_sessions_eval': 0,
            'adapt_sessions': [],
            'eval_sessions': [],
        }

    rng = np.random.default_rng(int(seed))
    uniq_shuf = np.asarray(uniq, dtype=object).copy()
    rng.shuffle(uniq_shuf)

    n_adapt_sessions = int(round(len(uniq_shuf) * float(ratio)))
    n_adapt_sessions = max(1, min(len(uniq_shuf) - 1, n_adapt_sessions))

    adapt_sessions = [str(s) for s in uniq_shuf[:n_adapt_sessions].tolist()]
    eval_sessions = [str(s) for s in uniq_shuf[n_adapt_sessions:].tolist()]

    is_adapt = np.isin(subj_sessions, adapt_sessions)
    adapt_idx = idx[is_adapt]
    eval_idx = idx[~is_adapt]

    info = {
        'n_sessions_total': int(len(uniq_shuf)),
        'n_sessions_adapt': int(len(adapt_sessions)),
        'n_sessions_eval': int(len(eval_sessions)),
        'adapt_sessions': sorted(adapt_sessions),
        'eval_sessions': sorted(eval_sessions),
    }

    return np.asarray(sorted(adapt_idx.tolist()), dtype=np.int64), np.asarray(sorted(eval_idx.tolist()), dtype=np.int64), info

def pad_windows(windows: list[np.ndarray], indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    idx = np.asarray(indices, dtype=np.int64)
    selected = [windows[int(i)] for i in idx]
    if not selected:
        raise ValueError('pad_windows received empty indices')

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


def model_logits(model: nn.Module, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    out = model(x, padding_mask=pad_mask)
    return out[0] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def evaluate_classifier_cache(
    model: nn.Module,
    cache: dict[str, Any],
    *,
    device: str,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    n = len(cache['y'])
    rng = np.random.default_rng(0)

    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    total_loss = 0.0

    for batch in iter_batches(n, batch_size, shuffle=False, rng=rng):
        x, m = pad_windows(cache['windows'], batch)
        x = x.to(device)
        m = m.to(device)
        y = torch.as_tensor(cache['y'][batch], dtype=torch.long, device=device)

        logits = model_logits(model, x, m)
        loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=1)

        total_loss += float(loss.item()) * len(batch)
        y_true_all.append(y.detach().cpu().numpy())
        y_pred_all.append(pred.detach().cpu().numpy())

    yt = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=np.int64)
    yp = np.concatenate(y_pred_all) if y_pred_all else np.array([], dtype=np.int64)

    return {
        'loss': total_loss / max(n, 1),
        'acc': float(np.mean(yt == yp)) if len(yt) else 0.0,
        'f1_macro': macro_f1_score(yt, yp),
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {'loss': float('nan'), 'acc': float('nan'), 'f1_macro': float('nan')}

    def _nanmean(key: str) -> float:
        vals = np.asarray([float(r.get(key, float('nan'))) for r in rows], dtype=np.float64)
        finite = np.isfinite(vals)
        if not np.any(finite):
            return float('nan')
        return float(np.mean(vals[finite]))

    return {
        'loss': _nanmean('loss'),
        'acc': _nanmean('acc'),
        'f1_macro': _nanmean('f1_macro'),
    }


def evaluate_adapted_val_subjects(
    base_model: nn.Module,
    *,
    val_subject_order: list[str],
    val_subject_caches: dict[str, dict[str, dict[str, Any]]],
    meta_adapt_cfg: dict[str, Any],
    monitor_adapt_episodes: int,
    device: str,
    batch_size: int,
    seed: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows_adapt_applied: list[dict[str, float]] = []
    setups: list[dict[str, Any]] = []

    monitor_cfg = dict(meta_adapt_cfg)
    monitor_cfg['episodes'] = int(monitor_adapt_episodes)

    for i, subj in enumerate(val_subject_order):
        subj = str(subj)
        adapted_model, setup = meta_adapt_on_subject(
            base_model,
            val_subject_caches[subj]['adapt'],
            cfg=monitor_cfg,
            device=device,
            seed=int(seed) + 1000 + i,
        )
        post = evaluate_subject_proto(
            adapted_model,
            support_cache=val_subject_caches[subj]['adapt'],
            query_cache=val_subject_caches[subj]['eval_adapted'],
            device=device,
            emb_norm=bool(meta_adapt_cfg['embedding_norm']),
            batch_size=int(batch_size),
        )
        adapt_applied = bool(setup.get('adapt_applied', False))
        valid_proto_eval = float(post.get('valid', 0.0)) > 0.5
        if adapt_applied and valid_proto_eval:
            rows_adapt_applied.append(post)
        setups.append({'subject': subj, **setup, 'valid': float(post.get('valid', 0.0))})

    if rows_adapt_applied:
        return mean_metrics(rows_adapt_applied), setups
    return {'loss': float('nan'), 'acc': float('nan'), 'f1_macro': float('nan')}, setups


def fit_stageA_classifier(
    model: nn.Module,
    *,
    train_cache: dict[str, Any],
    val_subject_order: list[str],
    val_subject_caches: dict[str, dict[str, dict[str, Any]]],
    cfg: dict[str, Any],
    meta_adapt_cfg: dict[str, Any],
    device: str,
    seed: int,
    fold_label: str,
) -> tuple[nn.Module, dict[str, Any]]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['lr']), weight_decay=float(cfg['weight_decay']))

    monitor_mode = str(cfg['monitor_mode'])
    if monitor_mode not in {'max', 'min'}:
        raise ValueError('monitor_mode must be max/min')

    best_state = copy.deepcopy(model.state_dict())
    best_score = -float('inf') if monitor_mode == 'max' else float('inf')
    best_epoch = 0
    bad_epochs = 0

    history: list[dict[str, float]] = []
    rng = np.random.default_rng(seed)

    n_train = len(train_cache['y'])

    total_epochs = int(cfg['epochs'])
    for epoch in range(1, total_epochs + 1):
        model.train()

        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0

        for batch in iter_batches(n_train, int(cfg['batch_size']), shuffle=True, rng=rng):
            x, m = pad_windows(train_cache['windows'], batch)
            x = x.to(device)
            m = m.to(device)
            y = torch.as_tensor(train_cache['y'][batch], dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            logits = model_logits(model, x, m)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg['grad_clip']))
            optimizer.step()

            pred = logits.argmax(dim=1)
            tr_correct += int((pred == y).sum().item())
            tr_total += int(len(batch))
            tr_loss += float(loss.item()) * int(len(batch))

        train_loss = tr_loss / max(tr_total, 1)
        train_acc = tr_correct / max(tr_total, 1)

        val_zero_rows = [
            evaluate_classifier_cache(model, val_subject_caches[str(s)]['eval_zero'], device=device, batch_size=int(cfg['batch_size']))
            for s in val_subject_order
        ]
        val_zero = mean_metrics(val_zero_rows)

        val_adapt_monitor, _monitor_setups = evaluate_adapted_val_subjects(
            model,
            val_subject_order=val_subject_order,
            val_subject_caches=val_subject_caches,
            meta_adapt_cfg=meta_adapt_cfg,
            monitor_adapt_episodes=int(cfg.get('monitor_adapt_episodes', meta_adapt_cfg.get('episodes', 20))),
            device=device,
            batch_size=int(cfg['batch_size']),
            seed=int(seed) + 10000 + epoch,
        )

        row = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'val_zero_shot_loss': float(val_zero['loss']),
            'val_zero_shot_acc': float(val_zero['acc']),
            'val_zero_shot_f1': float(val_zero['f1_macro']),
            'val_adapt_monitor_loss': float(val_adapt_monitor['loss']),
            'val_adapt_monitor_acc': float(val_adapt_monitor['acc']),
            'val_adapt_monitor_f1': float(val_adapt_monitor['f1_macro']),
        }
        history.append(row)

        print(
            f"[{fold_label}] epoch={epoch:03d}/{total_epochs:03d} "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.4f} "
            f"val0_f1={row['val_zero_shot_f1']:.4f} valAmon_f1={row['val_adapt_monitor_f1']:.4f}",
            flush=True,
        )
        current = float(row['val_adapt_monitor_f1'])
        if not np.isfinite(current):
            current = -1e9 if monitor_mode == 'max' else 1e9

        improved = current > best_score if monitor_mode == 'max' else current < best_score
        if best_epoch == 0:
            improved = True

        if improved:
            best_score = current
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(cfg['patience']):
                print(
                    f"[{fold_label}] StageA early stop at epoch={epoch}, "
                    f"best_epoch={best_epoch}, best_val_adapt_monitor_f1={best_score:.4f}",
                    flush=True,
                )
                break

    model.load_state_dict(best_state)
    return model, {
        'best_epoch_stageA': int(best_epoch),
        'best_val_adapt_monitor_f1': float(best_score),
        'history': history,
    }

def encoder_forward(model: nn.Module, x: torch.Tensor, pad_mask: torch.Tensor | None, *, emb_norm: bool) -> torch.Tensor:
    z = model.pre(x)
    z = model.cnn(z)
    z = z.transpose(1, 2)

    tr_mask = model._downsample_padding_mask(pad_mask, out_len=z.size(1))

    z = model.pos(z)
    z = model.tr(z, src_key_padding_mask=tr_mask)
    z = model.norm(z)

    if tr_mask is None:
        pooled = z.mean(dim=1)
    else:
        valid = (~tr_mask).unsqueeze(-1).to(dtype=z.dtype)
        pooled = (z * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    if emb_norm:
        pooled = F.normalize(pooled, p=2, dim=1)
    return pooled


def prototypical_logits(s_emb: torch.Tensor, s_y: torch.Tensor, q_emb: torch.Tensor) -> torch.Tensor:
    n_way = int(s_y.max().item()) + 1
    protos = []
    for c in range(n_way):
        protos.append(s_emb[s_y == c].mean(dim=0))
    proto = torch.stack(protos, dim=0)
    d2 = torch.cdist(q_emb, proto, p=2).pow(2)
    return -d2


def build_class_index(cache: dict[str, Any]) -> dict[int, np.ndarray]:
    y = np.asarray(cache['y'], dtype=np.int64)
    out: dict[int, np.ndarray] = {}
    for cls in np.unique(y):
        out[int(cls)] = np.where(y == cls)[0].astype(np.int64)
    return out


def find_feasible_meta_setup_single(
    cache: dict[str, Any],
    *,
    target_n_way: int,
    target_k_shot: int,
    target_q_query: int,
    min_n_way: int = 2,
) -> dict[str, Any]:
    cls_idx = build_class_index(cache)

    best: dict[str, Any] | None = None
    for q_query in range(int(target_q_query), 0, -1):
        for k_shot in range(int(target_k_shot), 0, -1):
            eligible = [c for c, idx in cls_idx.items() if len(idx) >= (k_shot + q_query)]
            n_way = min(int(target_n_way), len(eligible))

            cand = {
                'n_way': int(n_way),
                'k_shot': int(k_shot),
                'q_query': int(q_query),
                'eligible_classes': sorted(eligible),
            }
            if best is None or (cand['n_way'], cand['q_query'], cand['k_shot']) > (best['n_way'], best['q_query'], best['k_shot']):
                best = cand

            if n_way >= int(min_n_way):
                return cand

    if best is None:
        raise RuntimeError('No meta setup candidate found')
    return best


def sample_episode_single(
    class_to_idx: dict[int, np.ndarray],
    *,
    n_way: int,
    k_shot: int,
    q_query: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    eligible = [c for c, idx in class_to_idx.items() if len(idx) >= (k_shot + q_query)]
    if len(eligible) < n_way:
        raise RuntimeError('Not enough eligible classes for episode')

    chosen = rng.choice(np.asarray(eligible, dtype=np.int64), size=n_way, replace=False)

    s_idx: list[int] = []
    q_idx: list[int] = []
    s_y: list[int] = []
    q_y: list[int] = []

    for local_c, global_c in enumerate(chosen.tolist()):
        pool = class_to_idx[int(global_c)].copy()
        rng.shuffle(pool)
        s_pick = pool[:k_shot]
        q_pick = pool[k_shot:k_shot + q_query]

        s_idx.extend(s_pick.tolist())
        q_idx.extend(q_pick.tolist())
        s_y.extend([local_c] * k_shot)
        q_y.extend([local_c] * q_query)

    return {
        'support_idx': np.asarray(s_idx, dtype=np.int64),
        'query_idx': np.asarray(q_idx, dtype=np.int64),
        'support_y': np.asarray(s_y, dtype=np.int64),
        'query_y': np.asarray(q_y, dtype=np.int64),
        'episode_classes': np.asarray(chosen, dtype=np.int64),
    }


def meta_adapt_on_subject(
    base_model: nn.Module,
    adapt_cache: dict[str, Any],
    *,
    cfg: dict[str, Any],
    device: str,
    seed: int,
) -> tuple[nn.Module, dict[str, Any]]:
    model = copy.deepcopy(base_model).to(device)
    model.train()

    setup = find_feasible_meta_setup_single(
        adapt_cache,
        target_n_way=int(cfg['n_way']),
        target_k_shot=int(cfg['k_shot']),
        target_q_query=int(cfg['q_query']),
        min_n_way=2,
    )

    if int(setup['n_way']) < 2:
        return model, {**setup, 'episodes_done': 0, 'adapt_applied': False, 'skip_reason': 'n_way<2'}

    n_adapt = int(len(adapt_cache['y']))
    min_adapt = int(cfg.get('min_adapt_samples_for_train', 25))
    min_q = int(cfg.get('min_q_query_for_train', 2))

    if n_adapt < min_adapt:
        return model, {**setup, 'episodes_done': 0, 'adapt_applied': False, 'skip_reason': f'adapt_samples<{min_adapt}'}

    if int(setup['q_query']) < min_q:
        return model, {**setup, 'episodes_done': 0, 'adapt_applied': False, 'skip_reason': f'q_query<{min_q}'}

    class_to_idx = build_class_index(adapt_cache)
    rng = np.random.default_rng(seed)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg['lr']),
        weight_decay=float(cfg['weight_decay']),
    )

    for _ in range(int(cfg['episodes'])):
        ep = sample_episode_single(
            class_to_idx,
            n_way=int(setup['n_way']),
            k_shot=int(setup['k_shot']),
            q_query=int(setup['q_query']),
            rng=rng,
        )

        sx, sm = pad_windows(adapt_cache['windows'], ep['support_idx'])
        qx, qm = pad_windows(adapt_cache['windows'], ep['query_idx'])
        sx = sx.to(device)
        sm = sm.to(device)
        qx = qx.to(device)
        qm = qm.to(device)

        sy = torch.as_tensor(ep['support_y'], dtype=torch.long, device=device)
        qy = torch.as_tensor(ep['query_y'], dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)
        s_emb = encoder_forward(model, sx, sm, emb_norm=bool(cfg['embedding_norm']))
        q_emb = encoder_forward(model, qx, qm, emb_norm=bool(cfg['embedding_norm']))
        logits = prototypical_logits(s_emb, sy, q_emb)
        loss = F.cross_entropy(logits, qy)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(cfg['grad_clip']))
        optimizer.step()

    return model, {**setup, 'episodes_done': int(cfg['episodes']), 'adapt_applied': True, 'skip_reason': ''}


@torch.no_grad()
def evaluate_subject_proto(
    model: nn.Module,
    *,
    support_cache: dict[str, Any],
    query_cache: dict[str, Any],
    device: str,
    emb_norm: bool,
    batch_size: int,
) -> dict[str, float]:
    model.eval()

    s_class_idx = build_class_index(support_cache)
    proto_classes = sorted(s_class_idx.keys())
    if len(proto_classes) < 2:
        return {
            'loss': float('nan'),
            'acc': float('nan'),
            'f1_macro': float('nan'),
            'valid_query_count': 0.0,
            'valid': 0.0,
        }

    class_to_local = {c: i for i, c in enumerate(proto_classes)}

    s_emb_chunks: list[np.ndarray] = []
    s_lab_chunks: list[np.ndarray] = []
    n_support = len(support_cache['y'])
    rng = np.random.default_rng(0)

    for batch in iter_batches(n_support, batch_size, shuffle=False, rng=rng):
        x, m = pad_windows(support_cache['windows'], batch)
        x = x.to(device)
        m = m.to(device)
        emb = encoder_forward(model, x, m, emb_norm=emb_norm)
        s_emb_chunks.append(emb.detach().cpu().numpy())
        s_lab_chunks.append(np.asarray(support_cache['y'][batch], dtype=np.int64))

    if not s_emb_chunks:
        return {
            'loss': float('nan'),
            'acc': float('nan'),
            'f1_macro': float('nan'),
            'valid_query_count': 0.0,
            'valid': 0.0,
        }

    s_emb_np = np.concatenate(s_emb_chunks, axis=0)
    s_y_np = np.concatenate(s_lab_chunks, axis=0)

    protos = []
    for cls in proto_classes:
        protos.append(s_emb_np[s_y_np == cls].mean(axis=0))
    proto_np = np.stack(protos, axis=0)

    q_true_all: list[int] = []
    q_pred_all: list[int] = []
    losses: list[float] = []

    n_query = len(query_cache['y'])
    query_total = 0
    query_covered = 0
    local_to_class = np.asarray(proto_classes, dtype=np.int64)
    for batch in iter_batches(n_query, batch_size, shuffle=False, rng=rng):
        q_labels_global = np.asarray(query_cache['y'][batch], dtype=np.int64)
        x, m = pad_windows(query_cache['windows'], batch)
        x = x.to(device)
        m = m.to(device)
        q_emb = encoder_forward(model, x, m, emb_norm=emb_norm).detach().cpu().numpy()

        d2 = ((q_emb[:, None, :] - proto_np[None, :, :]) ** 2).sum(axis=2)
        logits_np = -d2

        pred_local = logits_np.argmax(axis=1).astype(np.int64)
        pred_global = local_to_class[pred_local]

        cover_mask = np.array([lbl in class_to_local for lbl in q_labels_global], dtype=bool)
        if np.any(cover_mask):
            y_local = np.array([class_to_local[int(v)] for v in q_labels_global[cover_mask]], dtype=np.int64)
            logits_t = torch.from_numpy(logits_np[cover_mask].astype(np.float32))
            y_t = torch.from_numpy(y_local.astype(np.int64))
            loss = F.cross_entropy(logits_t, y_t)
            losses.append(float(loss.item()))
            query_covered += int(np.sum(cover_mask))

        q_true_all.extend(q_labels_global.tolist())
        q_pred_all.extend(pred_global.tolist())
        query_total += int(len(q_labels_global))

    if query_total == 0:
        return {
            'loss': float('nan'),
            'acc': float('nan'),
            'f1_macro': float('nan'),
            'valid_query_count': 0.0,
            'query_total': 0.0,
            'query_covered': 0.0,
            'query_coverage': 0.0,
            'valid': 0.0,
        }

    yt = np.asarray(q_true_all, dtype=np.int64)
    yp = np.asarray(q_pred_all, dtype=np.int64)

    return {
        'loss': float(np.mean(losses)) if losses else float('nan'),
        'acc': float(np.mean(yt == yp)) if len(yt) else float('nan'),
        'f1_macro': macro_f1_score(yt, yp) if len(yt) else float('nan'),
        'valid_query_count': float(query_covered),
        'query_total': float(query_total),
        'query_covered': float(query_covered),
        'query_coverage': float(query_covered / max(query_total, 1)),
        'valid': 1.0,
    }





def safe_nanmean(series) -> float:
    arr = np.asarray(series, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return float('nan')
    return float(np.mean(arr[finite]))


def build_fold_subject_caches(
    ds: dict[str, np.ndarray],
    fold: SessionFold,
    normalizer: SubjectVarlenNormalizer,
    *,
    max_seq_len: int,
    fold_seed: int,
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
    str,
]:
    val_subject_caches: dict[str, dict[str, dict[str, Any]]] = {}
    val_split_info: dict[str, dict[str, Any]] = {}

    for i, subj in enumerate(fold.val_subjects):
        subj = str(subj)
        subj_idx = np.asarray([j for j in fold.val_idx if str(ds['subject'][j]) == subj], dtype=np.int64)
        adapt_idx, eval_idx, split_info = split_subject_adapt_eval_by_sessions(
            ds['session'], subj_idx, seed=fold_seed + 100 + i
        )

        if len(adapt_idx) == 0 or len(eval_idx) == 0:
            raise RuntimeError(f'invalid val session split for {subj}: {split_info}')

        adapt_cache = build_varlen_cache(
            ds,
            adapt_idx,
            normalizer,
            fit_unknown=True,
            max_seq_len=max_seq_len,
        )

        subj_stats = adapt_cache['subject_stats'].get(subj)
        if subj_stats is None:
            raise RuntimeError(f'missing adapt subject stats for {subj}')

        eval_cache = build_varlen_cache(
            ds,
            eval_idx,
            normalizer,
            fit_unknown=False,
            max_seq_len=max_seq_len,
            subject_stats_override={subj: subj_stats},
        )
        eval_zero_cache = build_varlen_cache(
            ds,
            eval_idx,
            normalizer,
            fit_unknown=False,
            max_seq_len=max_seq_len,
        )
        if eval_zero_cache['subject_stats_source'].get(subj) != 'global_fallback':
            raise RuntimeError(f'expected global fallback stats for val subject {subj} in zero-shot cache')

        val_subject_caches[subj] = {
            'adapt': adapt_cache,
            'eval': eval_cache,
            'eval_adapted': eval_cache,
            'eval_zero': eval_zero_cache,
        }
        val_split_info[subj] = split_info

    test_subj = str(fold.test_subjects[0])
    test_subj_idx = np.asarray([j for j in fold.test_idx if str(ds['subject'][j]) == test_subj], dtype=np.int64)
    test_adapt_idx, test_eval_idx, test_split_info = split_subject_adapt_eval_by_sessions(
        ds['session'], test_subj_idx, seed=fold_seed + 777
    )

    if len(test_adapt_idx) == 0 or len(test_eval_idx) == 0:
        raise RuntimeError(f'invalid test session split for {test_subj}: {test_split_info}')

    test_adapt_cache = build_varlen_cache(
        ds,
        test_adapt_idx,
        normalizer,
        fit_unknown=True,
        max_seq_len=max_seq_len,
    )

    test_subj_stats = test_adapt_cache['subject_stats'].get(test_subj)
    if test_subj_stats is None:
        raise RuntimeError(f'missing adapt subject stats for test subject {test_subj}')

    test_eval_cache = build_varlen_cache(
        ds,
        test_eval_idx,
        normalizer,
        fit_unknown=False,
        max_seq_len=max_seq_len,
        subject_stats_override={test_subj: test_subj_stats},
    )
    test_eval_zero_cache = build_varlen_cache(
        ds,
        test_eval_idx,
        normalizer,
        fit_unknown=False,
        max_seq_len=max_seq_len,
    )
    if test_eval_zero_cache['subject_stats_source'].get(test_subj) != 'global_fallback':
        raise RuntimeError(f'expected global fallback stats for test subject {test_subj} in zero-shot cache')

    test_subject_caches = {
        'adapt': test_adapt_cache,
        'eval': test_eval_cache,
        'eval_adapted': test_eval_cache,
        'eval_zero': test_eval_zero_cache,
    }

    return val_subject_caches, val_split_info, test_subject_caches, test_split_info, test_subj


def run_experiment_b(
    *,
    ds: dict[str, np.ndarray],
    folds: list[SessionFold],
    model_class,
    model_cfg: dict[str, Any],
    split_cfg: dict[str, Any],
    subject_norm_cfg: dict[str, Any],
    stagea_cfg: dict[str, Any],
    meta_adapt_cfg: dict[str, Any],
    length_cap_cfg: dict[str, Any],
    out_dir: Path,
    save_checkpoints: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    fold_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    val_detail_rows: list[dict[str, Any]] = []

    out_dir.mkdir(parents=True, exist_ok=True)

    for fold in folds:
        fold_id = int(fold.fold_id)
        fold_seed = int(stagea_cfg['seed']) + fold_id
        fold_label = f'fold {fold_id}'
        set_seed(fold_seed)
        print(
            f"[{fold_label}] start | seed={fold_seed} train={len(fold.train_idx)} "
            f"val={len(fold.val_idx)} test={len(fold.test_idx)}",
            flush=True,
        )

        normalizer = SubjectVarlenNormalizer(**subject_norm_cfg)
        normalizer.fit(ds, fold.train_idx)

        if str(length_cap_cfg.get('policy', 'mean_train')) != 'mean_train':
            raise ValueError("Only length_cap_cfg['policy']='mean_train' is supported")
        fold_train_lengths = np.asarray(ds['lengths'][fold.train_idx], dtype=np.int64)
        fold_max_seq_len = max(1, int(round(float(fold_train_lengths.mean()))))

        train_cache = build_varlen_cache(
            ds,
            fold.train_idx,
            normalizer,
            fit_unknown=True,
            max_seq_len=fold_max_seq_len,
        )

        val_subject_caches, val_split_info, test_subject_caches, test_split_info, test_subj = build_fold_subject_caches(
            ds,
            fold,
            normalizer,
            max_seq_len=fold_max_seq_len,
            fold_seed=fold_seed,
        )

        n_classes_total = int(len(np.unique(ds['target'])))
        model = model_class(
            in_ch=int(train_cache['in_ch']),
            n_classes=n_classes_total,
            **model_cfg,
            variable_length=True,
        )

        val_subject_order = [str(s) for s in fold.val_subjects]

        model, stageA_out = fit_stageA_classifier(
            model,
            train_cache=train_cache,
            val_subject_order=val_subject_order,
            val_subject_caches=val_subject_caches,
            cfg=stagea_cfg,
            meta_adapt_cfg=meta_adapt_cfg,
            device=str(stagea_cfg['device']),
            seed=fold_seed,
            fold_label=fold_label,
        )

        val_zero_rows: list[dict[str, float]] = []
        for subj in val_subject_order:
            val_zero_rows.append(
                evaluate_classifier_cache(
                    model,
                    val_subject_caches[subj]['eval_zero'],
                    device=str(stagea_cfg['device']),
                    batch_size=int(stagea_cfg['batch_size']),
                )
            )
        val_zero = mean_metrics(val_zero_rows)

        test_zero = evaluate_classifier_cache(
            model,
            test_subject_caches['eval_zero'],
            device=str(stagea_cfg['device']),
            batch_size=int(stagea_cfg['batch_size']),
        )

        val_pre_rows: list[dict[str, float]] = []
        val_post_rows: list[dict[str, float]] = []
        val_post_rows_adapt_applied: list[dict[str, float]] = []
        val_setup_used: list[dict[str, Any]] = []

        for i, subj in enumerate(val_subject_order):
            pre_metrics = evaluate_subject_proto(
                model,
                support_cache=val_subject_caches[subj]['adapt'],
                query_cache=val_subject_caches[subj]['eval_adapted'],
                device=str(stagea_cfg['device']),
                emb_norm=bool(meta_adapt_cfg['embedding_norm']),
                batch_size=int(stagea_cfg['batch_size']),
            )

            adapted_model, setup = meta_adapt_on_subject(
                model,
                val_subject_caches[subj]['adapt'],
                cfg=meta_adapt_cfg,
                device=str(stagea_cfg['device']),
                seed=fold_seed + 2000 + i,
            )
            val_setup_used.append({'subject': subj, **setup})

            post_metrics = evaluate_subject_proto(
                adapted_model,
                support_cache=val_subject_caches[subj]['adapt'],
                query_cache=val_subject_caches[subj]['eval_adapted'],
                device=str(stagea_cfg['device']),
                emb_norm=bool(meta_adapt_cfg['embedding_norm']),
                batch_size=int(stagea_cfg['batch_size']),
            )

            val_pre_rows.append(pre_metrics)
            val_post_rows.append(post_metrics)
            if bool(setup.get('adapt_applied', False)) and float(post_metrics.get('valid', 0.0)) > 0.5:
                val_post_rows_adapt_applied.append(post_metrics)

            s_info = val_split_info[subj]
            val_detail_rows.append(
                {
                    'fold': fold_id,
                    'val_subject': subj,
                    'zero_shot_acc': float(val_zero_rows[i]['acc']),
                    'zero_shot_f1_macro': float(val_zero_rows[i]['f1_macro']),
                    'pre_adapt_proto_acc': float(pre_metrics['acc']),
                    'pre_adapt_proto_f1_macro': float(pre_metrics['f1_macro']),
                    'adapt_acc': float(post_metrics['acc']),
                    'adapt_f1_macro': float(post_metrics['f1_macro']),
                    'adapt_valid': int(
                        bool(setup.get('adapt_applied', False))
                        and float(post_metrics.get('valid', 0.0)) > 0.5
                    ),
                    'n_sessions_total': int(s_info['n_sessions_total']),
                    'n_sessions_adapt': int(s_info['n_sessions_adapt']),
                    'n_sessions_eval': int(s_info['n_sessions_eval']),
                    'n_way': int(setup.get('n_way', 0)),
                    'k_shot': int(setup.get('k_shot', 0)),
                    'q_query': int(setup.get('q_query', 0)),
                    'episodes_done': int(setup.get('episodes_done', 0)),
                    'adapt_applied': int(bool(setup.get('adapt_applied', False))),
                    'skip_reason': str(setup.get('skip_reason', '')),
                }
            )

        val_pre = mean_metrics(val_pre_rows)
        if val_post_rows_adapt_applied:
            val_adapt = mean_metrics(val_post_rows_adapt_applied)
        else:
            val_adapt = {'loss': float('nan'), 'acc': float('nan'), 'f1_macro': float('nan')}

        test_pre = evaluate_subject_proto(
            model,
            support_cache=test_subject_caches['adapt'],
            query_cache=test_subject_caches['eval_adapted'],
            device=str(stagea_cfg['device']),
            emb_norm=bool(meta_adapt_cfg['embedding_norm']),
            batch_size=int(stagea_cfg['batch_size']),
        )

        test_adapt_model, test_setup = meta_adapt_on_subject(
            model,
            test_subject_caches['adapt'],
            cfg=meta_adapt_cfg,
            device=str(stagea_cfg['device']),
            seed=fold_seed + 3000,
        )
        test_adapt = evaluate_subject_proto(
            test_adapt_model,
            support_cache=test_subject_caches['adapt'],
            query_cache=test_subject_caches['eval_adapted'],
            device=str(stagea_cfg['device']),
            emb_norm=bool(meta_adapt_cfg['embedding_norm']),
            batch_size=int(stagea_cfg['batch_size']),
        )
        test_adapt_valid = int(
            bool(test_setup.get('adapt_applied', False))
            and float(test_adapt.get('valid', 0.0)) > 0.5
        )

        if int(test_pre.get('query_total', 0.0)) != int(len(test_subject_caches['eval_adapted']['y'])):
            raise RuntimeError(f'proto pre-adapt query accounting mismatch in fold {fold_id}')
        if int(test_adapt.get('query_total', 0.0)) != int(len(test_subject_caches['eval_adapted']['y'])):
            raise RuntimeError(f'proto adapted query accounting mismatch in fold {fold_id}')

        if save_checkpoints:
            ckpt_dir = out_dir / 'checkpoints'
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f'nm_ce_then_adapt__seed{stagea_cfg["seed"]}__fold{fold_id}.pt'
            torch.save(
                {
                    'state_dict_stageA': model.state_dict(),
                    'fold_id': fold_id,
                    'seed': fold_seed,
                    'fold_subjects': {
                        'train': fold.train_subjects,
                        'val': fold.val_subjects,
                        'test': fold.test_subjects,
                    },
                    'stageA_out': stageA_out,
                    'val_zero': val_zero,
                    'val_pre': val_pre,
                    'val_adapt': val_adapt,
                    'test_zero': test_zero,
                    'test_pre': test_pre,
                    'test_adapt': test_adapt,
                    'test_meta_setup': test_setup,
                    'val_meta_setups': val_setup_used,
                    'val_session_split': val_split_info,
                    'test_session_split': test_split_info,
                    'max_seq_len': int(fold_max_seq_len),
                    'stageA_cfg': stagea_cfg,
                    'meta_adapt_cfg': meta_adapt_cfg,
                    'split_cfg': split_cfg,
                    'subject_norm_cfg': subject_norm_cfg,
                    'model_cfg': model_cfg,
                },
                ckpt_path,
            )
        else:
            ckpt_path = out_dir / f'nm_ce_then_adapt__seed{stagea_cfg["seed"]}__fold{fold_id}.pt'

        row = {
            'fold': fold_id,
            'seed': fold_seed,
            'train_subjects': ','.join(fold.train_subjects),
            'val_subjects': ','.join(fold.val_subjects),
            'test_subject': ','.join(fold.test_subjects),
            'n_train': int(len(fold.train_idx)),
            'n_val': int(len(fold.val_idx)),
            'n_test': int(len(fold.test_idx)),
            'val_adapt_monitor_f1': float(stageA_out['best_val_adapt_monitor_f1']),
            'val_zero_shot_acc': float(val_zero['acc']),
            'val_zero_shot_f1_macro': float(val_zero['f1_macro']),
            'val_pre_adapt_proto_acc': float(val_pre['acc']),
            'val_pre_adapt_proto_f1_macro': float(val_pre['f1_macro']),
            'val_adapt_acc': float(val_adapt['acc']),
            'val_adapt_f1_macro': float(val_adapt['f1_macro']),
            'test_zero_shot_acc': float(test_zero['acc']),
            'test_zero_shot_f1_macro': float(test_zero['f1_macro']),
            'test_pre_adapt_proto_acc': float(test_pre['acc']),
            'test_pre_adapt_proto_f1_macro': float(test_pre['f1_macro']),
            'test_adapt_acc': float(test_adapt['acc']),
            'test_adapt_f1_macro': float(test_adapt['f1_macro']),
            'test_adapt_valid': test_adapt_valid,
            'test_pre_query_total': float(test_pre.get('query_total', 0.0)),
            'test_pre_query_covered': float(test_pre.get('query_covered', 0.0)),
            'test_pre_query_coverage': float(test_pre.get('query_coverage', 0.0)),
            'test_adapt_query_total': float(test_adapt.get('query_total', 0.0)),
            'test_adapt_query_covered': float(test_adapt.get('query_covered', 0.0)),
            'test_adapt_query_coverage': float(test_adapt.get('query_coverage', 0.0)),
            'test_subject_n_sessions_total': int(test_split_info['n_sessions_total']),
            'test_subject_n_sessions_adapt': int(test_split_info['n_sessions_adapt']),
            'test_subject_n_sessions_eval': int(test_split_info['n_sessions_eval']),
            'val_subjects_sessions': json.dumps({s: val_split_info[s] for s in val_subject_order}, ensure_ascii=False),
            'n_way': int(test_setup.get('n_way', 0)),
            'k_shot': int(test_setup.get('k_shot', 0)),
            'q_query': int(test_setup.get('q_query', 0)),
            'max_seq_len': int(fold_max_seq_len),
            'best_epoch_stageA': int(stageA_out['best_epoch_stageA']),
            'adapt_applied_test': int(bool(test_setup.get('adapt_applied', False))),
            'skip_reason_test': str(test_setup.get('skip_reason', '')),
            'checkpoint': str(ckpt_path),
        }
        fold_rows.append(row)

        for h in stageA_out['history']:
            history_rows.append({'fold': fold_id, 'seed': fold_seed, **h})

        print(
            f"[{fold_label}] val_monitor_f1={stageA_out['best_val_adapt_monitor_f1']:.4f} "
            f"val0_f1={val_zero['f1_macro']:.4f} val_pre_f1={val_pre['f1_macro']:.4f} valA_f1={val_adapt['f1_macro']:.4f} | "
            f"test0_f1={test_zero['f1_macro']:.4f} test_pre_f1={test_pre['f1_macro']:.4f} testA_f1={test_adapt['f1_macro']:.4f} | "
            f"valid_test={test_adapt_valid} max_seq_len={fold_max_seq_len} "
            f"test_sessions={test_split_info['n_sessions_total']}",
            flush=True,
        )

    if not fold_rows:
        raise RuntimeError('No fold results were produced.')

    fold_df = pd.DataFrame(fold_rows).sort_values('fold').reset_index(drop=True)
    hist_df = pd.DataFrame(history_rows).sort_values(['fold', 'epoch']).reset_index(drop=True)
    val_detail_df = pd.DataFrame(val_detail_rows).sort_values(['fold', 'val_subject']).reset_index(drop=True)

    fold_csv = out_dir / 'nm_ce_then_adapt__fold_metrics.csv'
    hist_csv = out_dir / 'nm_ce_then_adapt__history.csv'
    val_detail_csv = out_dir / 'nm_ce_then_adapt__val_subject_detail.csv'
    summary_json = out_dir / 'nm_ce_then_adapt__summary.json'

    fold_df.to_csv(fold_csv, index=False)
    hist_df.to_csv(hist_csv, index=False)
    val_detail_df.to_csv(val_detail_csv, index=False)

    valid_adapt_mask = (
        (np.asarray(fold_df['test_adapt_valid'], dtype=np.float64) > 0.5)
        & (np.asarray(fold_df['adapt_applied_test'], dtype=np.float64) > 0.5)
    )
    valid_adapt_df = fold_df.loc[valid_adapt_mask].copy()

    summary = {
        'n_folds': int(len(fold_df)),
        'n_valid_test_adapt_folds': int(np.sum(valid_adapt_mask)),
        'n_skipped_test_adapt_folds': int(len(fold_df) - np.sum(valid_adapt_mask)),
        'val_adapt_monitor_f1_mean': safe_nanmean(fold_df['val_adapt_monitor_f1']),
        'val_zero_shot_acc_mean': safe_nanmean(fold_df['val_zero_shot_acc']),
        'val_zero_shot_f1_mean': safe_nanmean(fold_df['val_zero_shot_f1_macro']),
        'val_pre_adapt_proto_f1_mean': safe_nanmean(fold_df['val_pre_adapt_proto_f1_macro']),
        'val_adapt_acc_mean': safe_nanmean(fold_df['val_adapt_acc']),
        'val_adapt_f1_mean': safe_nanmean(fold_df['val_adapt_f1_macro']),
        'test_zero_shot_acc_mean': safe_nanmean(fold_df['test_zero_shot_acc']),
        'test_zero_shot_f1_mean': safe_nanmean(fold_df['test_zero_shot_f1_macro']),
        'test_pre_adapt_proto_f1_mean': safe_nanmean(fold_df['test_pre_adapt_proto_f1_macro']),
        'test_adapt_acc_mean': safe_nanmean(valid_adapt_df['test_adapt_acc']),
        'test_adapt_f1_mean': safe_nanmean(valid_adapt_df['test_adapt_f1_macro']),
        'test_adapt_acc_mean_valid_only': safe_nanmean(valid_adapt_df['test_adapt_acc']),
        'test_adapt_f1_mean_valid_only': safe_nanmean(valid_adapt_df['test_adapt_f1_macro']),
        'test_adapt_acc_mean_all': safe_nanmean(fold_df['test_adapt_acc']),
        'test_adapt_f1_mean_all': safe_nanmean(fold_df['test_adapt_f1_macro']),
        'test_pre_query_coverage_mean': safe_nanmean(fold_df['test_pre_query_coverage']),
        'test_adapt_query_coverage_mean': safe_nanmean(fold_df['test_adapt_query_coverage']),
        'zero_shot_definition': 'pure_zero_shot_no_target_subject_statistics',
        'split_cfg': split_cfg,
        'subject_norm_cfg': subject_norm_cfg,
        'length_cap_cfg': length_cap_cfg,
        'stageA_cfg': stagea_cfg,
        'meta_adapt_cfg': meta_adapt_cfg,
        'model_cfg': model_cfg,
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    print('Saved fold metrics:', fold_csv, flush=True)
    print('Saved history:', hist_csv, flush=True)
    print('Saved val subject detail:', val_detail_csv, flush=True)
    print('Saved summary:', summary_json, flush=True)

    return fold_df, hist_df, val_detail_df, summary
