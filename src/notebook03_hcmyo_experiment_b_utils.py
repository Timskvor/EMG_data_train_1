from __future__ import annotations

from pathlib import Path
import copy
import json
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

import notebook02_experiment_b_utils as nb7


set_seed = nb7.set_seed
load_model_class = nb7.load_model_class
SubjectVarlenNormalizer = nb7.SubjectVarlenNormalizer
build_varlen_cache = nb7.build_varlen_cache
fit_stageA_classifier = nb7.fit_stageA_classifier
evaluate_classifier_cache = nb7.evaluate_classifier_cache
evaluate_subject_proto = nb7.evaluate_subject_proto
meta_adapt_on_subject = nb7.meta_adapt_on_subject
split_subject_adapt_eval_by_sessions = nb7.split_subject_adapt_eval_by_sessions


def safe_nanmean(series) -> float:
    arr = np.asarray(series, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return float('nan')
    return float(np.mean(arr[finite]))


def load_hcmyo_postcrop_varlen(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as d:
        X = np.asarray(d['X'], dtype=np.float32)  # [S, N, C, T]
        y = np.asarray(d['y'], dtype=np.int64)    # [S, N]
        subjects = np.asarray(d['subjects']).astype(str)  # [S]
        lens = np.asarray(d['lens'], dtype=np.int64)      # [S, N]

    if X.ndim != 4 or y.ndim != 2 or lens.ndim != 2:
        raise RuntimeError('Unexpected NPZ structure for HCMYO preprocessed dataset')
    if X.shape[:2] != y.shape or y.shape != lens.shape:
        raise RuntimeError('Shape mismatch among X, y, lens')

    windows: list[np.ndarray] = []
    targets: list[int] = []
    subj_arr: list[str] = []
    trial_ids: list[str] = []
    lengths: list[int] = []

    n_subj, n_trials, n_ch, max_t = X.shape
    for si in range(n_subj):
        subj = str(subjects[si])
        for ti in range(n_trials):
            real_len = int(lens[si, ti])
            real_len = max(1, min(real_len, int(max_t)))
            x_ct = X[si, ti, :, :real_len]          # [C, T]
            x_tc = np.transpose(x_ct, (1, 0))       # [T, C]
            windows.append(x_tc.astype(np.float32, copy=False))
            targets.append(int(y[si, ti]))
            subj_arr.append(subj)
            trial_ids.append(f'{subj}_trial_{ti:05d}')
            lengths.append(real_len)

    if not windows:
        raise RuntimeError('No windows extracted from NPZ')

    offsets = [0]
    for w in windows:
        offsets.append(offsets[-1] + int(w.shape[0]))

    emg_concat = np.concatenate(windows, axis=0).astype(np.float32, copy=False)

    return {
        'emg_concat': emg_concat,
        'offsets': np.asarray(offsets, dtype=np.int64),
        'target': np.asarray(targets, dtype=np.int64),
        'subject': np.asarray(subj_arr, dtype=str),
        'session': np.asarray(trial_ids, dtype=str),
        'lengths': np.asarray(lengths, dtype=np.int64),
    }


def split_target_adapt_eval_stratified(
    y_full: np.ndarray,
    target_idx: np.ndarray,
    *,
    seed: int,
    adapt_ratio: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    idx = np.asarray(target_idx, dtype=np.int64).reshape(-1)
    y = np.asarray(y_full, dtype=np.int64)
    rng = np.random.default_rng(int(seed))

    adapt_chunks: list[np.ndarray] = []
    eval_chunks: list[np.ndarray] = []

    for cls in sorted(np.unique(y[idx]).tolist()):
        cls_idx = idx[y[idx] == int(cls)].copy()
        rng.shuffle(cls_idx)

        if len(cls_idx) <= 1:
            adapt_chunks.append(cls_idx[:1])
            continue

        n_adapt = int(round(len(cls_idx) * float(adapt_ratio)))
        n_adapt = max(1, min(len(cls_idx) - 1, n_adapt))

        adapt_chunks.append(cls_idx[:n_adapt])
        eval_chunks.append(cls_idx[n_adapt:])

    adapt_idx = np.concatenate(adapt_chunks) if adapt_chunks else np.asarray([], dtype=np.int64)
    eval_idx = np.concatenate(eval_chunks) if eval_chunks else np.asarray([], dtype=np.int64)

    adapt_idx = np.asarray(sorted(adapt_idx.tolist()), dtype=np.int64)
    eval_idx = np.asarray(sorted(eval_idx.tolist()), dtype=np.int64)

    info = {
        'n_total': int(len(idx)),
        'n_adapt': int(len(adapt_idx)),
        'n_eval': int(len(eval_idx)),
        'adapt_ratio_realized': float(len(adapt_idx) / max(len(idx), 1)),
        'classes': sorted(np.unique(y[idx]).tolist()),
    }
    return adapt_idx, eval_idx, info


def build_three_subject_runs(subjects: np.ndarray) -> list[dict[str, Any]]:
    uniq = sorted(np.unique(np.asarray(subjects).astype(str)).tolist())
    if len(uniq) != 3:
        raise RuntimeError(f'Expected exactly 3 subjects, got {len(uniq)}: {uniq}')

    runs = []
    for i, target_subj in enumerate(uniq, start=1):
        train_subjects = [s for s in uniq if s != target_subj]
        runs.append({
            'run_id': int(i),
            'target_subject': str(target_subj),
            'train_subjects': train_subjects,
        })
    return runs


def run_hcmyo_experiment_b(
    *,
    ds: dict[str, np.ndarray],
    runs: list[dict[str, Any]],
    model_class,
    model_cfg: dict[str, Any],
    stagea_cfg: dict[str, Any],
    meta_adapt_cfg: dict[str, Any],
    subject_norm_cfg: dict[str, Any],
    adapt_ratio: float,
    out_dir: Path,
    save_checkpoints: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    fold_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []

    out_dir.mkdir(parents=True, exist_ok=True)

    subj_full = np.asarray(ds['subject']).astype(str)
    y_full = np.asarray(ds['target'], dtype=np.int64)
    source_train_ratio = 0.8

    for run in runs:
        run_id = int(run['run_id'])
        target_subj = str(run['target_subject'])
        train_subjects = [str(s) for s in run['train_subjects']]
        run_seed = int(stagea_cfg['seed']) + run_id

        set_seed(run_seed)

        source_train_chunks: list[np.ndarray] = []
        source_monitor_pool_by_subject: dict[str, np.ndarray] = {}
        for i, subj in enumerate(sorted(train_subjects)):
            subj_idx = np.where(subj_full == subj)[0].astype(np.int64)
            src_train_idx, src_monitor_idx, _src_info = split_subject_adapt_eval_by_sessions(
                ds['session'],
                subj_idx,
                seed=run_seed + 100 + i,
                ratio=float(source_train_ratio),
            )
            if len(src_train_idx) == 0 or len(src_monitor_idx) == 0:
                raise RuntimeError(f'invalid source train/monitor split for {subj}')
            if np.intersect1d(src_train_idx, src_monitor_idx).size > 0:
                raise RuntimeError(f'source split overlap for {subj}')
            source_train_chunks.append(src_train_idx)
            source_monitor_pool_by_subject[subj] = src_monitor_idx

        if not source_train_chunks:
            raise RuntimeError('source_train_chunks is empty')
        train_idx = np.unique(np.concatenate(source_train_chunks)).astype(np.int64)
        source_monitor_total = int(sum(len(v) for v in source_monitor_pool_by_subject.values()))
        if len(train_idx) == 0 or source_monitor_total == 0:
            raise RuntimeError('invalid source split: empty train or monitor pool')

        target_all_idx = np.where(subj_full == target_subj)[0].astype(np.int64)

        target_adapt_idx, target_eval_idx, split_info = split_target_adapt_eval_stratified(
            y_full,
            target_all_idx,
            seed=run_seed + 17,
            adapt_ratio=float(adapt_ratio),
        )

        if len(target_adapt_idx) == 0 or len(target_eval_idx) == 0:
            raise RuntimeError(f'Invalid target split for {target_subj}: {split_info}')

        normalizer = SubjectVarlenNormalizer(**subject_norm_cfg)
        normalizer.fit(ds, train_idx)

        train_lengths = np.asarray(ds['lengths'][train_idx], dtype=np.int64)
        max_seq_len = max(1, int(round(float(train_lengths.mean()))))

        train_cache = build_varlen_cache(
            ds,
            train_idx,
            normalizer,
            fit_unknown=True,
            max_seq_len=max_seq_len,
        )

        source_val_subject_order = sorted(train_subjects)
        if not source_val_subject_order:
            raise RuntimeError('source_val_subject_order is empty')
        if target_subj in set(source_val_subject_order):
            raise RuntimeError('target subject leakage into StageA monitor subjects')

        val_subject_caches: dict[str, dict[str, Any]] = {}
        for i, subj in enumerate(source_val_subject_order):
            subj_idx = source_monitor_pool_by_subject[subj]
            vadapt_idx, veval_idx, _vinfo = split_subject_adapt_eval_by_sessions(
                ds['session'],
                subj_idx,
                seed=run_seed + 300 + i,
                ratio=0.5,
            )
            if len(vadapt_idx) == 0 or len(veval_idx) == 0:
                raise RuntimeError(f'invalid source val split for {subj}')

            vadapt_cache = build_varlen_cache(
                ds,
                vadapt_idx,
                normalizer,
                fit_unknown=False,
                max_seq_len=max_seq_len,
                subject_stats_override={subj: normalizer._estimate_from_indices(ds, vadapt_idx)},
            )
            subj_monitor_stats = vadapt_cache['subject_stats'].get(subj)
            if subj_monitor_stats is None:
                raise RuntimeError(f'missing source monitor-adapt stats for {subj}')

            veval_adapted_cache = build_varlen_cache(
                ds,
                veval_idx,
                normalizer,
                fit_unknown=False,
                max_seq_len=max_seq_len,
                subject_stats_override={subj: subj_monitor_stats},
            )
            if veval_adapted_cache['subject_stats_source'].get(subj) != 'override':
                raise RuntimeError(f'expected override stats for source monitor eval on {subj}')

            veval_zero_cache = build_varlen_cache(
                ds,
                veval_idx,
                normalizer,
                fit_unknown=False,
                max_seq_len=max_seq_len,
            )
            if veval_zero_cache['subject_stats_source'].get(subj) != 'train_subject':
                raise RuntimeError(f'expected train_subject stats for source zero-shot eval on {subj}')

            if np.intersect1d(train_idx, vadapt_idx).size > 0 or np.intersect1d(train_idx, veval_idx).size > 0:
                raise RuntimeError(f'source monitor overlap with StageA train for {subj}')

            val_subject_caches[subj] = {
                'adapt': vadapt_cache,
                'eval': veval_adapted_cache,
                'eval_adapted': veval_adapted_cache,
                'eval_zero': veval_zero_cache,
            }

        target_adapt_cache = build_varlen_cache(
            ds,
            target_adapt_idx,
            normalizer,
            fit_unknown=True,
            max_seq_len=max_seq_len,
        )

        target_eval_zero_cache = build_varlen_cache(
            ds,
            target_eval_idx,
            normalizer,
            fit_unknown=False,
            max_seq_len=max_seq_len,
        )
        if target_eval_zero_cache['subject_stats_source'].get(target_subj) != 'global_fallback':
            raise RuntimeError(f'expected global fallback stats for pure zero-shot on {target_subj}')

        target_stats = target_adapt_cache['subject_stats'].get(target_subj)
        if target_stats is None:
            raise RuntimeError(f'Missing target stats for {target_subj}')

        target_eval_adapted_cache = build_varlen_cache(
            ds,
            target_eval_idx,
            normalizer,
            fit_unknown=False,
            max_seq_len=max_seq_len,
            subject_stats_override={target_subj: target_stats},
        )
        if target_eval_adapted_cache['subject_stats_source'].get(target_subj) != 'override':
            raise RuntimeError(f'expected override target stats for adapted eval on {target_subj}')

        # Stage A monitor uses source-only validation subjects (no target peeking).
        val_subject_order = source_val_subject_order

        n_classes = int(len(np.unique(y_full)))
        model = model_class(
            in_ch=int(train_cache['in_ch']),
            n_classes=n_classes,
            **model_cfg,
            variable_length=True,
        )

        model, stageA_out = fit_stageA_classifier(
            model,
            train_cache=train_cache,
            val_subject_order=val_subject_order,
            val_subject_caches=val_subject_caches,
            cfg=stagea_cfg,
            meta_adapt_cfg=meta_adapt_cfg,
            device=str(stagea_cfg['device']),
            seed=run_seed,
            fold_label=f'run {run_id}',
        )

        test_zero = evaluate_classifier_cache(
            model,
            target_eval_zero_cache,
            device=str(stagea_cfg['device']),
            batch_size=int(stagea_cfg['batch_size']),
        )

        test_pre = evaluate_subject_proto(
            model,
            support_cache=target_adapt_cache,
            query_cache=target_eval_adapted_cache,
            device=str(stagea_cfg['device']),
            emb_norm=bool(meta_adapt_cfg['embedding_norm']),
            batch_size=int(stagea_cfg['batch_size']),
        )

        test_adapt_model, test_setup = meta_adapt_on_subject(
            model,
            target_adapt_cache,
            cfg=meta_adapt_cfg,
            device=str(stagea_cfg['device']),
            seed=run_seed + 1000,
        )

        test_adapt = evaluate_subject_proto(
            test_adapt_model,
            support_cache=target_adapt_cache,
            query_cache=target_eval_adapted_cache,
            device=str(stagea_cfg['device']),
            emb_norm=bool(meta_adapt_cfg['embedding_norm']),
            batch_size=int(stagea_cfg['batch_size']),
        )
        test_adapt_valid = int(
            bool(test_setup.get('adapt_applied', False))
            and float(test_adapt.get('valid', 0.0)) > 0.5
        )

        if int(test_pre.get('query_total', 0.0)) != int(len(target_eval_adapted_cache['y'])):
            raise RuntimeError(f'proto pre-adapt query accounting mismatch in run {run_id}')
        if int(test_adapt.get('query_total', 0.0)) != int(len(target_eval_adapted_cache['y'])):
            raise RuntimeError(f'proto adapted query accounting mismatch in run {run_id}')

        ckpt_path = out_dir / f'hcmyo_ce_then_adapt__seed{stagea_cfg["seed"]}__run{run_id}.pt'
        if save_checkpoints:
            ckpt_dir = out_dir / 'checkpoints'
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / ckpt_path.name
            torch.save(
                {
                    'state_dict_stageA': model.state_dict(),
                    'run': run,
                    'seed': run_seed,
                    'split_info': split_info,
                    'stageA_out': stageA_out,
                    'test_zero': test_zero,
                    'test_pre': test_pre,
                    'test_adapt': test_adapt,
                    'test_meta_setup': test_setup,
                    'stageA_cfg': stagea_cfg,
                    'meta_adapt_cfg': meta_adapt_cfg,
                    'subject_norm_cfg': subject_norm_cfg,
                    'model_cfg': model_cfg,
                    'adapt_ratio': float(adapt_ratio),
                },
                ckpt_path,
            )

        fold_rows.append(
            {
                'run': run_id,
                'seed': run_seed,
                'train_subjects': ','.join(train_subjects),
                'target_subject': target_subj,
                'n_train': int(len(train_idx)),
                'n_source_monitor_pool': int(source_monitor_total),
                'n_target_total': int(split_info['n_total']),
                'n_target_adapt': int(split_info['n_adapt']),
                'n_target_eval': int(split_info['n_eval']),
                'adapt_ratio_realized': float(split_info['adapt_ratio_realized']),
                'best_epoch_stageA': int(stageA_out['best_epoch_stageA']),
                'val_adapt_monitor_f1': float(stageA_out['best_val_adapt_monitor_f1']),
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
                'n_way': int(test_setup.get('n_way', 0)),
                'k_shot': int(test_setup.get('k_shot', 0)),
                'q_query': int(test_setup.get('q_query', 0)),
                'adapt_applied': int(bool(test_setup.get('adapt_applied', False))),
                'skip_reason': str(test_setup.get('skip_reason', '')),
                'source_only_stageA_monitor': 1,
                'source_monitor_disjoint_from_train': 1,
                'checkpoint': str(ckpt_path),
            }
        )

        for h in stageA_out['history']:
            history_rows.append({'run': run_id, 'seed': run_seed, **h})

        print(
            f"[run {run_id}] target={target_subj} train={train_subjects} "
            f"adapt_ratio={split_info['adapt_ratio_realized']:.3f} "
            f"test0_f1={test_zero['f1_macro']:.4f} "
            f"test_pre_f1={test_pre['f1_macro']:.4f} "
            f"test_adapt_f1={test_adapt['f1_macro']:.4f}",
            flush=True,
        )

    if not fold_rows:
        raise RuntimeError('No run results were produced')

    fold_df = pd.DataFrame(fold_rows).sort_values('run').reset_index(drop=True)
    hist_df = pd.DataFrame(history_rows).sort_values(['run', 'epoch']).reset_index(drop=True)

    fold_csv = out_dir / 'hcmyo_ce_then_adapt__run_metrics.csv'
    hist_csv = out_dir / 'hcmyo_ce_then_adapt__history.csv'
    summary_json = out_dir / 'hcmyo_ce_then_adapt__summary.json'

    fold_df.to_csv(fold_csv, index=False)
    hist_df.to_csv(hist_csv, index=False)

    valid_adapt_mask = (
        (np.asarray(fold_df['test_adapt_valid'], dtype=np.float64) > 0.5)
        & (np.asarray(fold_df['adapt_applied'], dtype=np.float64) > 0.5)
    )
    valid_adapt_df = fold_df.loc[valid_adapt_mask].copy()

    summary = {
        'n_runs': int(len(fold_df)),
        'adapt_ratio': float(adapt_ratio),
        'source_train_ratio': float(source_train_ratio),
        'n_valid_test_adapt_runs': int(np.sum(valid_adapt_mask)),
        'n_skipped_test_adapt_runs': int(len(fold_df) - np.sum(valid_adapt_mask)),
        'val_adapt_monitor_f1_mean': safe_nanmean(fold_df['val_adapt_monitor_f1']),
        'test_zero_shot_f1_mean': safe_nanmean(fold_df['test_zero_shot_f1_macro']),
        'test_pre_adapt_proto_f1_mean': safe_nanmean(fold_df['test_pre_adapt_proto_f1_macro']),
        'test_adapt_f1_mean': safe_nanmean(valid_adapt_df['test_adapt_f1_macro']),
        'test_adapt_f1_mean_valid_only': safe_nanmean(valid_adapt_df['test_adapt_f1_macro']),
        'test_adapt_f1_mean_all': safe_nanmean(fold_df['test_adapt_f1_macro']),
        'n_source_monitor_pool_mean': safe_nanmean(fold_df['n_source_monitor_pool']),
        'test_pre_query_coverage_mean': safe_nanmean(fold_df['test_pre_query_coverage']),
        'test_adapt_query_coverage_mean': safe_nanmean(fold_df['test_adapt_query_coverage']),
        'zero_shot_definition': 'pure_zero_shot_no_target_subject_statistics',
        'source_only_stageA_monitor': True,
        'source_monitor_disjoint_from_train': True,
        'stageA_cfg': stagea_cfg,
        'meta_adapt_cfg': meta_adapt_cfg,
        'subject_norm_cfg': subject_norm_cfg,
        'model_cfg': model_cfg,
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    print('Saved run metrics:', fold_csv, flush=True)
    print('Saved history:', hist_csv, flush=True)
    print('Saved summary:', summary_json, flush=True)

    return fold_df, hist_df, summary
