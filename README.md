# code_bobr_678

Repository for sEMG handwriting experiments:
- **Experiment 01**: cross-dataset transfer vs scratch
- **Experiment 02**: NM subject adaptation (`CE -> meta-adapt`)
- **Experiment 03**: HCMYO-A low-budget adaptation (3-run LOO)

## 1. Repository Layout

```text
code_bobr_678/
  data/
    preprocessing/                # dataset build notebooks
    raw/                          # local raw datasets (not versioned)
    datasets/                     # local built NPZ datasets (not versioned)
  models/                         # model definitions
  src/                            # experiment utilities
  notebooks/                      # 01 / 02 / 03 experiment notebooks
  reports/                        # paper docs, figures, artifacts
  requirements.txt
```

## 2. Environment Setup

From `code_bobr_678/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Dataset Installation (Required)

The experiments expect **three prepared NPZ files** in `data/datasets/`:

- `old_digits_varlen_8ch.npz`
- `nm_digits_varlen_16ch.npz`
- `hcmyo_preprocessed.npz`

You can provide them in one of two ways:

### Option A: Use prebuilt NPZ files

Place NPZ files directly into:

```text
code_bobr_678/data/datasets/
```

### Option B: Rebuild NPZ files from raw datasets

1. Place raw data locally (outside git tracking) using this structure:

```text
code_bobr_678/data/raw/
  HCMYO-A/
    data/...
  nm000106-main/
    sub-*/ses-*/emg/...
  data_8ch_EMG/
    ...
  dataset_metadata.json
```

2. Open preprocessing notebooks:
- `data/preprocessing/preprocessing.ipynb` (builds `hcmyo_preprocessed.npz`)
- `data/preprocessing/01_preprocess_nm000106.ipynb` (builds `old_digits_varlen_8ch.npz` and `nm_digits_varlen_16ch.npz`)

3. In the first config cell of each notebook, set local paths (`DATA_DIR`, `NM_ROOT`, `OLD_ROOT`, `OLD_META`) to your machine paths.

4. Run notebooks end-to-end and verify created files exist in `data/datasets/`.

## 4. Running Experiments

Run notebooks in this order:

1. `notebooks/01_old_pretrain_nm_subject50_pca8_vs_scratch.ipynb`
2. `notebooks/02_experiment_b_only.ipynb`
3. `notebooks/03_hcmyo_ce_then_adapt_3runs.ipynb`

Generated outputs are written under:
- `reports/artifacts/notebook01|02|03/`
- `reports/figures/notebook01|02|03/`

## 5. Reports

Main document:
- `reports/emg_paper_v3.docx`

## 6. Reproducibility Notes

- Split protocols are exported per notebook as:
  - `split_protocol.json`
  - `split_summary.csv`
- Random seed rules are documented in these artifacts and in notebook protocol sections.
- Avoid mixing old/new artifacts across reruns; clear or archive old outputs before a fresh full run.
