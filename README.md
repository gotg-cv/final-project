# Classroom engagement (VideoMAE + DAiSEE)

Fine-tune `MCG-NJU/videomae-base-finetuned-kinetics` on DAiSEE clips and four affect scores (Boredom, Confusion, Engagement, Frustration). Labels are collapsed to **one dominant class per clip** to match our 4-way head.

Training uses Hugging Face `Trainer` with **validation accuracy** and **macro F1**. The checkpoint with best **macro F1** is loaded at the end before saving to `./outputs/…/final/`.

Paper / dataset citation: Gupta et al., *DAiSEE: Towards User Engagement Recognition in the Wild*, [arXiv:1609.01885](https://arxiv.org/abs/1609.01885).

---

## Repository layout (naming)

```text
Final_Project/
├── src/
│   ├── train.py            # Fine-tuning
│   ├── evaluate.py         # Validation metrics + confusion matrix
│   ├── inference.py       # Single-clip prediction
│   ├── daisee_io.py        # CSV → paths + labels
│   ├── metrics_utils.py    # accuracy / macro-F1 for Trainer
│   └── …
├── kaggle/KAGGLE_RUN.txt   # Notebook cells for Kaggle
├── fixtures/daisee_minimal/
├── run_sweep.py            # Sequential hyperparameter runs
├── sweep_presets.json      # Sweep definitions
├── check_daisee_paths.py
├── dry_run.py, smoke_test_dataset.py
├── config.json
├── requirements.txt
```

**Conventions**

- **`--data_root`**: absolute or relative path whose **children** are **`Labels/`** and **`DataSet/`** exactly as DAiSEE ships. Do not rename those folders.
- **Local full dataset:** put it under **`data/daisee/`** so it never gets committed (see `.gitignore`).
- **Runs:** use **`outputs/<run_name>/`** (e.g. `outputs/run01/`) for `--output_dir`; final weights in **`outputs/<run_name>/final/`**.

---

## Environment

From the repo root (`Final_Project/`):

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For a **matching install** across machines use the lockfile (same Python major helps):

```bash
pip install -r requirements-lock.txt
```

The lockfile in this repo was produced on **macOS (Darwin)** with **Python 3.14.x**. Pins can differ by OS and Python version — if `pip install -r requirements-lock.txt` fails on Linux or Colab, use `requirements.txt` instead or regenerate the lock (`pip freeze`) in a venv on that machine.

`FORCE_CPU=1` forces CPU for inference and `dry_run` if you hit device issues.

---

## Getting DAiSEE

1. **Course / team archive ([`DAiSEE.zip` on Google Drive](https://drive.google.com/file/d/1yrk_wyhZ-c7q0Mcyyi888ylFkl_JDELi/view))** — download and verify the zip matches the folder layout below.

2. **Official source** (form, terms, ~15 GB): [IITH DAiSEE](https://people.iith.ac.in/vineethnb/resources/daisee/index.html).

3. Unzip so the **root you pass to `--data_root`** looks like this (names are fixed by the dataset):

```text
<data_root>/                 # e.g. Final_Project/data/daisee
  Labels/
    TrainLabels.csv
    ValidationLabels.csv
  DataSet/
    Train/<6-digit>/<clip_id>/<ClipID.avi or .mp4>
    Validation/...
```

Recommended on this machine:

```bash
mkdir -p data/daisee
# unzip so that data/daisee/Labels and data/daisee/DataSet exist
```

**`fixtures/daisee_minimal/`** is a minimal copy for tests only (includes one intentional **missing** clip row). It is **not** the full dataset.

### Kaggle (not streaming)

[Kaggle DAiSEE](https://www.kaggle.com/datasets/olgaparfenova/daisee) does **not** support incremental **streaming** in the Hugging Face sense for **video**. Our loader uses **OpenCV** on **file paths** and uniform frame indices, which means the data must live on a filesystem the process can **`open`** (disk, SSD, or notebook mount)—not lazily streamed from the internet during each epoch.

**What you can do instead:**

- **Kaggle notebooks:** Attach the dataset, then find the mount under **`/kaggle/input/<dataset-name>/`** and pass the directory that contains **`Labels/`** and **`DataSet/`** as **`--data_root`** (unzip archives into **`/kaggle/working`** once if everything is zipped).
- **`kaggle`** CLI or **`kagglehub`** on your laptop: downloads to a cache directory; unzip once, same layout as **`data/daisee/`**.

So “from Kaggle” really means **download or mount**, then **`--data_root`** points there—same pipeline as Drive or IIT.

Confirm paths before training:

```bash
.venv/bin/python check_daisee_paths.py --data_root ./data/daisee
# or: --data_root ./fixtures/daisee_minimal
```

Exits non-zero if Train or Validation has **zero** matching clips. Otherwise missing rows are skipped in training; watch **`Samples — train:`** in logs.

---

## Checks before training

| Step | Command |
|------|---------|
| Model + logits `(1,4)` | `.venv/bin/python dry_run.py` |
| Decoder + preprocessor | `.venv/bin/python smoke_test_dataset.py` |
| DAiSEE CSV vs disk | `.venv/bin/python check_daisee_paths.py --data_root ./data/daisee` |

---

## Training

From `Final_Project/`:

```bash
.venv/bin/python src/train.py --data_root ./data/daisee --output_dir outputs/run01
```

`config.json` controls lr, epochs, batch size, gradient accumulation, and `freeze_base`.

**Resume:**

```bash
.venv/bin/python src/train.py \
  --data_root ./data/daisee \
  --output_dir outputs/run01 \
  --resume_from_checkpoint outputs/run01/checkpoint-3
```

Optional caps for quick trials:

```bash
.venv/bin/python src/train.py --data_root ./data/daisee --output_dir outputs/run01 \
  --max_train_samples 512 --max_eval_samples 256
```

---

## Evaluate

After training, **`outputs/.../final/`** holds the best checkpoint. Report on the **Validation** split:

```bash
.venv/bin/python src/evaluate.py \
  --data_root ./data/daisee \
  --model_path outputs/run01/final \
  --out_dir outputs/run01/eval_report
```

Produces **`metrics.json`** (macro F1, sklearn report, confusion matrix) and **`confusion_matrix.png`** if matplotlib is available.

---

## Hyperparameter sweep

Edit **`sweep_presets.json`**, then run (each run trains + evaluates):

```bash
python run_sweep.py --data_root ./data/daisee --output_parent outputs/sweeps \
  --max_train_samples 400 --max_eval_samples 200
```

**`outputs/sweeps/sweep_summary.json`** lists **`macro_f1`** per tag. Remove sample caps for full training.

---

## Kaggle

Use branch **`kaggle-pipeline`** (full deps + evaluation + inference fix). Copy-paste cells from **`kaggle/KAGGLE_RUN.txt`**.

```bash
git clone -b kaggle-pipeline --single-branch https://github.com/gotg-cv/final-project.git
```

---

## Inference

```bash
.venv/bin/python src/inference.py --video_path /path/to/clip.mp4 --model_path outputs/run01/final
```

Default **`--model_path`** is `outputs/daisee_videomae/final` — override if your run directory differs.
