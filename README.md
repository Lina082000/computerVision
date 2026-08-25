# AMIA Public Challenge 2026 — RT-DETR Chest X-ray Detection

Object detection on the AMIA Public Challenge 2026 dataset (a subset of VinBigData Chest X-ray),
using RT-DETR (via Ultralytics) to localize 14 thoracic abnormality classes. Competition metric:
mean Average Precision at IoU > 0.4 (mAP@0.4), computed on the ground-truth boxes fused across
multiple radiologist annotations per image.

## Repo contents

This repo only tracks code and results — **not** the raw dataset or full model
checkpoints (see "What's missing" below). Layout:

```
scripts/
  train_detr.py            Trains one RT-DETR variant end-to-end: builds a YOLO-format dataset
                            from train.csv (with optional WBF box fusion across raters), trains,
                            evaluates at the competition's mAP@0.4, saves plots + weights.
  predict_submission.py     Loads a trained checkpoint, runs inference over the test set, and
                            writes a Kaggle-format submission.csv. Handles the coordinate rescaling
                            described below — read the top-of-file docstring before changing it.
  requirements.txt         Python packages needed for both scripts above (torch is intentionally
                            unpinned — see file for why).

slurm/
  train_detr.sbatch        Submits one train_detr.py run on FU Berlin's Curta HPC cluster.
                            Usage: sbatch train_detr.sbatch <run-name> <model.pt> <imgsz> <batch>
  predict_detr.sbatch      Same, for predict_submission.py (GPU inference is much faster than CPU).
                            Usage: sbatch predict_detr.sbatch <run-name> <imgsz>

kaggle_notebooks/
  compare_models.ipynb     Loads every run's metrics_summary.json + results.csv, builds all the
                            comparison plots (mAP@0.4 per model, precision/recall/mAP50/mAP50-95,
                            per-class AP heatmap, training-time vs. accuracy trade-off, training
                            curves) and a written conclusion. Already executed — open it as-is to
                            see the results without re-running anything.

environment_compare.yml    Minimal conda env to just open and (re-)run compare_models.ipynb
                            (pandas, matplotlib, numpy, jupyterlab, ipykernel — no torch needed).

runs/rtdetr-l_640/         Three trained model variants, each with:
runs/rtdetr-l_1024/          metrics_summary.json   final competition metric + per-class AP + env info
runs/rtdetr-x_1024/          results.csv             per-epoch training log
                              results.png             Ultralytics' own training-curve plot
                              args.yaml               exact hyperparameters used
                              weights/best.pt          trained checkpoint (see exception below)
                              BoxF1/BoxPR/BoxP/BoxR_curve.png, confusion_matrix*.png   diagnostic plots
                              train_batch*.jpg, val_batch*_labels.jpg, val_batch*_pred.jpg   sample batches
                              example_predictions/    qualitative GT-vs-prediction images on val images
                              val_diagnostic/         same diagnostic plots, duplicated by train_detr.py
runs/rtdetr-l_640/submission.csv   the actual file submitted to Kaggle for this run

## Current results (local validation split)

| run            | mAP@0.4 (competition) | mAP50 | mAP50-95 | train time |
|----------------|:---:|:---:|:---:|:---:|
| rtdetr-l_640   | **0.420** | 0.310 | 0.157 | 10.6h |
| rtdetr-l_1024  | 0.380 | 0.277 | 0.143 | 10.6h |
| rtdetr-x_1024  | 0.324 | 0.233 | 0.117 | 10.6h |

`rtdetr-l_640` is the strongest model and the one currently submitted to Kaggle:
**public score 0.375, private score 0.353**. Details/discussion in
`kaggle_notebooks/compare_models.ipynb`.

## What's missing (and how to get it)

Deliberately left out of git (see `.gitignore`) — either too large for GitHub, personal/secret, or
easy to regenerate:

- **`data/`** (~13GB): the competition dataset. Download it yourself:
  ```bash
  kaggle competitions download -c amia-public-challenge-2026 -p data
  # then unzip into data/, so you end up with data/train/, data/test/, data/train.csv, etc.
  ```
  You'll need your own Kaggle API token for this (`kaggle.com` → Settings → "Create New Token"),
  saved to `~/.kaggle/kaggle.json`. **Never commit your own token to this repo.**
- **`runs/*/weights/last.pt`**: the second-to-last checkpoint saved during training (Ultralytics
  keeps both `best.pt` and `last.pt`). Not needed for inference — `best.pt` is what you want.
- **`runs/rtdetr-x_1024/weights/best.pt`** (130MB): the one checkpoint that exceeds GitHub's 100MB
  hard push limit. Shared via the team's OneDrive folder instead.
- **`kaggle_api_token.txt`**: personal Kaggle credential, never share this file with anyone.

If you need the two files above, ask in the team chat / grab them from the shared OneDrive folder.

## Setup

**Just want to look at the results/plots?**
```bash
conda env create -f environment_compare.yml
conda activate compV
jupyter lab kaggle_notebooks/compare_models.ipynb
```

**Want to train or run inference yourself (local or on Curta)?**
```bash
conda create -n detr_env python=3.11
conda activate detr_env
# install the torch build matching your GPU/CUDA first, e.g.:
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r scripts/requirements.txt
```
Then either submit a training job on Curta:
```bash
sbatch slurm/train_detr.sbatch rtdetr-l_640 rtdetr-l.pt 640 4
```
or run inference locally / on Curta (adjust `--imgsz` to match the checkpoint):
```bash
python scripts/predict_submission.py --run-name rtdetr-l_640 --imgsz 640
kaggle competitions submit -c amia-public-challenge-2026 -f runs/rtdetr-l_640/submission.csv -m "..."
```

## Important gotcha (already fixed, but read this before touching predict_submission.py)

Every PNG under `data/train` and `data/test` is stored resized to a fixed **1024x1024**, but
`img_size.csv` (and the box columns in `train.csv`) refer to the **original, non-square DICOM
dimensions** (e.g. 2880x2304) — that's also the coordinate space Kaggle's scorer expects
predictions in. `model.predict()` returns boxes in the pixel space of whatever file it was given
(1024x1024), so `predict_submission.py` rescales every box back to the original size using
`img_size.csv` before writing the submission. Skipping this rescaling is *not* an error Ultralytics
will warn you about — it silently produces a syntactically valid but essentially useless submission
(this happened once already: public score dropped to 0.002 without it, vs. 0.375 with it).
