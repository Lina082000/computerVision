# Chest X-Ray Abnormality Detection with RT-DETR

**AMIA Public Challenge 2026 — Model section: DETR (RT-DETR)**
Author: Lina Polenz · Course: Computer Vision for Biomedical Images, SoSe 2026, Freie Universität Berlin

> This is a **skeleton**, filled in wherever information is already known from the dataset/code/lecture
> material. Sections that depend on actual training runs (Results, most of Discussion) are marked
> `TODO` — the plan is to run the model overnight on Curta first (see `scripts/train_detr.py` /
> `slurm/train_detr.sbatch`), then fill these in from the outputs in `runs/<run-name>/`.

---

## Abstract

*TODO once results exist.*
- What was the goal of the project?
- What was done?
- What are the results?

---

## Introduction

### Problem statement

The AMIA Public Challenge 2026 asks participants to detect and classify 14 categories of thoracic
abnormalities (e.g. cardiomegaly, aortic enlargement, pleural effusion, ...) in chest X-rays, or to
predict "No finding" if none are present. It is a multi-label, multi-instance **object detection**
task (not classification): each image can contain zero, one, or many bounding boxes across multiple
classes, annotated independently by up to three radiologists per image.

### Explanation of data

The dataset is derived from **VinDr-CXR** (Nguyen et al., 2022) [1]. Key characteristics (see
`kaggle_notebooks/biovision-project.ipynb` for the full EDA and `biovision-project.txt` for a summary
of what was added there):

- 8,573 training images, 15 classes (14 abnormalities + "No finding"), each read independently by
  exactly 3 radiologists.
- **60.4%** of images are "No finding" only — a substantial class/background imbalance.
- Abnormality classes are themselves imbalanced ~28:1 (most vs. least frequent).
- Native image resolutions are **heterogeneous** (e.g. 2430×1994, 3072×3072, 2880×2304 — not a fixed
  1024×1024 as one might assume from some preprocessing code); bounding box coordinates are given in
  each image's own native pixel grid.
- ~5.8% of annotated boxes qualify as "small objects" (<32×32 px, COCO convention) — relevant since
  transformer-based detectors are known to be comparatively weak on small objects (VL "Object
  Detection", RetinaNet feature-pyramid discussion [2]).
- Because 3 radiologists annotate independently, the same finding is often marked by more than one
  rater with overlapping-but-not-identical boxes (inter-rater variability).

### Explanation of techniques used

**RT-DETR** (Real-Time DEtection TRansformer, via the Ultralytics implementation) was chosen as the
DETR variant to train and evaluate. Background/context:

- The original **DETR** (Carion et al., 2020 [3]) reframed object detection as a set-prediction
  problem: a CNN backbone extracts features, a transformer encoder-decoder (self-attention +
  cross-attention, as covered in VL "Transformers" [4]) attends over the whole image, and a fixed
  number of learned object queries are matched to ground-truth boxes via bipartite (Hungarian)
  matching — removing the need for hand-designed anchors and NMS post-processing that classical
  detectors like Faster R-CNN / YOLO / RetinaNet rely on (VL "Object Detection" [2]).
- **RT-DETR** (Lv et al., 2023/2024, "DETRs Beat YOLOs on Real-Time Object Detection" [5]) is a
  real-time-oriented evolution of DETR: an efficient hybrid CNN+transformer encoder and an
  uncertainty-minimal query selection scheme give YOLO-competitive inference speed while keeping
  DETR's NMS-free, end-to-end design. It was chosen over vanilla DETR mainly for practical reasons —
  faster convergence and a maintained, well-documented training pipeline (Ultralytics), which matters
  given the limited overnight compute budget on the cluster.
- Positional encoding: like all transformer architectures, RT-DETR needs explicit positional
  information (sinusoidal position embeddings), since attention itself is permutation-invariant and,
  unlike CNNs, has no built-in spatial inductive bias (VL "Transformers", positional encoding section
  [4]).

---

## Results

*TODO — to be filled in from `runs/<run-name>/metrics_summary.json` and the Ultralytics-generated plots
(`results.png`, `confusion_matrix.png`, `PR_curve.png`, ...) once training has completed on the
cluster. Structure prepared below.*

### Dataset characteristics and mitigation strategies

*(Reasoning already available from EDA — fill in final numbers/plots once re-run on the full set.)*

| Characteristic | Effect on training | Mitigation implemented |
|---|---|---|
| 60.4% "No finding" images | Model could learn to always predict nothing | Empty images kept as explicit negatives (not dropped); RT-DETR's classification loss is a focal/varifocal-style loss that down-weights easy negatives |
| ~28:1 class imbalance among abnormalities | Rare classes under-represented in gradient signal | Documented as limitation; no explicit resampling (would be ambiguous at the multi-label, multi-box image level) — see Discussion |
| Multiple radiologists per image, overlapping boxes | Redundant/conflicting supervision for the same finding | Weighted Boxes Fusion (WBF) merges overlapping rater boxes before training (`scripts/train_detr.py::fuse_multi_rater_boxes`) |
| Heterogeneous native resolutions | Naive fixed-size resizing distorts aspect ratio / box coordinates differently per image | Per-image normalization using each image's own `dim0`/`dim1` when building YOLO-format labels |
| Chest X-ray laterality is diagnostically meaningful | Horizontal flip augmentation would produce anatomically implausible / mislabeled images | `fliplr=0.0` in training config |
| ~5.8% small objects (<32×32px) | Known weak point of transformer detectors | Trained at higher `imgsz` (1024) where compute allowed, to preserve small-lesion detail |

### Overview of design and training process

- Architecture: RT-DETR-L / RT-DETR-X (Ultralytics), COCO-pretrained weights as initialization.
- Multiple variants trained independently and compared (see `runs/*/metrics_summary.json`):
  *TODO: list actual run names once submitted, e.g. rtdetr-l_640, rtdetr-l_1024, rtdetr-x_1024.*
- Optimized for: the competition's own metric, **mAP @ IoU > 0.4** (PASCAL VOC-2010-style), computed
  explicitly via `torchmetrics` — see "Explain what you measured" below for why this needed a custom
  implementation.

### Show examples

*TODO: embed a few images from `runs/<run-name>/example_predictions/` (green = ground truth, red =
prediction).*

### Explain what you measured (metrics) and to what end

Two distinct sets of metrics are reported per run, deliberately kept apart:

1. **Ultralytics' built-in validation metrics** (precision, recall, mAP50, mAP50-95): useful for
   comparing training dynamics and for sanity-checking against typical Ultralytics benchmarks, but
   **mAP50-95 is the COCO convention** (averaged over IoU thresholds 0.5 to 0.95) — it is *not* the
   metric the competition leaderboard actually uses.
2. **Competition metric, mAP@IoU>0.4** (PASCAL VOC 2010-style mean Average Precision at a single IoU
   threshold of 0.4), computed explicitly with `torchmetrics.detection.MeanAveragePrecision`.

**A difficulty worth spelling out explicitly:** the original notebook already passed `iou=0.40` to
Ultralytics' `model.val()`/`model.predict()`. It is tempting to assume this controls the IoU threshold
at which Average Precision is computed — **it does not**. In Ultralytics, the `iou` argument to
`val()`/`predict()` sets the **NMS suppression threshold** (how aggressively overlapping predicted
boxes are merged/discarded before scoring), not the matching threshold used when comparing predictions
to ground truth for AP. `map50` and `map` (mAP50-95) are computed at fixed, hardcoded IoU thresholds
regardless of the `iou=` argument. Relying on the default Ultralytics output alone would therefore have
silently reported the wrong metric relative to the leaderboard. This is exactly the kind of
metric-definition pitfall worth being explicit about, and is why this project computes mAP@0.4
separately and independently.

**Why IoU>0.4 specifically (and not the stricter COCO default of 0.5):** a lower IoU threshold is more
forgiving of small, boundary-ambiguous discrepancies between predicted and ground-truth boxes — plausible
given the inter-rater variability documented above (the "ground truth" itself is somewhat fuzzy when
three radiologists don't draw identical boxes for the same finding). Any pixel-unit metric is computed
per image at that image's own native resolution (see dataset characteristics above), not a shared fixed
canvas size.

---

## Discussion

*TODO once results exist.*

- Particular challenges with data/task (draft, to expand with concrete numbers): severe class
  imbalance, inter-rater box disagreement, small-object detection, heterogeneous image resolutions.
- Reflection of results (what went wrong / right / why).
- Comparison to other (published/winning) approaches on VinDr-CXR / similar challenges — *TODO,
  research top Kaggle/VinBigData-competition solutions once own results exist for a fair comparison
  baseline.*
- What could be done better next time (draft candidates worth revisiting once a first result exists):
  - Try the original DETR (Carion et al., 2020) as an additional comparison point — the starting
    notebook already noted this as an alternative but it was out of scope given the available time.
  - Explore class-balanced sampling or loss re-weighting beyond RT-DETR's built-in focal-style loss.
  - Explainability pass (GradCAM / attention-map visualization, VL "QC & Explainability" [6]) to sanity
    check whether the model is actually attending to the annotated lesion regions rather than spurious
    background cues — directly relevant for medical imaging trustworthiness.

---

## Methods

### Dataset description

- Source: AMIA Public Challenge 2026 (Kaggle), based on VinDr-CXR [1].
- 8,573 training images (chest X-rays, grayscale, heterogeneous native resolution), 15,000 images
  total including the (unlabeled) test split.
- 14 abnormality classes + "No finding"; each image independently annotated by exactly 3 radiologists.
- Annotation completeness: every image has at least one entry per radiologist (either a "No finding"
  row or ≥1 bounding box); no missing/corrupted image files found in a full scan of the training set.
- Quality/imbalance: see "Dataset characteristics" table above.

### Architecture description, incl. hyperparameters

- RT-DETR-L and RT-DETR-X (Ultralytics implementation), COCO-pretrained initialization.
- Training image size: up to 1024 px (vs. the 640 px used in the original exploratory Kaggle notebook)
  — chosen because native resolutions go up to ~3072×3072 and the small-object analysis showed ~5.8%
  of boxes are small; more compute is available on Curta than on the free Kaggle GPU tier.
- Batch size: 8 (cluster) vs. 4 (original Kaggle notebook, GPU-memory constrained there).
- Automatic Mixed Precision (AMP) enabled (VL "Hardware Optimization" [7]) to reduce memory footprint
  and increase throughput, allowing the larger batch/image size above within the same GPU memory
  budget.
- Augmentation: horizontal/vertical flip disabled (`fliplr=0.0`, `flipud=0.0` — chest X-ray laterality
  is diagnostically meaningful); mosaic reduced (0.5) rather than disabled outright; mixup disabled
  (blending two chest X-rays has no clinical meaning); HSV jitter disabled (images are grayscale).
- Early stopping: patience = 15 epochs without validation improvement.

### Preprocessing, postprocessing

- Multi-radiologist box fusion via **Weighted Boxes Fusion** (`ensemble-boxes`, IoU threshold 0.5)
  before converting to YOLO-format training labels, to avoid feeding multiple near-duplicate boxes for
  the same finding as independent training targets.
- "No finding" class excluded from detection labels (images become negative/background examples with
  an empty label file, not removed from training).
- Per-image box normalization uses each image's own `dim0`/`dim1` (native resolution), not a shared
  constant.
- Postprocessing: NMS at IoU 0.4-0.5 (Ultralytics default pipeline) to remove duplicate detections at
  inference time.

### Training settings

- Optimizer / LR schedule: Ultralytics RT-DETR defaults (AdamW with built-in cosine/linear schedule
  and warmup) — *TODO: confirm/quote exact values actually used from the Ultralytics run config
  (`runs/<run-name>/args.yaml`) once training has run.*
- Loss: RT-DETR's built-in matching + classification (varifocal-style) + box regression (L1 + GIoU)
  losses; Hungarian bipartite matching between predictions and (WBF-fused) ground-truth boxes.
- Epochs: up to 60, early stopping patience 15.
- Hardware budget: single GPU, ≤ ~10.5h wall-clock per run (Slurm time limit minus safety margin),
  see `slurm/train_detr.sbatch`.

### Reproducibility: versions etc.

*Populated automatically per run in `runs/<run-name>/metrics_summary.json["environment"]`* (Python,
PyTorch, CUDA, Ultralytics, ensemble-boxes, torchmetrics versions, GPU model). *TODO: paste the actual
values here once a run has completed.*

---

## Code

- `kaggle_notebooks/biovision-project.ipynb` — EDA / dataset overview (shared across the group, my
  additions documented in `kaggle_notebooks/biovision-project.txt`).
- `scripts/train_detr.py` — standalone training + evaluation script (the actual long-running job on
  Curta); produces `runs/<run-name>/metrics_summary.json`, Ultralytics' own diagnostic plots, and
  qualitative example-prediction images.
- `slurm/train_detr.sbatch` — Slurm submission script; one job = one model variant.
- `kaggle_notebooks/visualize_predictions.ipynb` — qualitative inspection of a single trained run.
- `kaggle_notebooks/compare_models.ipynb` — quantitative comparison across all trained runs
  (mAP@0.4, per-class AP, training curves).

---

## References

[1] H. Q. Nguyen et al., "VinDr-CXR: An open dataset of chest X-rays with radiologist's annotations,"
    *Scientific Data*, 9, 429 (2022). https://doi.org/10.1038/s41597-022-01498-w

[2] R. Girshick et al., "Rich feature hierarchies for accurate object detection and semantic
    segmentation" (R-CNN), CVPR 2014; R. Girshick, "Fast R-CNN," ICCV 2015; S. Ren et al., "Faster
    R-CNN: Towards Real-Time Object Detection with Region Proposal Networks," NeurIPS 2015;
    T.-Y. Lin et al., "Focal Loss for Dense Object Detection" (RetinaNet), ICCV 2017.
    — as covered in: S. Lukassen, *Computer Vision for Biomedical Images*, VL "Object Detection"
    (day6_object_detection.pptx), FU Berlin, SoSe 2026.

[3] N. Carion et al., "End-to-End Object Detection with Transformers" (DETR), ECCV 2020.

[4] A. Vaswani et al., "Attention Is All You Need," NeurIPS 2017.
    — as covered in: S. Lukassen, *Computer Vision for Biomedical Images*, VL "Transformers"
    (day7_transformers.pptx) and VL "QC & Explainability" (day8_qc_explainability.pptx, DETR slides),
    FU Berlin, SoSe 2026.

[5] Y. Zhao / W. Lv et al., "DETRs Beat YOLOs on Real-Time Object Detection" (RT-DETR), CVPR 2024
    (arXiv:2304.08069).

[6] S. Woo et al., "CBAM: Convolutional Block Attention Module," ECCV 2018.
    — as covered in: S. Lukassen, *Computer Vision for Biomedical Images*, VL "QC & Explainability"
    (day8_qc_explainability.pptx), FU Berlin, SoSe 2026.

[7] S. Lukassen, *Computer Vision for Biomedical Images*, VL "Hardware Optimization"
    (day10_optimization.pptx), FU Berlin, SoSe 2026 (automatic mixed precision, quantization,
    hardware considerations).

[8] Kaggle, "AMIA Public Challenge 2026" competition page (evaluation metric, dataset description,
    submission format). https://www.kaggle.com/competitions/amia-public-challenge-2026
