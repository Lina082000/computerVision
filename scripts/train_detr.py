"""
Train an RT-DETR object detector on the AMIA Public Challenge 2026 (VinDr-CXR-based) dataset.

Designed to run standalone (no notebook / no internet needed beyond the initial pretrained-weight
download) as a long, unattended job on the FU Berlin HPC cluster (Curta), submitted via Slurm.

Each invocation is one independent "run" (one model config). To compare multiple DETR variants,
submit this script multiple times with different --run-name / --model / --imgsz, e.g.:

    python train_detr.py --run-name rtdetr-l_640  --model rtdetr-l.pt --imgsz 640
    python train_detr.py --run-name rtdetr-l_1024 --model rtdetr-l.pt --imgsz 1024
    python train_detr.py --run-name rtdetr-x_1024 --model rtdetr-x.pt --imgsz 1024

Each run writes everything needed to compare it against the others later (weights, Ultralytics'
own plots, a custom metrics_summary.json using the competition's actual metric, and example
prediction images) under <work-dir>/runs/<run-name>/.

Competition evaluation metric (confirmed via Kaggle competition page): PASCAL VOC-style mean
Average Precision at IoU > 0.4 (mAP@0.4) — NOT the COCO-style mAP50-95 that Ultralytics reports
by default, and NOT simply "IoU threshold used for NMS" (the `iou=` argument to model.val()/
model.predict() controls NMS box suppression, it does NOT change which IoU threshold AP is
computed at). Because of this mismatch, this script computes mAP@0.4 explicitly via torchmetrics,
in addition to the standard Ultralytics metrics (kept for diagnostic/comparison purposes).
"""

import argparse
import json
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split


# --------------------------------------------------------------------------------------
# Constants tied to the dataset (see kaggle_notebooks/biovision-project.ipynb for the EDA
# that these are based on)
# --------------------------------------------------------------------------------------
NO_FINDING = 14
NC = 14  # number of real abnormality classes (No finding is excluded from detection training)
COMPETITION_IOU = 0.4  # the actual leaderboard metric: mAP @ IoU > 0.4


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--run-name", required=True, help="Unique name for this run, e.g. rtdetr-l_640")
    p.add_argument("--model", default="rtdetr-l.pt", choices=["rtdetr-l.pt", "rtdetr-x.pt"],
                   help="Ultralytics RT-DETR checkpoint to start from (l=large, x=extra-large)")

    p.add_argument("--data-dir", default=str(Path.home() / "dogs_project" / "cv" / "data"),
                   help="Folder containing train.csv, img_size.csv, train/train/*.png etc.")
    p.add_argument("--work-dir", default=str(Path.home() / "dogs_project" / "cv" / "detr_work"),
                   help="Scratch folder for the YOLO-format dataset copy (labels + symlinked images)")
    p.add_argument("--out-dir", default=str(Path.home() / "dogs_project" / "cv" / "runs"),
                   help="Where run outputs (weights, plots, metrics) are written")

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=1024,
                   help="Images are natively much larger and non-square (see EDA); 1024 keeps most "
                        "detail while staying tractable on a single GPU overnight.")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--patience", type=int, default=15, help="Early-stopping patience (epochs)")
    p.add_argument("--workers", type=int, default=8, help="Dataloader workers (Curta has far more "
                   "CPU cores available per node than Kaggle's 2)")
    p.add_argument("--max-hours", type=float, default=None,
                   help="Optional wall-clock budget; Ultralytics stops gracefully and keeps last.pt "
                        "if training would exceed it (use to leave headroom before a Slurm time limit)")

    p.add_argument("--val-conf", type=float, default=0.001,
                   help="Confidence threshold for the *custom* mAP@0.4 evaluation pass. Kept low "
                        "(near-zero) like standard COCO/VOC eval protocols so the full precision-recall "
                        "curve is captured; NOT the threshold you'd use for a real submission.")
    p.add_argument("--wbf-iou-thr", type=float, default=0.5,
                   help="IoU threshold used to fuse overlapping boxes from different radiologists "
                        "(Weighted Boxes Fusion) before writing training labels")
    p.add_argument("--no-wbf", action="store_true",
                   help="Disable multi-radiologist box fusion (keep every rad_id's box as a separate "
                        "training instance, i.e. the original notebook's behaviour)")

    p.add_argument("--subset", type=int, default=None,
                   help="Debug/smoke-test only: use just N training images instead of the full set")
    p.add_argument("--resume", action="store_true",
                   help="Resume this run from its own last.pt checkpoint if one exists")
    p.add_argument("--n-example-preds", type=int, default=12,
                   help="Number of validation images to save qualitative GT-vs-prediction plots for")

    return p.parse_args()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# Dataset preparation: raw VinDr-style CSV -> YOLO-format labels, with WBF fusion of
# overlapping multi-radiologist boxes (see biovision-project.ipynb, "Mehrere Radiologen
# pro Bild" section, for why this matters).
# --------------------------------------------------------------------------------------

def fuse_multi_rater_boxes(group, img_w, img_h, iou_thr):
    """Fuse overlapping ground-truth boxes from multiple radiologists for one image using WBF.

    Each radiologist is treated as one "model" in the ensemble (all with weight/confidence 1.0,
    since these are human ground-truth annotations, not model predictions). Boxes are fused
    per class implicitly (weighted_boxes_fusion keeps labels attached and only fuses same-label
    boxes with sufficient IoU overlap).
    """
    from ensemble_boxes import weighted_boxes_fusion

    boxes_by_rad, scores_by_rad, labels_by_rad = [], [], []
    for _, rad_group in group.groupby("rad_id"):
        boxes = rad_group[["x_min", "y_min", "x_max", "y_max"]].to_numpy(dtype=float)
        norm = boxes / [img_w, img_h, img_w, img_h]
        norm = norm.clip(0.0, 1.0)
        boxes_by_rad.append(norm.tolist())
        scores_by_rad.append([1.0] * len(rad_group))
        labels_by_rad.append(rad_group["class_id"].tolist())

    fused_boxes, _, fused_labels = weighted_boxes_fusion(
        boxes_by_rad, scores_by_rad, labels_by_rad,
        iou_thr=iou_thr, skip_box_thr=0.0,
    )
    fused_boxes = fused_boxes * [img_w, img_h, img_w, img_h]
    return fused_boxes, fused_labels


def prepare_yolo_dataset(train_csv, img_size_csv, train_img_dir, work_dir, class_names,
                          use_wbf, wbf_iou_thr, subset=None, seed=42):
    df = pd.read_csv(train_csv)
    img_size_df = pd.read_csv(img_size_csv)
    df = df.merge(img_size_df, on="image_id", how="left")

    image_ids = sorted(p.stem for p in Path(train_img_dir).glob("*.png"))
    if subset:
        rng = random.Random(seed)
        image_ids = rng.sample(image_ids, min(subset, len(image_ids)))

    train_ids, val_ids = train_test_split(image_ids, test_size=0.15, random_state=seed)

    work_dir = Path(work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)

    n_fused_from, n_fused_to = 0, 0

    for split, ids in [("train", train_ids), ("val", val_ids)]:
        img_out = work_dir / "images" / split
        label_out = work_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        id_set = set(ids)
        split_df = df[df["image_id"].isin(id_set)]

        for image_id, group in split_df.groupby("image_id"):
            src_img = Path(train_img_dir) / f"{image_id}.png"
            dst_img = img_out / f"{image_id}.png"
            if not dst_img.exists():
                dst_img.symlink_to(src_img)

            findings = group[group["class_id"] != NO_FINDING]
            lines = []

            if len(findings) > 0:
                img_h = float(findings["dim0"].iloc[0])
                img_w = float(findings["dim1"].iloc[0])

                if use_wbf and split == "train":
                    n_fused_from += len(findings)
                    boxes, labels = fuse_multi_rater_boxes(findings, img_w, img_h, wbf_iou_thr)
                    n_fused_to += len(boxes)
                else:
                    boxes = findings[["x_min", "y_min", "x_max", "y_max"]].to_numpy(dtype=float)
                    labels = findings["class_id"].tolist()

                for (x_min, y_min, x_max, y_max), class_id in zip(boxes, labels):
                    x_min = max(0.0, min(x_min, img_w - 1))
                    y_min = max(0.0, min(y_min, img_h - 1))
                    x_max = max(0.0, min(x_max, img_w - 1))
                    y_max = max(0.0, min(y_max, img_h - 1))
                    if x_max <= x_min or y_max <= y_min:
                        continue
                    xc = ((x_min + x_max) / 2) / img_w
                    yc = ((y_min + y_max) / 2) / img_h
                    bw = (x_max - x_min) / img_w
                    bh = (y_max - y_min) / img_h
                    lines.append(f"{int(class_id)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            # Images with no (or only "No finding") annotations get an empty label file:
            # they stay in training as explicit negative examples, they are not dropped.
            (label_out / f"{image_id}.txt").write_text("\n".join(lines))

    names_text = "\n".join(f"  {cid}: {name}" for cid, name in sorted(class_names.items()))
    yaml_path = work_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {work_dir}\ntrain: images/train\nval: images/val\nnc: {NC}\nnames:\n{names_text}\n"
    )

    if use_wbf and n_fused_from:
        log(f"WBF fusion (train split): {n_fused_from} raw rater boxes -> {n_fused_to} fused boxes "
            f"({n_fused_from - n_fused_to} duplicates merged)")

    return yaml_path, len(train_ids), len(val_ids)


# --------------------------------------------------------------------------------------
# Custom evaluation at the competition's actual metric: mAP @ IoU > 0.4
# --------------------------------------------------------------------------------------

def load_yolo_labels(label_path, img_w, img_h):
    boxes, labels = [], []
    if label_path.exists() and label_path.stat().st_size > 0:
        for line in label_path.read_text().splitlines():
            cls_id, xc, yc, bw, bh = map(float, line.split())
            boxes.append([
                (xc - bw / 2) * img_w, (yc - bh / 2) * img_h,
                (xc + bw / 2) * img_w, (yc + bh / 2) * img_h,
            ])
            labels.append(int(cls_id))
    return boxes, labels


def compute_map_at_iou(model, val_img_dir, val_label_dir, imgsz, class_names,
                        iou_threshold=COMPETITION_IOU, conf=0.001):
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    metric = MeanAveragePrecision(iou_thresholds=[iou_threshold], class_metrics=True, box_format="xyxy")

    preds, targets = [], []
    img_paths = sorted(Path(val_img_dir).glob("*.png"))

    for img_path in img_paths:
        result = model.predict(source=str(img_path), imgsz=imgsz, conf=conf, iou=0.5, verbose=False)[0]
        preds.append({
            "boxes": result.boxes.xyxy.cpu(),
            "scores": result.boxes.conf.cpu(),
            "labels": result.boxes.cls.cpu().int(),
        })

        with Image.open(img_path) as im:
            w, h = im.size
        label_path = Path(val_label_dir) / f"{img_path.stem}.txt"
        gt_boxes, gt_labels = load_yolo_labels(label_path, w, h)
        targets.append({
            "boxes": torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(gt_labels, dtype=torch.int64) if gt_labels else torch.zeros((0,), dtype=torch.int64),
        })

    metric.update(preds, targets)
    result = metric.compute()

    per_class = {}
    if "classes" in result and result["map_per_class"].ndim > 0 and result["map_per_class"].numel() > 1:
        for cls_tensor, ap in zip(result["classes"], result["map_per_class"]):
            cid = int(cls_tensor.item())
            per_class[class_names.get(cid, str(cid))] = float(ap.item())

    return {
        f"map_at_{iou_threshold}": float(result["map"].item()),
        "n_val_images": len(img_paths),
        "per_class_ap": per_class,
    }


# --------------------------------------------------------------------------------------
# Qualitative example predictions (saved as PNGs, not just shown inline — this runs
# headless on the cluster)
# --------------------------------------------------------------------------------------

def save_example_predictions(model, val_img_dir, out_dir, imgsz, class_names, n=12, conf=0.25, seed=42):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(Path(val_img_dir).glob("*.png"))
    rng = random.Random(seed)
    sample = rng.sample(img_paths, min(n, len(img_paths)))

    for img_path in sample:
        result = model.predict(source=str(img_path), imgsz=imgsz, conf=conf, iou=0.4, verbose=False)[0]
        with Image.open(img_path) as im:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(im, cmap="gray")

            label_path = Path(val_img_dir).parent.parent / "labels" / "val" / f"{img_path.stem}.txt"
            w, h = im.size
            gt_boxes, gt_labels = load_yolo_labels(label_path, w, h)
            for (x1, y1, x2, y2), cid in zip(gt_boxes, gt_labels):
                ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                                linewidth=2, edgecolor="lime", facecolor="none"))
                ax.text(x1, max(y1 - 5, 10), class_names.get(cid, str(cid)),
                        color="lime", fontsize=7, bbox=dict(facecolor="black", alpha=0.6, pad=1))

            for box, cls, score in zip(result.boxes.xyxy.cpu(), result.boxes.cls.cpu(), result.boxes.conf.cpu()):
                x1, y1, x2, y2 = box.tolist()
                ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                                linewidth=2, edgecolor="red", facecolor="none"))
                ax.text(x1, min(y2 + 12, h - 5), f"{class_names.get(int(cls.item()), '?')}: {score:.2f}",
                        color="red", fontsize=7, bbox=dict(facecolor="black", alpha=0.6, pad=1))

            ax.set_title(f"{img_path.stem}  (green=GT, red=pred, conf>={conf})", fontsize=9)
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(out_dir / f"{img_path.stem}.png", dpi=120)
            plt.close(fig)


# --------------------------------------------------------------------------------------
# Environment/versions, for the report's "Reproducibility" section
# --------------------------------------------------------------------------------------

def collect_environment_info():
    def pip_version(pkg):
        try:
            import importlib.metadata as m
            return m.version(pkg)
        except Exception:
            return "unknown"

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "ultralytics": pip_version("ultralytics"),
        "ensemble_boxes": pip_version("ensemble-boxes"),
        "torchmetrics": pip_version("torchmetrics"),
    }


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    args = parse_args()
    from ultralytics import RTDETR

    data_dir = Path(args.data_dir)
    train_img_dir = data_dir / "train" / "train"
    train_csv = data_dir / "train.csv"
    img_size_csv = data_dir / "img_size.csv"

    work_dir = Path(args.work_dir) / args.run_name
    out_dir = Path(args.out_dir)
    run_dir = out_dir / args.run_name

    class_df = pd.read_csv(train_csv)[["class_id", "class_name"]].drop_duplicates()
    class_names = {int(r.class_id): r.class_name for r in class_df.itertuples()
                   if int(r.class_id) != NO_FINDING}

    log(f"Run '{args.run_name}': model={args.model} imgsz={args.imgsz} epochs={args.epochs} "
        f"batch={args.batch} wbf={not args.no_wbf}")

    log("Preparing YOLO-format dataset (with multi-radiologist WBF fusion)..." if not args.no_wbf
        else "Preparing YOLO-format dataset (WBF disabled)...")
    yaml_path, n_train, n_val = prepare_yolo_dataset(
        train_csv, img_size_csv, train_img_dir, work_dir, class_names,
        use_wbf=not args.no_wbf, wbf_iou_thr=args.wbf_iou_thr, subset=args.subset,
    )
    log(f"Dataset ready: {n_train} train / {n_val} val images -> {yaml_path}")

    device = 0 if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log("WARNING: no CUDA GPU visible — check that this job actually landed on a GPU node "
            "(e.g. `--gres=gpu:1` in the Slurm script and a GPU partition).")

    # AMP (automatic mixed precision) relies on FP16 Tensor Cores, introduced with Volta
    # (compute capability 7.0). Pre-Volta GPUs (e.g. Pascal / GTX 1080 Ti, CC 6.1) have no Tensor
    # Cores, gain no speedup from AMP, and were observed to crash with "RuntimeError: GET was
    # unable to find an engine to execute this computation" during Ultralytics' AMP sanity check —
    # cuDNN has no FP16 conv engine for this architecture under the newer cuDNN v8 API. Detecting
    # this and disabling AMP accordingly lets the same script run unmodified on whichever GPU Slurm
    # assigns, and is itself a documented finding for the report's discussion of AMP applicability.
    use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7
    if torch.cuda.is_available() and not use_amp:
        cc_major, cc_minor = torch.cuda.get_device_capability(0)
        log(f"GPU compute capability {cc_major}.{cc_minor} has no Tensor Cores (needs >=7.0) — "
            f"disabling AMP (would crash / gives no benefit on this hardware).")
        # Turns out disabling AMP alone wasn't enough on this hardware: the SAME
        # "RuntimeError: GET was unable to find an engine to execute this computation" also hit a
        # plain FP32 conv during Ultralytics' batch-size probing. Setting
        # TORCH_CUDNN_V8_API_DISABLED=1 (falls back to cuDNN's legacy algorithm-search path) changed
        # the error message ("Unable to find a valid cuDNN algorithm to run convolution") but did NOT
        # fix it — confirmed directly on a GTX 1080 Ti node: cuDNN 9.2 (bundled with torch==2.5.1) has
        # no working convolution engine for compute capability 6.1 (Pascal) at all, on EITHER API path.
        # The only combination that actually ran a Conv2d forward pass on this hardware (verified
        # interactively) was disabling cuDNN entirely and falling back to PyTorch's native (non-cuDNN)
        # conv kernels. Slower than cuDNN would be, but the alternative is not training at all. Must be
        # set before any convolution runs (cuDNN initializes lazily on first use), so set it here,
        # before the model is created / moved to GPU.
        torch.backends.cudnn.enabled = False
        log("Also disabling cuDNN entirely (torch.backends.cudnn.enabled = False) — cuDNN 9.x has no "
            "working conv engine for this GPU's compute capability, confirmed interactively.")

    model = RTDETR(args.model)

    last_ckpt = run_dir / "weights" / "last.pt"
    resume = args.resume and last_ckpt.exists()
    if args.resume and not resume:
        log(f"--resume given but no checkpoint found at {last_ckpt}, starting fresh instead.")

    start_time = time.time()
    train_kwargs = dict(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(out_dir),
        name=args.run_name,
        exist_ok=True,
        patience=args.patience,
        workers=args.workers,
        device=device,
        plots=True,
        amp=use_amp,    # automatic mixed precision (VL day10_optimization): halves memory, speeds
                         # up training on modern GPUs with negligible accuracy impact — auto-disabled
                         # on GPUs without Tensor Cores, see check above
        fliplr=0.0,      # NO horizontal flip: chest X-ray laterality is diagnostically meaningful
                         # (e.g. cardiac silhouette, aortic enlargement) — flipping would create
                         # anatomically implausible / mislabeled training images
        flipud=0.0,
        mosaic=0.5,      # keep some mosaic augmentation but reduced: full-strength mosaic can create
                         # anatomically nonsensical composite chest X-rays
        mixup=0.0,       # same reasoning: mixing two chest X-rays is not clinically meaningful
        hsv_h=0.0, hsv_s=0.0,  # images are grayscale; hue/saturation jitter has no signal to act on
    )
    if args.max_hours is not None:
        train_kwargs["time"] = args.max_hours
    if resume:
        train_kwargs["resume"] = True
        log(f"Resuming from {last_ckpt}")

    model.train(**train_kwargs)
    training_time_h = (time.time() - start_time) / 3600
    log(f"Training finished in {training_time_h:.2f} h")

    best_ckpt = run_dir / "weights" / "best.pt"
    model = RTDETR(str(best_ckpt))
    log(f"Loaded best checkpoint: {best_ckpt}")

    log("Running standard Ultralytics validation (diagnostic only — NOT the competition metric)...")
    # project/name pinned into this run's own folder: without them Ultralytics defaults to
    # runs/detect/val relative to the cwd, which every parallel run-name job (all launched with the
    # same cwd, see sbatch script) would share and race/overwrite.
    std_metrics = model.val(data=str(yaml_path), imgsz=args.imgsz, batch=args.batch, conf=0.05, iou=0.40,
                             project=str(run_dir), name="val_diagnostic", exist_ok=True)

    log(f"Computing competition metric mAP@{COMPETITION_IOU} via torchmetrics "
        f"(this re-runs inference over the full validation set, may take a while)...")
    comp_metrics = compute_map_at_iou(
        model, work_dir / "images" / "val", work_dir / "labels" / "val",
        imgsz=args.imgsz, class_names=class_names, iou_threshold=COMPETITION_IOU, conf=args.val_conf,
    )

    log(f"Saving {args.n_example_preds} example prediction plots...")
    save_example_predictions(
        model, work_dir / "images" / "val", run_dir / "example_predictions",
        imgsz=args.imgsz, class_names=class_names, n=args.n_example_preds,
    )

    summary = {
        "run_name": args.run_name,
        "model": args.model,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "wbf_enabled": not args.no_wbf,
        "wbf_iou_thr": args.wbf_iou_thr,
        "n_train_images": n_train,
        "n_val_images": n_val,
        "training_time_h": round(training_time_h, 3),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": collect_environment_info(),
        "ultralytics_val_metrics": {
            "precision": float(std_metrics.box.mp),
            "recall": float(std_metrics.box.mr),
            "map50": float(std_metrics.box.map50),
            "map50_95": float(std_metrics.box.map),
        },
        "competition_metric": comp_metrics,
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log(f"Done. Summary written to {run_dir / 'metrics_summary.json'}")
    log(f"Competition metric mAP@{COMPETITION_IOU}: {comp_metrics[f'map_at_{COMPETITION_IOU}']:.4f}")
    log(f"(For reference) Ultralytics mAP50-95: {summary['ultralytics_val_metrics']['map50_95']:.4f}, "
        f"mAP50: {summary['ultralytics_val_metrics']['map50']:.4f}")


if __name__ == "__main__":
    main()
