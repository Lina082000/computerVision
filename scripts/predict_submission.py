"""
Run inference with a trained RT-DETR checkpoint over the competition's test set and
write a Kaggle submission.csv in the required format (image_id, PredictionString).

PredictionString format (confirmed from data/sample_submission.csv): for each predicted box,
    "<class_id> <confidence> <x_min> <y_min> <x_max> <y_max>"
space-separated, all boxes for one image concatenated with a single space. Images with no
detection above --conf get the dataset's "no finding" placeholder, matching sample_submission.csv:
    "14 1 0 0 1 1"

Confidence threshold is kept low by default (--conf 0.001), matching the training script's own
mAP@0.4 evaluation (train_detr.py's compute_map_at_iou uses conf=0.001) — mAP is computed over the
whole precision-recall curve, so filtering out low-confidence boxes before submitting can only hurt
the score, never help it.

IMPORTANT coordinate rescaling: every PNG under data/train and data/test is stored resized to a
fixed 1024x1024, but img_size.csv (and the x_min/y_min/x_max/y_max columns in train.csv) refer to
the *original* DICOM pixel dimensions (non-square, e.g. 2880x2304) that the competition's own
ground truth is defined in. model.predict() returns box coordinates in the pixel space of the file
it was given (1024x1024), so those must be rescaled per-axis back to the original (dim1, dim0)
=(width, height) from img_size.csv before writing the submission — otherwise every box is off by
roughly a factor of 2-3x and lands nowhere near the true box (confirmed: an unscaled submission
scored ~0.002 instead of the ~0.4 the local validation split predicted).

Usage (mirrors train_detr.py's --run-name convention):
    python scripts/predict_submission.py --run-name rtdetr-l_640 --imgsz 640
    python scripts/predict_submission.py --run-name rtdetr-l_640 --imgsz 640 --limit 20   # smoke test
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", default=None, help="Run under --runs-dir to load weights/best.pt from")
    p.add_argument("--weights", default=None, help="Explicit checkpoint path (overrides --run-name)")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--imgsz", type=int, required=True,
                   help="Must match the imgsz the checkpoint was trained with (640 for rtdetr-l_640, "
                        "1024 for rtdetr-l_1024 / rtdetr-x_1024)")
    p.add_argument("--conf", type=float, default=0.001, help="Confidence threshold kept for the submission")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--device", default=None, help="e.g. 'cpu' or '0'; default = auto")
    p.add_argument("--limit", type=int, default=None, help="Only run on the first N test images (smoke test)")
    p.add_argument("--log-every", type=int, default=200, help="Log progress every N images")
    return p.parse_args()


def main():
    args = parse_args()

    if args.weights:
        weights = Path(args.weights)
    else:
        if not args.run_name:
            raise SystemExit("Pass either --run-name or --weights")
        weights = Path(args.runs_dir) / args.run_name / "weights" / "best.pt"
    if not weights.exists():
        raise SystemExit(f"Checkpoint not found: {weights}")

    if torch.cuda.is_available() and torch.cuda.get_device_capability(0) < (7, 0):
        # Same Pascal/cuDNN-9 incompatibility as training (see train_detr.py) — hits inference too.
        torch.backends.cudnn.enabled = False
        log("Pascal-era GPU detected, disabling cuDNN (matches training-time workaround).")
    elif not torch.cuda.is_available():
        log("No CUDA GPU visible — running on CPU (fine for --limit smoke tests, slow for the full test set).")

    from ultralytics import RTDETR
    log(f"Loading {weights}")
    model = RTDETR(str(weights))

    data_dir = Path(args.data_dir)
    sample_sub = pd.read_csv(data_dir / "sample_submission.csv")
    image_ids = sample_sub["image_id"].tolist()
    if args.limit:
        image_ids = image_ids[: args.limit]
    test_img_dir = data_dir / "test" / "test"

    img_size_df = pd.read_csv(data_dir / "img_size.csv").set_index("image_id")

    log(f"Running inference on {len(image_ids)} test images "
        f"(imgsz={args.imgsz}, conf={args.conf}, iou={args.iou})")

    rows = []
    n_empty = 0
    for i, image_id in enumerate(image_ids):
        img_path = test_img_dir / f"{image_id}.png"
        if not img_path.exists():
            raise SystemExit(f"Missing test image: {img_path}")

        result = model.predict(source=str(img_path), imgsz=args.imgsz, conf=args.conf,
                                iou=args.iou, verbose=False, device=args.device)[0]

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            pred_string = "14 1 0 0 1 1"
            n_empty += 1
        else:
            with Image.open(img_path) as im:
                file_w, file_h = im.size
            orig_h = float(img_size_df.loc[image_id, "dim0"])
            orig_w = float(img_size_df.loc[image_id, "dim1"])
            scale_x = orig_w / file_w
            scale_y = orig_h / file_h

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            tokens = [
                f"{cls_id} {c:.4f} {x1 * scale_x:.1f} {y1 * scale_y:.1f} {x2 * scale_x:.1f} {y2 * scale_y:.1f}"
                for (x1, y1, x2, y2), c, cls_id in zip(xyxy, confs, clss)
            ]
            pred_string = " ".join(tokens)

        rows.append({"image_id": image_id, "PredictionString": pred_string})

        if (i + 1) % args.log_every == 0:
            pct = 100.0 * (i + 1) / len(image_ids)
            log(f"  {i + 1}/{len(image_ids)} done ({pct:.1f}%)")

    sub_df = pd.DataFrame(rows)
    sub_df.to_csv(args.out, index=False)
    log(f"Wrote {args.out} ({len(sub_df)} rows, {n_empty} predicted as 'no finding')")


if __name__ == "__main__":
    main()
