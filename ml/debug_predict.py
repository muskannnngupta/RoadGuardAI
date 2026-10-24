#!/usr/bin/env python3
"""
ml/debug_predict.py
===================
Debug and comparison utility for RoadGuard AI road damage detection.

Compares:
1. Direct raw YOLOv8 inference: model.predict(conf=0.01, imgsz=640)
2. Production service pipeline inference: DamageDetector.predict(conf_lower=0.01)

Verifies detection parity and highlights any discrepancies.
"""

import sys
import argparse
from pathlib import Path
from PIL import Image

# Ensure inference-service and repository root are on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
INFERENCE_DIR = REPO_ROOT / "inference-service"

if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

# Ultralytics & Service Imports
try:
    from ultralytics import YOLO
except ImportError:
    print("Error: 'ultralytics' package is required. Install via: pip install ultralytics")
    sys.exit(1)

try:
    from app.model import DamageDetector
except ImportError as e:
    print(f"Error importing service DamageDetector: {e}")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare direct raw YOLO inference vs. RoadGuard AI service inference."
    )
    parser.add_argument(
        "images_dir",
        nargs="?",
        default="samples",
        help="Path to folder containing test images (default: 'samples')"
    )
    parser.add_argument(
        "--model-path",
        default="inference-service/models/best.pt",
        help="Path to YOLO model weights (default: 'inference-service/models/best.pt')"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.01,
        help="Confidence threshold for detection filtering (default: 0.01)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640)"
    )
    return parser.parse_args()


def find_images(target_dir: Path):
    """Finds image files directly or recursively within target directory."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted([
        p for p in target_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts
    ])
    if not images:
        images = sorted([
            p for p in target_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in valid_exts
        ])
    return images


def main():
    args = parse_args()

    # Resolve target directory
    img_dir_path = Path(args.images_dir)
    if not img_dir_path.exists():
        # Check relative to REPO_ROOT
        alt_path = REPO_ROOT / args.images_dir
        if alt_path.exists():
            img_dir_path = alt_path
        else:
            print(f"Error: Images directory '{args.images_dir}' not found.")
            sys.exit(1)

    # Resolve model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        alt_model = REPO_ROOT / args.model_path
        if alt_model.exists():
            model_path = alt_model
        else:
            print(f"Error: Model file '{args.model_path}' not found.")
            sys.exit(1)

    images = find_images(img_dir_path)
    if not images:
        print(f"Error: No image files (.jpg, .jpeg, .png) found in '{img_dir_path}'.")
        sys.exit(1)

    print("=" * 85)
    print(" ROADGUARD AI — DIRECT RAW YOLO vs. SERVICE INFERENCE COMPARISON")
    print("=" * 85)
    print(f" Model Weights : {model_path} ({model_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f" Images Dir    : {img_dir_path} ({len(images)} images found)")
    print(f" Threshold     : conf={args.conf}, imgsz={args.imgsz}")
    print("=" * 85)

    # 1. Initialize Direct Raw YOLO Model (pure Ultralytics, no service code)
    print("\n[1/2] Loading Direct Ultralytics YOLO model...")
    direct_model = YOLO(str(model_path))

    # 2. Initialize Service DamageDetector
    print("[2/2] Loading Service DamageDetector (from app.model)...")
    service_detector = DamageDetector(model_path=str(model_path))

    comparison_records = []

    print("\n" + "-" * 85)
    print(" DETAILED PER-IMAGE INFERENCE")
    print("-" * 85)

    for idx, img_path in enumerate(images, start=1):
        rel_name = img_path.name
        print(f"\n[{idx}/{len(images)}] Processing: {rel_name}")

        # A. Direct Raw YOLO Prediction
        direct_results = direct_model.predict(
            source=str(img_path),
            conf=args.conf,
            imgsz=args.imgsz,
            verbose=False
        )

        direct_detections = []
        if direct_results and len(direct_results) > 0:
            r = direct_results[0]
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    conf = float(box.conf[0].item())
                    cls_id = int(box.cls[0].item())
                    cls_name = direct_model.names.get(cls_id, f"D{cls_id}")
                    xyxy = [round(float(c), 2) for c in box.xyxy[0].tolist()]
                    direct_detections.append({
                        "class": cls_name,
                        "conf": round(conf, 4),
                        "bbox": xyxy
                    })

        # B. Service Pipeline Prediction (using Image.open convert RGB as done in /detect)
        with Image.open(img_path) as pil_raw:
            pil_img = pil_raw.convert("RGB")
            service_detections = service_detector.predict(
                image=pil_img,
                conf_lower=args.conf,
                conf_upper=0.50
            )

        # Print direct details
        print(f"  • Direct Raw YOLO ({len(direct_detections)} detections):")
        if direct_detections:
            for d in direct_detections:
                print(f"      - Class: {d['class']:<4} | Conf: {d['conf']:.4f} | BBox: {d['bbox']}")
        else:
            print("      (No detections above threshold)")

        # Print service details
        print(f"  • Service Pipeline ({len(service_detections)} detections):")
        if service_detections:
            for s in service_detections:
                print(f"      - Class: {s.class_name:<4} | Conf: {s.confidence:.4f} | BBox: {s.bbox} | Flag: {s.flag}")
        else:
            print("      (No detections above threshold)")

        # Compare counts
        d_count = len(direct_detections)
        s_count = len(service_detections)
        is_match = (d_count == s_count)

        if is_match:
            print(f"  ✓ Status: MATCH ({d_count} == {s_count})")
        else:
            print(f"  ❌ Status: MISMATCH! (Direct: {d_count} != Service: {s_count})")

        comparison_records.append({
            "name": rel_name,
            "direct_count": d_count,
            "service_count": s_count,
            "match": is_match
        })

    # Summary Table
    print("\n" + "=" * 85)
    print(" INFERENCE COMPARISON SUMMARY TABLE")
    print("=" * 85)
    hdr = f"{'Image Name':<30} | {'Direct Count':<14} | {'Service Count':<14} | {'Result':<15}"
    print(hdr)
    print("-" * len(hdr))

    matches_total = 0
    mismatches_total = 0

    for rec in comparison_records:
        if rec["match"]:
            status_str = "✓ MATCH"
            matches_total += 1
        else:
            status_str = "❌ MISMATCH"
            mismatches_total += 1

        print(f"{rec['name']:<30} | {rec['direct_count']:<14} | {rec['service_count']:<14} | {status_str:<15}")

    print("-" * len(hdr))
    print(f"Total Images Evaluated : {len(comparison_records)}")
    print(f"Total Matches          : {matches_total}")
    print(f"Total Mismatches       : {mismatches_total}")

    if mismatches_total == 0:
        print("\n🎉 PERFECT PARITY: Direct YOLO and Service Pipeline produced identical detection counts on all images!")
    else:
        print(f"\n⚠️ ATTENTION: {mismatches_total} image(s) had mismatching detection counts between direct YOLO and service pipeline.")

    print("=" * 85)


if __name__ == "__main__":
    main()
