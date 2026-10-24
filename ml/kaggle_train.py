"""
ml/kaggle_train.py
==================
Kaggle-ready pipeline for Road Damage Detection (RDD2022) using YOLOv8.
Engineered for cell-by-cell execution in Kaggle Notebooks (delimited by '# %% Cell N').

Key Objectives:
1. Inspect input dataset structure and metadata without modifications.
2. Filter India-specific road damage records into /kaggle/working/india/.
3. Standardize annotations to 4 target damage classes: D00, D10, D20, D40.
4. Fine-tune YOLOv8n with configurable hyperparameters.
5. Evaluate model performance and extract genuine validation metrics.
6. Export best model weights and evaluation metrics to /kaggle/working/outputs/.
"""

# %% Cell 1: Dataset Discovery & Structure Inspection
import os
import shutil
import json
from pathlib import Path
import yaml

# Base paths for Kaggle environment
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

print("=" * 70)
print("[CELL 1] INSPECTING INPUT DATASET DIRECTORY")
print("=" * 70)

if not KAGGLE_INPUT.exists():
    # Fallback simulation if running outside Kaggle
    print(f"Warning: {KAGGLE_INPUT} does not exist. Using local placeholder or current directory.")
    input_datasets = [Path(".")]
else:
    input_datasets = [p for p in KAGGLE_INPUT.iterdir() if p.is_dir()]

print(f"Found {len(input_datasets)} dataset(s) in {KAGGLE_INPUT}:")
for ds in input_datasets:
    print(f"  - {ds.name} ({ds})")

# Locate primary dataset root containing data.yaml or splits
dataset_root = None
original_data_yaml_path = None

for root_candidate in input_datasets:
    yaml_candidates = list(root_candidate.rglob("*.yaml")) + list(root_candidate.rglob("*.yml"))
    if yaml_candidates:
        original_data_yaml_path = yaml_candidates[0]
        dataset_root = original_data_yaml_path.parent
        break

if not dataset_root and input_datasets:
    dataset_root = input_datasets[0]

print(f"\nTarget Dataset Root: {dataset_root}")

# Print directory structure (Top 2 levels)
print("\n--- Directory Hierarchy (Top 2 Levels) ---")
if dataset_root and dataset_root.exists():
    for item in sorted(dataset_root.iterdir()):
        if item.is_dir():
            print(f"📁 {item.name}/")
            try:
                for sub in sorted(item.iterdir()):
                    if sub.is_dir():
                        print(f"    └── 📁 {sub.name}/")
                    else:
                        print(f"    └── 📄 {sub.name}")
            except PermissionError:
                print("    └── [Permission Denied]")
        else:
            print(f"📄 {item.name}")

# Read & Display data.yaml if found
original_yaml_data = {}
if original_data_yaml_path and original_data_yaml_path.exists():
    print(f"\n--- Original data.yaml ({original_data_yaml_path}) ---")
    with open(original_data_yaml_path, "r", encoding="utf-8") as f:
        original_yaml_data = yaml.safe_load(f)
    print(yaml.dump(original_yaml_data, default_flow_style=False))
else:
    print("\nNo data.yaml found in dataset root.")

# Count images and labels per split in the raw input
splits = ["train", "val", "test"]
image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

print("\n--- Raw Input Dataset Counts ---")
for split in splits:
    split_imgs = []
    split_lbls = []
    
    # Check both <split>/images and images/<split> patterns
    candidates_img = [
        dataset_root / split / "images",
        dataset_root / "images" / split,
        dataset_root / split
    ]
    candidates_lbl = [
        dataset_root / split / "labels",
        dataset_root / "labels" / split,
        dataset_root / split
    ]
    
    for c in candidates_img:
        if c.exists() and c.is_dir():
            split_imgs = [p for p in c.iterdir() if p.suffix.lower() in image_exts]
            if split_imgs:
                break
                
    for c in candidates_lbl:
        if c.exists() and c.is_dir():
            split_lbls = [p for p in c.iterdir() if p.suffix.lower() == ".txt"]
            if split_lbls:
                break
                
    print(f"Split '{split}': {len(split_imgs)} images, {len(split_lbls)} label files")

print("\n[CELL 1 COMPLETE] Inspection complete. No files were modified.")


# %% Cell 2: Filter India Files & Setup Working Splits
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

print("=" * 70)
print("[CELL 2] FILTERING INDIA DATASET RECORDS (PARALLEL & RESUMABLE)")
print("=" * 70)

WORKING_INDIA = KAGGLE_WORKING / "india"
splits = ["train", "val", "test"]

# Create clean destination folder structure
for split in splits:
    (WORKING_INDIA / split / "images").mkdir(parents=True, exist_ok=True)
    (WORKING_INDIA / split / "labels").mkdir(parents=True, exist_ok=True)

def copy_image_and_label(img_path: Path, src_lbl_dir: Path, dst_img_dir: Path, dst_lbl_dir: Path):
    """
    Copies an image and its corresponding label file.
    Skips copying if the destination file already exists (allows resumable execution).
    Returns: (img_copied, img_skipped, lbl_copied, lbl_skipped, lbl_missing)
    """
    img_copied = False
    img_skipped = False
    lbl_copied = False
    lbl_skipped = False
    lbl_missing = False

    # 1. Process Image
    dst_img_file = dst_img_dir / img_path.name
    if dst_img_file.exists():
        img_skipped = True
    else:
        shutil.copy2(img_path, dst_img_file)
        img_copied = True

    # 2. Process Corresponding Label
    if src_lbl_dir and src_lbl_dir.exists():
        src_lbl_file = src_lbl_dir / f"{img_path.stem}.txt"
        if src_lbl_file.exists():
            dst_lbl_file = dst_lbl_dir / src_lbl_file.name
            if dst_lbl_file.exists():
                lbl_skipped = True
            else:
                shutil.copy2(src_lbl_file, dst_lbl_file)
                lbl_copied = True
        else:
            lbl_missing = True
    else:
        lbl_missing = True

    return img_copied, img_skipped, lbl_copied, lbl_skipped, lbl_missing

total_missing_labels = 0
total_images_copied = 0
total_images_skipped = 0
total_labels_copied = 0
total_labels_skipped = 0

for split in splits:
    # Resolve source images and labels paths
    src_img_dir = None
    src_lbl_dir = None
    
    for c in [dataset_root / split / "images", dataset_root / "images" / split, dataset_root / split]:
        if c.exists() and c.is_dir():
            src_img_dir = c
            break
            
    for c in [dataset_root / split / "labels", dataset_root / "labels" / split, dataset_root / split]:
        if c.exists() and c.is_dir():
            src_lbl_dir = c
            break

    if not src_img_dir or not src_img_dir.exists():
        print(f"Warning: Image directory for split '{split}' not found at {dataset_root}.")
        continue

    # Filter files containing 'India' (case-insensitive)
    india_images = [
        f for f in src_img_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in image_exts and "india" in f.name.lower()
    ]
    
    dst_img_dir = WORKING_INDIA / split / "images"
    dst_lbl_dir = WORKING_INDIA / split / "labels"
    
    split_img_copied = 0
    split_img_skipped = 0
    split_lbl_copied = 0
    split_lbl_skipped = 0
    split_lbl_missing = 0

    print(f"\nProcessing split '{split}' ({len(india_images)} images found)...")

    # Parallel copy execution with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [
            executor.submit(copy_image_and_label, img_p, src_lbl_dir, dst_img_dir, dst_lbl_dir)
            for img_p in india_images
        ]
        
        for future in tqdm(as_completed(futures), total=len(india_images), desc=f"Split [{split}]"):
            ic, iskp, lc, lskp, lm = future.result()
            if ic:
                split_img_copied += 1
            if iskp:
                split_img_skipped += 1
            if lc:
                split_lbl_copied += 1
            if lskp:
                split_lbl_skipped += 1
            if lm:
                split_lbl_missing += 1

    # Accumulate global totals
    total_images_copied += split_img_copied
    total_images_skipped += split_img_skipped
    total_labels_copied += split_lbl_copied
    total_labels_skipped += split_lbl_skipped
    total_missing_labels += split_lbl_missing

    # Print summary per split
    print(f"Split '{split}' completed:")
    print(f"  • Images : {split_img_copied} copied, {split_img_skipped} skipped (already present)")
    print(f"  • Labels : {split_lbl_copied} copied, {split_lbl_skipped} skipped (already present)")
    print(f"  • Missing: {split_lbl_missing} images had no matching label file")

# Overall summary
print(f"\n--- India File Isolation Summary ---")
print(f"Total Images: {total_images_copied} copied, {total_images_skipped} skipped")
print(f"Total Labels: {total_labels_copied} copied, {total_labels_skipped} skipped")
print(f"Total images with missing label files across all splits: {total_missing_labels}")

# Test split label verification (checks destination folder directly for resumability)
test_lbl_files = list((WORKING_INDIA / "test" / "labels").glob("*.txt"))
test_has_labels = len(test_lbl_files) > 0

if not test_has_labels:
    print("\n" + "!" * 70)
    print("NOTICE: Test split has NO label annotations (standard unannotated test set).")
    print("Action: 'val' split will be designated as the final evaluation and validation split.")
    print("!" * 70 + "\n")
else:
    print(f"\nNotice: Test split contains {len(test_lbl_files)} label annotations.\n")

print("[CELL 2 COMPLETE] India subset isolation complete.")


# %% Cell 3: Verify Classes, Remap Annotations & Generate data.yaml
print("=" * 70)
print("[CELL 3] CLASS VERIFICATION & DATA.YAML GENERATION")
print("=" * 70)

# Target 4 classes per RoadGuard AI specifications
TARGET_CLASSES = ["D00", "D10", "D20", "D40"]
CLASS_DESCRIPTIONS = {
    "D00": "Longitudinal Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole"
}
target_class_to_id = {cls_name: idx for idx, cls_name in enumerate(TARGET_CLASSES)}

print(f"Target Classes ({len(TARGET_CLASSES)}):")
for idx, cls_name in enumerate(TARGET_CLASSES):
    print(f"  {idx}: {cls_name} ({CLASS_DESCRIPTIONS[cls_name]})")

# Determine original class mapping from original data.yaml
orig_names = original_yaml_data.get("names", {})
orig_id_to_name = {}
if isinstance(orig_names, dict):
    orig_id_to_name = {int(k): str(v) for k, v in orig_names.items()}
elif isinstance(orig_names, list):
    orig_id_to_name = {i: name for i, name in enumerate(orig_names)}

print(f"\nOriginal Dataset Class Mapping: {orig_id_to_name}")

# Build translation dictionary: original_class_id -> target_class_id (0..3)
id_remap = {}
for orig_id, orig_name in orig_id_to_name.items():
    cleaned_name = orig_name.strip()
    if cleaned_name in target_class_to_id:
        id_remap[orig_id] = target_class_to_id[cleaned_name]

# If mapping could not be inferred from original yaml, assume direct 0..3 correspondence if counts match
if not id_remap:
    print("Notice: Mapping not found in data.yaml. Assuming default direct index mapping (0: D00, 1: D10, 2: D20, 3: D40).")
    id_remap = {i: i for i in range(len(TARGET_CLASSES))}
else:
    print(f"Inferred Class Remapping (Original ID -> Target ID): {id_remap}")

# Standardize annotations and count boxes per split and class
split_box_counts = {split: {cls_name: 0 for cls_name in TARGET_CLASSES} for split in splits}

for split in splits:
    lbl_dir = WORKING_INDIA / split / "labels"
    if not lbl_dir.exists():
        continue
        
    for lbl_file in lbl_dir.glob("*.txt"):
        with open(lbl_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        valid_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
                
            orig_cls = int(float(parts[0]))
            if orig_cls in id_remap:
                target_cls = id_remap[orig_cls]
                target_name = TARGET_CLASSES[target_cls]
                split_box_counts[split][target_name] += 1
                
                # Write back standardized target class ID
                valid_lines.append(f"{target_cls} " + " ".join(parts[1:]) + "\n")
                
        # Rewrite label file with validated 4-class annotations
        with open(lbl_file, "w", encoding="utf-8") as f:
            f.writelines(valid_lines)

# Print Bounding Box Distribution Table
print("\n--- Bounding Box Counts per Class & Split ---")
header = f"{'Split':<10} | " + " | ".join([f"{cls:<6}" for cls in TARGET_CLASSES]) + " | Total Boxes"
print("-" * len(header))
print(header)
print("-" * len(header))

for split in splits:
    counts = [split_box_counts[split][cls] for cls in TARGET_CLASSES]
    total_split = sum(counts)
    counts_str = " | ".join([f"{c:<6}" for c in counts])
    print(f"{split:<10} | {counts_str} | {total_split}")
print("-" * len(header))

# Write standardized data.yaml for YOLOv8
new_data_yaml_path = WORKING_INDIA / "data.yaml"
eval_split_path = "val/images" if not test_has_labels else "test/images"

data_yaml_content = {
    "path": str(WORKING_INDIA.resolve()),
    "train": "train/images",
    "val": "val/images",
    "test": eval_split_path,
    "names": {i: name for i, name in enumerate(TARGET_CLASSES)},
    "nc": len(TARGET_CLASSES)
}

with open(new_data_yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(data_yaml_content, f, default_flow_style=False)

print(f"\nGenerated New Standardized data.yaml at: {new_data_yaml_path}")
with open(new_data_yaml_path, "r", encoding="utf-8") as f:
    print(f.read())

print("[CELL 3 COMPLETE] Classes standardized and data.yaml prepared.")


# %% Cell 3b: Re-split India data
import random
import shutil
import json
from collections import defaultdict
from pathlib import Path
import yaml

print("=" * 70)
print("[CELL 3b] RE-SPLITTING INDIA DATA (80/10/10 STRATIFIED + 10% BACKGROUND)")
print("=" * 70)
print("Reason: Original test split is unannotated (0 boxes) and val has severe class imbalance (only 8 D10 boxes).")

SPLIT_SEED = 0
random.seed(SPLIT_SEED)

WORKING_INDIA = KAGGLE_WORKING / "india"
OUTPUTS_DIR = KAGGLE_WORKING / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Pool all labeled images and identify background images from train and val
pooled_labeled = {}      # img_name: {"img_path": Path, "lbl_path": Path, "boxes": [(cls_id, xc, yc, w, h)]}
pooled_background = {}   # img_name: {"img_path": Path, "lbl_path": Path or None}

for src_split in ["train", "val"]:
    img_dir = WORKING_INDIA / src_split / "images"
    lbl_dir = WORKING_INDIA / src_split / "labels"
    
    if not img_dir.exists():
        continue
        
    for img_p in img_dir.iterdir():
        if img_p.suffix.lower() not in image_exts:
            continue
            
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        boxes = []
        
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(float(parts[0]))
                        coords = [float(x) for x in parts[1:5]]
                        boxes.append((cls_id, *coords))
                        
        if boxes:
            pooled_labeled[img_p.name] = {
                "img_path": img_p,
                "lbl_path": lbl_p,
                "boxes": boxes
            }
        else:
            # Empty label file or unannotated -> Background image
            pooled_background[img_p.name] = {
                "img_path": img_p,
                "lbl_path": lbl_p if lbl_p.exists() else None
            }

num_labeled = len(pooled_labeled)
num_available_bg = len(pooled_background)
print(f"Total labeled images pooled from train + val:    {num_labeled}")
print(f"Total background images found in train + val:    {num_available_bg}")

# 2. Sample 10% background images relative to labeled count
target_bg_count = int(round(0.10 * num_labeled))
actual_bg_count = min(num_available_bg, target_bg_count)

bg_rng = random.Random(SPLIT_SEED + 999)
all_bg_names = sorted(list(pooled_background.keys()))
bg_rng.shuffle(all_bg_names)
sampled_bg_names = all_bg_names[:actual_bg_count]

print(f"Target background images (10% of labeled):       {target_bg_count}")
print(f"Sampled background images:                       {actual_bg_count}")

# Distribute background images proportionally (80/10/10)
n_bg_train = int(round(actual_bg_count * 0.80))
n_bg_val = int(round(actual_bg_count * 0.10))
n_bg_test = actual_bg_count - n_bg_train - n_bg_val

bg_allocations = {
    "train": sampled_bg_names[:n_bg_train],
    "val": sampled_bg_names[n_bg_train:n_bg_train + n_bg_val],
    "test": sampled_bg_names[n_bg_train + n_bg_val:]
}

print(f"Background allocations: Train={len(bg_allocations['train'])}, Val={len(bg_allocations['val'])}, Test={len(bg_allocations['test'])}")

# 3. Sequence / Location Analysis from filename
has_trip_metadata = False
sample_names = list(pooled_labeled.keys())[:15]
sample_prefixes = set()
for name in sample_names:
    parts = name.split("_")
    if len(parts) > 2:
        sample_prefixes.add(parts[1])

if len(sample_prefixes) > 1:
    print(f"Detected potential route/trip grouping prefixes: {sample_prefixes}")
    has_trip_metadata = True
else:
    print("Notice: File names follow flat naming pattern ('India_xxxxxx.jpg') without explicit route/sequence/location metadata.")
    print("Sequence-grouped splitting is not possible from filenames; proceeding with image-level stratified split.")

# 4. Stratification by Rarest Class for Labeled Images
class_box_totals = defaultdict(int)
for info in pooled_labeled.values():
    for b in info["boxes"]:
        class_box_totals[b[0]] += 1

print("\nPooled Bounding Box Distribution (Labeled):")
for cls_id in sorted(class_box_totals.keys()):
    cls_name = TARGET_CLASSES[cls_id] if cls_id < len(TARGET_CLASSES) else f"Class_{cls_id}"
    print(f"  {cls_name}: {class_box_totals[cls_id]} boxes")

rarity_rank = sorted(class_box_totals.keys(), key=lambda c: class_box_totals[c])
print(f"Rarity order (rarest first): {[TARGET_CLASSES[c] for c in rarity_rank]}")

strata_groups = defaultdict(list)
for img_name, info in pooled_labeled.items():
    present_classes = {b[0] for b in info["boxes"]}
    assigned_stratum = None
    for r_cls in rarity_rank:
        if r_cls in present_classes:
            assigned_stratum = r_cls
            break
    strata_groups[assigned_stratum].append(img_name)

# Partition labeled images (80% train, 10% val, 10% test)
split_allocations = {"train": [], "val": [], "test": []}

for stratum_cls, img_list in strata_groups.items():
    rng = random.Random(SPLIT_SEED + (stratum_cls if isinstance(stratum_cls, int) else 0))
    shuffled_imgs = list(img_list)
    rng.shuffle(shuffled_imgs)
    
    n = len(shuffled_imgs)
    n_val = max(1, int(round(n * 0.10))) if n >= 10 else (1 if n >= 3 else 0)
    n_test = max(1, int(round(n * 0.10))) if n >= 10 else (1 if n >= 2 else 0)
    n_train = n - n_val - n_test
    
    if n_train < 1 and n > 0:
        n_train = 1
        if n_val > n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1

    split_allocations["train"].extend(shuffled_imgs[:n_train])
    split_allocations["val"].extend(shuffled_imgs[n_train:n_train + n_val])
    split_allocations["test"].extend(shuffled_imgs[n_train + n_val:])

print(f"\nLabeled Re-split Allocations (Seed={SPLIT_SEED}):")
print(f"  Train: {len(split_allocations['train'])} images ({len(split_allocations['train'])/num_labeled*100:.1f}%)")
print(f"  Val:   {len(split_allocations['val'])} images ({len(split_allocations['val'])/num_labeled*100:.1f}%)")
print(f"  Test:  {len(split_allocations['test'])} images ({len(split_allocations['test'])/num_labeled*100:.1f}%)")

# 5. Save split lists to JSON with separate key for background images
split_json_path = OUTPUTS_DIR / f"split_seed{SPLIT_SEED}.json"
split_export_payload = {
    "metadata": {
        "seed": SPLIT_SEED,
        "train_ratio": 0.80,
        "val_ratio": 0.10,
        "test_ratio": 0.10,
        "total_labeled_images": num_labeled,
        "total_background_images": actual_bg_count,
        "total_images": num_labeled + actual_bg_count
    },
    "splits": {
        "train": sorted(split_allocations["train"]),
        "val": sorted(split_allocations["val"]),
        "test": sorted(split_allocations["test"])
    },
    "background_images": {
        "train": sorted(bg_allocations["train"]),
        "val": sorted(bg_allocations["val"]),
        "test": sorted(bg_allocations["test"])
    }
}
with open(split_json_path, "w", encoding="utf-8") as f:
    json.dump(split_export_payload, f, indent=2)
print(f"\nSplit lists saved to: {split_json_path}")

# 6. Reorganize directory structure
temp_staging_dir = WORKING_INDIA / "_temp_pool"
temp_staging_dir.mkdir(parents=True, exist_ok=True)
(temp_staging_dir / "images").mkdir(parents=True, exist_ok=True)
(temp_staging_dir / "labels").mkdir(parents=True, exist_ok=True)

# Move labeled files to staging
for img_name, info in pooled_labeled.items():
    shutil.move(info["img_path"], temp_staging_dir / "images" / img_name)
    shutil.move(info["lbl_path"], temp_staging_dir / "labels" / f"{Path(img_name).stem}.txt")

# Move sampled background files to staging
for bg_name in sampled_bg_names:
    bg_info = pooled_background[bg_name]
    shutil.move(bg_info["img_path"], temp_staging_dir / "images" / bg_name)
    bg_stem = Path(bg_name).stem
    if bg_info["lbl_path"] and bg_info["lbl_path"].exists():
        shutil.move(bg_info["lbl_path"], temp_staging_dir / "labels" / f"{bg_stem}.txt")
    else:
        # Create empty label file for background sample in staging
        (temp_staging_dir / "labels" / f"{bg_stem}.txt").touch()

# Clean existing split directories
for s in ["train", "val", "test"]:
    s_dir = WORKING_INDIA / s
    if s_dir.exists():
        shutil.rmtree(s_dir)
    (s_dir / "images").mkdir(parents=True, exist_ok=True)
    (s_dir / "labels").mkdir(parents=True, exist_ok=True)

# Distribute labeled files to new splits
for split_name, img_names in split_allocations.items():
    dst_img = WORKING_INDIA / split_name / "images"
    dst_lbl = WORKING_INDIA / split_name / "labels"
    for img_name in img_names:
        shutil.move(temp_staging_dir / "images" / img_name, dst_img / img_name)
        stem = Path(img_name).stem
        lbl_file = temp_staging_dir / "labels" / f"{stem}.txt"
        if lbl_file.exists():
            shutil.move(lbl_file, dst_lbl / f"{stem}.txt")

# Distribute background files to new splits
for split_name, bg_names in bg_allocations.items():
    dst_img = WORKING_INDIA / split_name / "images"
    dst_lbl = WORKING_INDIA / split_name / "labels"
    for bg_name in bg_names:
        shutil.move(temp_staging_dir / "images" / bg_name, dst_img / bg_name)
        stem = Path(bg_name).stem
        lbl_file = temp_staging_dir / "labels" / f"{stem}.txt"
        if lbl_file.exists():
            shutil.move(lbl_file, dst_lbl / f"{stem}.txt")
        else:
            (dst_lbl / f"{stem}.txt").touch()

# Remove staging directory
shutil.rmtree(temp_staging_dir, ignore_errors=True)

# Test split now contains verified ground-truth labels
test_has_labels = True

# 7. Rewrite data.yaml with updated verified splits
new_data_yaml_path = WORKING_INDIA / "data.yaml"
data_yaml_content = {
    "path": str(WORKING_INDIA.resolve()),
    "train": "train/images",
    "val": "val/images",
    "test": "test/images",
    "names": {i: name for i, name in enumerate(TARGET_CLASSES)},
    "nc": len(TARGET_CLASSES)
}
with open(new_data_yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(data_yaml_content, f, default_flow_style=False)

print(f"\nUpdated data.yaml written to: {new_data_yaml_path}")

# 8. Print comprehensive image, background, and bounding box counts table
resplit_box_counts = {s: {cls_name: 0 for cls_name in TARGET_CLASSES} for s in ["train", "val", "test"]}
resplit_total_img_counts = {s: 0 for s in ["train", "val", "test"]}
resplit_bg_img_counts = {s: len(bg_allocations[s]) for s in ["train", "val", "test"]}

for s in ["train", "val", "test"]:
    lbl_dir = WORKING_INDIA / s / "labels"
    img_dir = WORKING_INDIA / s / "images"
    resplit_total_img_counts[s] = len(list(img_dir.glob("*.*")))
    
    for lbl_f in lbl_dir.glob("*.txt"):
        with open(lbl_f, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    c_id = int(float(parts[0]))
                    if c_id < len(TARGET_CLASSES):
                        resplit_box_counts[s][TARGET_CLASSES[c_id]] += 1

print("\n--- Re-Split Distribution Table (Images, Background & Class Boxes) ---")
hdr = f"{'Split':<8} | {'Images':<8} | {'Bg Imgs':<8} | " + " | ".join([f"{cls:<6}" for cls in TARGET_CLASSES]) + " | Total Boxes"
print("-" * len(hdr))
print(hdr)
print("-" * len(hdr))

for s in ["train", "val", "test"]:
    counts = [resplit_box_counts[s][cls] for cls in TARGET_CLASSES]
    total_boxes = sum(counts)
    counts_str = " | ".join([f"{c:<6}" for c in counts])
    print(f"{s:<8} | {resplit_total_img_counts[s]:<8} | {resplit_bg_img_counts[s]:<8} | {counts_str} | {total_boxes}")
print("-" * len(hdr))

print("[CELL 3b COMPLETE] Re-split with background images complete and verified.")


# %% Cell 3c: Visual check
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

print("=" * 70)
print("[CELL 3c] VISUAL VERIFICATION OF CLASS MAPPING")
print("=" * 70)

# Colors for bounding boxes: D00: Cyan, D10: Orange, D20: LimeGreen, D40: Deep Pink
CLASS_COLORS = {
    "D00": "#00FFFF",  # Cyan (Longitudinal Crack)
    "D10": "#FF8C00",  # Dark Orange (Transverse Crack)
    "D20": "#32CD32",  # Lime Green (Alligator Crack)
    "D40": "#FF1493"   # Deep Pink (Pothole)
}

train_img_dir = WORKING_INDIA / "train" / "images"
train_lbl_dir = WORKING_INDIA / "train" / "labels"

# Find images containing each class
class_to_images = defaultdict(list)
for lbl_f in train_lbl_dir.glob("*.txt"):
    stem = lbl_f.stem
    img_candidates = list(train_img_dir.glob(f"{stem}.*"))
    if not img_candidates:
        continue
    img_path = img_candidates[0]
    
    with open(lbl_f, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                c_id = int(float(parts[0]))
                if c_id < len(TARGET_CLASSES):
                    class_to_images[TARGET_CLASSES[c_id]].append(img_path)

# Sample 2 random images per class
random.seed(SPLIT_SEED)
selected_samples = {}
for cls_name in TARGET_CLASSES:
    available = list(set(class_to_images[cls_name]))
    if len(available) >= 2:
        selected_samples[cls_name] = random.sample(available, 2)
    elif len(available) == 1:
        selected_samples[cls_name] = [available[0], available[0]]
    else:
        selected_samples[cls_name] = []
        print(f"Warning: No training images found for {cls_name}!")

# Create 4 rows x 2 columns plot
fig, axes = plt.subplots(nrows=len(TARGET_CLASSES), ncols=2, figsize=(14, 18))
plt.subplots_adjust(hspace=0.3, wspace=0.15)

for row_idx, cls_name in enumerate(TARGET_CLASSES):
    imgs = selected_samples.get(cls_name, [])
    for col_idx in range(2):
        ax = axes[row_idx, col_idx]
        if col_idx < len(imgs):
            img_p = imgs[col_idx]
            img = Image.open(img_p)
            w_img, h_img = img.size
            
            ax.imshow(img)
            ax.set_title(f"[{cls_name}] {CLASS_DESCRIPTIONS[cls_name]}\n{img_p.name}", fontsize=10, fontweight="bold")
            
            # Draw all bounding boxes on this image
            lbl_p = train_lbl_dir / f"{img_p.stem}.txt"
            if lbl_p.exists():
                with open(lbl_p, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            b_cls = int(float(parts[0]))
                            xc, yc, bw, bh = [float(x) for x in parts[1:5]]
                            
                            # Convert YOLO normalized coords to pixel coords
                            xmin = (xc - bw / 2.0) * w_img
                            ymin = (yc - bh / 2.0) * h_img
                            width_px = bw * w_img
                            height_px = bh * h_img
                            
                            b_name = TARGET_CLASSES[b_cls] if b_cls < len(TARGET_CLASSES) else str(b_cls)
                            box_color = CLASS_COLORS.get(b_name, "yellow")
                            
                            rect = patches.Rectangle(
                                (xmin, ymin), width_px, height_px,
                                linewidth=2.5,
                                edgecolor=box_color,
                                facecolor="none"
                            )
                            ax.add_patch(rect)
                            
                            # Add text label badge
                            ax.text(
                                xmin, max(0, ymin - 5),
                                f"{b_name}",
                                color="black",
                                fontsize=9,
                                fontweight="bold",
                                bbox=dict(boxstyle="square,pad=0.2", facecolor=box_color, alpha=0.9, edgecolor="none")
                            )
        else:
            ax.text(0.5, 0.5, "No Sample Available", ha="center", va="center")
        ax.axis("off")

plt.suptitle("RoadGuard AI - RDD2022 Class Ground Truth Visual Verification", fontsize=15, fontweight="bold", y=0.99)
plt.tight_layout()

# Save plot to outputs
visual_check_path = OUTPUTS_DIR / "dataset_visual_check.png"
plt.savefig(visual_check_path, dpi=150, bbox_inches="tight")
print(f"Visual check figure saved to: {visual_check_path}")
plt.show()

print("[CELL 3c COMPLETE] Visual verification completed.")


# %% Cell 4: Environment Setup & YOLOv8n Fine-Tuning
import subprocess
import sys
import torch

print("=" * 70)
print("[CELL 4] TRAINING YOLOV8N ON INDIA ROAD DAMAGE DATASET")
print("=" * 70)

# ==========================================
# Configurable Hyperparameters & Variables
# ==========================================
EPOCHS = 50          # Production training epochs
IMGSZ = 640          # Standard image resolution for road distress detection
BATCH = 16           # Batch size (adjust based on GPU VRAM availability)
SEED = 0             # Deterministic seed for reproducible results
PATIENCE = 15        # Early stopping patience (stop if no val improvement in 15 epochs)
MODEL_NAME = "yolov8n.pt"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
RUN_NAME = f"yolov8n_seed{SEED}"

# Check and install Ultralytics if not already installed
try:
    import ultralytics
    from ultralytics import YOLO
    print(f"Ultralytics version {ultralytics.__version__} is already installed.")
except ImportError:
    print("Installing ultralytics package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
    import ultralytics
    from ultralytics import YOLO
    print(f"Ultralytics installed successfully: version {ultralytics.__version__}")

print(f"\nHyperparameters Configuration:")
print(f"  - Model Base:      {MODEL_NAME}")
print(f"  - Epochs:          {EPOCHS}")
print(f"  - Image Size:      {IMGSZ}")
print(f"  - Batch Size:      {BATCH}")
print(f"  - Seed:            {SEED}")
print(f"  - Patience:        {PATIENCE} (early stopping)")
print(f"  - Run Name:        {RUN_NAME}")
print(f"  - Device:          {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print(f"  - Data Config:     {new_data_yaml_path}")

# Initialize model
model = YOLO(MODEL_NAME)

# Target directory for this specific run
train_project = KAGGLE_WORKING / "runs" / "detect"
run_dir = train_project / RUN_NAME

# Execute training run
train_results = model.train(
    data=str(new_data_yaml_path),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    seed=SEED,
    patience=PATIENCE,
    device=DEVICE,
    project=str(train_project),
    name=RUN_NAME,
    exist_ok=True,
    verbose=True
)

print(f"\n[CELL 4 COMPLETE] Training finished. Output saved to: {run_dir}")


# %% Cell 5: Model Evaluation on TEST Split & Genuine Metric Extraction
print("=" * 70)
print("[CELL 5] RIGOROUS EVALUATION ON TEST SPLIT USING BEST WEIGHTS")
print("=" * 70)

best_weights_path = run_dir / "weights" / "best.pt"
if not best_weights_path.exists():
    # Fallback to last.pt if best.pt is not present
    best_weights_path = run_dir / "weights" / "last.pt"

print(f"Loading evaluated weights from: {best_weights_path}")
best_model = YOLO(str(best_weights_path))

# Run evaluation strictly on the TEST split (NOT validation split)
eval_split = "test"
print(f"Running evaluation strictly on split: '{eval_split}' (Hold-out Test Set)...")

test_metrics = best_model.val(
    data=str(new_data_yaml_path),
    split=eval_split,
    imgsz=IMGSZ,
    batch=BATCH,
    device=DEVICE,
    verbose=False
)

# Extract genuine, non-fabricated metrics directly from Ultralytics result objects
overall_precision = float(test_metrics.box.mp)
overall_recall = float(test_metrics.box.mr)
overall_map50 = float(test_metrics.box.map50)
overall_map50_95 = float(test_metrics.box.map)
fitness_score = float(test_metrics.fitness)

# Extract per-class genuine metrics
per_class_metrics = {}
ap50_per_class = test_metrics.box.ap50
ap_per_class = test_metrics.box.ap
p_per_class = test_metrics.box.p
r_per_class = test_metrics.box.r

for idx, class_name in enumerate(TARGET_CLASSES):
    p_val = float(p_per_class[idx]) if idx < len(p_per_class) else 0.0
    r_val = float(r_per_class[idx]) if idx < len(r_per_class) else 0.0
    ap50_val = float(ap50_per_class[idx]) if idx < len(ap50_per_class) else 0.0
    ap_val = float(ap_per_class[idx]) if idx < len(ap_per_class) else 0.0
    
    per_class_metrics[class_name] = {
        "class_id": idx,
        "description": CLASS_DESCRIPTIONS[class_name],
        "precision": round(p_val, 4),
        "recall": round(r_val, 4),
        "mAP50": round(ap50_val, 4),
        "mAP50_95": round(ap_val, 4)
    }

# Split file name reference
split_file_name = f"split_seed{SEED}.json"

# Format full evaluation report
evaluation_payload = {
    "metadata": {
        "model_architecture": "YOLOv8n",
        "dataset": "RDD2022-India",
        "evaluation_split": eval_split,
        "epochs_trained": EPOCHS,
        "patience": PATIENCE,
        "imgsz": IMGSZ,
        "batch_size": BATCH,
        "seed": SEED,
        "split_file": split_file_name,
        "weights_path": str(best_weights_path),
        "environment": {
            "ultralytics_version": ultralytics.__version__,
            "torch_version": torch.__version__,
            "device": str(DEVICE)
        }
    },
    "overall_metrics": {
        "precision": round(overall_precision, 4),
        "recall": round(overall_recall, 4),
        "mAP50": round(overall_map50, 4),
        "mAP50_95": round(overall_map50_95, 4),
        "fitness": round(fitness_score, 4)
    },
    "per_class_metrics": per_class_metrics
}

# Display results in console table
print("\n--- Overall Model Performance (Test Split) ---")
print(f"Precision:   {overall_precision:.4f}")
print(f"Recall:      {overall_recall:.4f}")
print(f"mAP@50:      {overall_map50:.4f}")
print(f"mAP@50-95:   {overall_map50_95:.4f}")
print(f"Fitness:     {fitness_score:.4f}")

print("\n--- Per-Class Metrics (Test Split) ---")
class_table_hdr = f"{'Class':<6} | {'Description':<20} | {'Precision':<10} | {'Recall':<10} | {'mAP50':<10} | {'mAP50-95':<10}"
print("-" * len(class_table_hdr))
print(class_table_hdr)
print("-" * len(class_table_hdr))
for cls_name, m in per_class_metrics.items():
    print(f"{cls_name:<6} | {m['description']:<20} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['mAP50']:<10.4f} | {m['mAP50_95']:<10.4f}")
print("-" * len(class_table_hdr))

# Save genuine metrics to results.json
results_json_path = KAGGLE_WORKING / "results.json"
with open(results_json_path, "w", encoding="utf-8") as f:
    json.dump(evaluation_payload, f, indent=2)

print(f"\nEvaluation metrics successfully written to: {results_json_path}")
print("[CELL 5 COMPLETE] Metrics evaluated on test split without fabrication.")


# %% Cell 6: Export Artifacts & Archive to /kaggle/working/outputs/
print("=" * 70)
print("[CELL 6] EXPORTING MODEL ARTIFACTS & CREATING ZIP ARCHIVE")
print("=" * 70)

OUTPUTS_DIR = KAGGLE_WORKING / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

exported_files = []

def copy_to_outputs(src_path: Path, dst_name: str = None):
    """Safely copies a file to OUTPUTS_DIR and records its metadata."""
    if not src_path or not src_path.exists():
        return
    dest = OUTPUTS_DIR / (dst_name if dst_name else src_path.name)
    shutil.copy2(src_path, dest)
    size = dest.stat().st_size
    size_str = f"{size / (1024 * 1024):.2f} MB" if size >= 1024 * 1024 else f"{size / 1024:.2f} KB"
    exported_files.append((dest.name, size_str, str(dest)))

# 1. Export best.pt weights
if best_weights_path.exists():
    copy_to_outputs(best_weights_path, "best.pt")

# 2. Export results.json
if results_json_path.exists():
    copy_to_outputs(results_json_path, "results.json")

# 3. Export split file (split_seed{SEED}.json)
split_src = OUTPUTS_DIR / f"split_seed{SEED}.json"
if not split_src.exists():
    candidate_split = KAGGLE_WORKING / f"split_seed{SEED}.json"
    if candidate_split.exists():
        copy_to_outputs(candidate_split)
else:
    size = split_src.stat().st_size
    size_str = f"{size / 1024:.2f} KB"
    exported_files.append((split_src.name, size_str, str(split_src)))

# 4. Export results.csv from the training run
results_csv_path = run_dir / "results.csv"
if results_csv_path.exists():
    copy_to_outputs(results_csv_path, "results.csv")

# 5. Export training plots and performance curves from YOLO run directory
plot_patterns = [
    "results.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "*PR_curve*.png",
    "*F1_curve*.png",
    "*P_curve*.png",
    "*R_curve*.png"
]

for pattern in plot_patterns:
    for plot_file in run_dir.glob(pattern):
        copy_to_outputs(plot_file)

# Also preserve data.yaml in outputs
if new_data_yaml_path.exists():
    copy_to_outputs(new_data_yaml_path, "data.yaml")

# 6. Create Zip Archive of /kaggle/working/outputs/
zip_base_name = KAGGLE_WORKING / f"roadguard_outputs_seed{SEED}"
zip_archive_path = shutil.make_archive(
    base_name=str(zip_base_name),
    format="zip",
    root_dir=str(OUTPUTS_DIR)
)

# Also create standard outputs.zip for convenient download
standard_zip = KAGGLE_WORKING / "outputs.zip"
shutil.copy2(zip_archive_path, standard_zip)

zip_size_mb = Path(zip_archive_path).stat().st_size / (1024 * 1024)

print("\n--- Exported Artifacts in /kaggle/working/outputs/ ---")
for fname, size, fpath in exported_files:
    print(f"  ✓ {fname:<32} ({size}) -> {fpath}")

print("\n--- Generated Zip Archives ---")
print(f"  📦 {Path(zip_archive_path).name:<32} ({zip_size_mb:.2f} MB) -> {zip_archive_path}")
print(f"  📦 {standard_zip.name:<32} ({zip_size_mb:.2f} MB) -> {standard_zip}")

print("\nPipeline execution & packaging completed successfully.")
print("[CELL 6 COMPLETE] Ready for deployment and model serving.")
