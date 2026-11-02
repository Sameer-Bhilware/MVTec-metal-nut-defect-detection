import os
import random
import shutil
from pathlib import Path
import cv2
import numpy as np

# ─────────────────────────── CONFIG ───────────────────────────
# CHANGE THIS IF YOUR DATASET IS SOMEWHERE ELSE
DATA_ROOT    = Path(r"C:\games\metal_nut\metal_nut")       
PROJECT_ROOT = Path(r"C:\games\InSiSo Technologies Assignment")

CLASS_NAMES  = ["bent", "color", "flip", "scratch"]
SPLIT_RATIOS = {"train": 0.60, "val": 0.20, "test": 0.20}
GOOD_SPLIT   = {"train": 0.80, "val": 0.10, "test": 0.10}
MIN_AREA_PX  = 8      
SEED         = 42     
# ──────────────────────────────────────────────────────────────

def mask_to_boxes(mask_path, cls_id):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None: return []
    h, w = mask.shape
    binary = (mask > 127).astype(np.uint8)
    
    # Find connected components (blobs of white pixels)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for i in range(1, n): # skip 0 (background)
        x, y, bw, bh, area = stats[i]
        if area < MIN_AREA_PX: continue # Ignore tiny noise
        boxes.append((
            cls_id,
            (x + bw / 2) / w, (y + bh / 2) / h, 
            bw / w, bh / h
        ))
    return boxes

def write_label(label_path, boxes):
    with open(label_path, "w") as f:
        for c, xc, yc, bw, bh in boxes:
            f.write(f"{c} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

def copy_image(src, dst):
    img = cv2.imread(str(src))
    if img is not None:
        cv2.imwrite(str(dst), img)

def main():
    random.seed(SEED)
    img_dir = PROJECT_ROOT / "images"
    lbl_dir = PROJECT_ROOT / "labels"
    
    for split in ("train", "val", "test"):
        (img_dir / split).mkdir(parents=True, exist_ok=True)
        (lbl_dir / split).mkdir(parents=True, exist_ok=True)

    summary = {}

    # 1. Process Defective Images
    for cls_id, cls in enumerate(CLASS_NAMES):
        imgs = sorted((DATA_ROOT / "test" / cls).glob("*.png"))
        if not imgs:
            print(f"WARNING: No images found for {cls} at {DATA_ROOT}/test/{cls}")
            continue
            
        random.shuffle(imgs)
        n = len(imgs)
        n_tr = int(round(n * SPLIT_RATIOS["train"]))
        n_va = int(round(n * SPLIT_RATIOS["val"]))
        chunks = {"train": imgs[:n_tr], "val": imgs[n_tr:n_tr + n_va], "test": imgs[n_tr + n_va:]}

        for split, files in chunks.items():
            for src in files:
                mask_path = DATA_ROOT / "ground_truth" / cls / f"{src.stem}_mask.png"
                if not mask_path.exists(): continue
                
                boxes = mask_to_boxes(mask_path, cls_id)
                if not boxes: continue

                stem = f"{cls}_{src.stem}"
                copy_image(src, img_dir / split / f"{stem}.png")
                write_label(lbl_dir / split / f"{stem}.txt", boxes)
                summary[(split, cls)] = summary.get((split, cls), 0) + 1

    # 2. Process Good Images (Negatives)
    good = sorted((DATA_ROOT / "train" / "good").glob("*.png"))
    if good:
        random.shuffle(good)
        n = len(good)
        n_tr = int(n * GOOD_SPLIT["train"])
        n_va = int(n * GOOD_SPLIT["val"])
        good_chunks = {"train": good[:n_tr], "val": good[n_tr:n_tr + n_va], "test": good[n_tr + n_va:]}

        for split, files in good_chunks.items():
            for src in files:
                stem = f"good_{src.stem}"
                copy_image(src, img_dir / split / f"{stem}.png")
                write_label(lbl_dir / split / f"{stem}.txt", []) # Empty label
            summary[("negatives", split)] = len(files)
    else:
        print(f"WARNING: No good images found at {DATA_ROOT}/train/good")

    # 3. Write data.yaml
    yaml_path = PROJECT_ROOT / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {PROJECT_ROOT.as_posix()}\n")
        f.write("train: images/train\nval: images/val\ntest: images/test\n")
        f.write("names:\n")
        for i, c in enumerate(CLASS_NAMES):
            f.write(f"  {i}: {c}\n")

    print("\n=== DATASET SUMMARY ===")
    for split in ("train", "val", "test"):
        row = [f"{cls}={summary.get((split, cls), 0)}" for cls in CLASS_NAMES]
        neg = summary.get(("negatives", split), 0)
        print(f"{split:5s}: " + ", ".join(row) + f"  | good(neg)={neg}")

    print(f"\nDone! Dataset built at: {PROJECT_ROOT}")
    print(f"Check the 'data.yaml' file created there.")

if __name__ == "__main__":
    main()