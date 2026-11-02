import cv2
import random
from pathlib import Path

PROJECT_ROOT = Path(r"C:\games\InSiSo Technologies Assignment")
CLASS_NAMES = ["bent", "color", "flip", "scratch"]

# Get 5 random training images
img_dir = PROJECT_ROOT / "images" / "train"
images = list(img_dir.glob("*.png"))
random.shuffle(images)

out_dir = PROJECT_ROOT / "sanity_check"
out_dir.mkdir(exist_ok=True)

for img_path in images[:5]:
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    lbl_path = PROJECT_ROOT / "labels" / "train" / f"{img_path.stem}.txt"
    
    with open(lbl_path, "r") as f:
        for line in f:
            c, xc, yc, bw, bh = map(float, line.split())
            # Convert YOLO format back to pixels
            x1 = int((xc - bw/2) * w)
            y1 = int((yc - bh/2) * h)
            x2 = int((xc + bw/2) * w)
            y2 = int((yc + bh/2) * h)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, CLASS_NAMES[int(c)], (x1, max(y1-5, 12)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
    cv2.imwrite(str(out_dir / img_path.name), img)
print(f"Saved 5 images to {out_dir} - go look at them!")