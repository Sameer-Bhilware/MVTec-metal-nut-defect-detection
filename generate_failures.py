from ultralytics import YOLO
from pathlib import Path
import cv2

if __name__ == '__main__':
    # Load your V3 Champion model
    model_path = r'C:\games\InSiSo Technologies Assignment\improved_v3_best.pt'
    model = YOLO(model_path)
    
    out_dir = Path(r'C:\games\InSiSo Technologies Assignment\failure_cases')
    out_dir.mkdir(exist_ok=True)
    
    img_dir = Path(r'C:\games\InSiSo Technologies Assignment\images\test')
    lbl_dir = Path(r'C:\games\InSiSo Technologies Assignment\labels\test')
    
    class_names = ["bent", "color", "flip", "scratch"]

    print("Generating failure cases with Ground Truth (Red) and Predictions (Blue)...")
    
    results = model.predict(source=str(img_dir), conf=0.25, save=False)
    
    for r in results:
        img_path = Path(r.path)
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        
        # 1. Draw GROUND TRUTH (Red boxes)
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                for line in f:
                    c, xc, yc, bw, bh = map(float, line.split())
                    x1 = int((xc - bw/2) * w)
                    y1 = int((yc - bh/2) * h)
                    x2 = int((xc + bw/2) * w)
                    y2 = int((yc + bh/2) * h)
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red
                    cv2.putText(img, f"GT: {class_names[int(c)]}", (x1, max(y1-5, 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # 2. Draw PREDICTIONS (Blue boxes)
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue
            cv2.putText(img, f"Pred: {class_names[int(box.cls[0])]}", (x1, max(y1-15, 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
        cv2.imwrite(str(out_dir / img_path.name), img)

    print(f"Done! Open the folder to view them: {out_dir}")