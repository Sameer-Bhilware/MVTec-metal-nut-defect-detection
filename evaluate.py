from ultralytics import YOLO

if __name__ == '__main__':
    # Pointing directly to the renamed file in the main folder
    model = YOLO(r'C:\games\InSiSo Technologies Assignment\runs\detect\runs\metal_nut_improved_v3\weights\best.pt')
    
    # Evaluate on the unseen TEST split
    print("Evaluating Baseline on Test Set...")
    metrics = model.val(data=r'C:\games\InSiSo Technologies Assignment\data.yaml', split='test')
    
    print(f"\n=== BASELINE TEST RESULTS ===")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall:    {metrics.box.mr:.3f}")
    print(f"mAP@0.5:   {metrics.box.map50:.3f}")