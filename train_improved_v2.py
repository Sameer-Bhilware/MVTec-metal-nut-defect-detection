from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt') 
    
    results = model.train(
        data=r'C:\games\InSiSo Technologies Assignment\data.yaml',
        epochs=150,          # <--- Train longer! (Baseline stopped at 100)
        imgsz=640,           # <--- Revert to 640 (Best for edge CPU speed)
        batch=16,            
        amp=True,            
        cache='ram',         
        device=0,            
        workers=8,           
        
        # MILD AUGMENTATION (Let YOLO do what it does best)
        fliplr=0.5,          # <--- Re-enable flips (Standard YOLO default)
        copy_paste=0.0,      # <--- Turn off copy-paste
        mosaic=1.0,          # Standard mosaic
        
        patience=30,         # Give it 30 epochs of no improvement before stopping
        project='runs',      
        name='metal_nut_improved_v2' 
    )