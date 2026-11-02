from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt') 
    
    results = model.train(
        data=r'C:\games\InSiSo Technologies Assignment\data.yaml',
        epochs=100,          
        
        # BALANCED FOR ACCURACY AND EDGE DEPLOYMENT:
        imgsz=800,           # Bigger than 640 for scratches, smaller than 1024 for edge CPU speed
        batch=16,            # Fits perfectly in your 4GB VRAM with FP16
        rect=True,           # Saves VRAM and CPU
        amp=True,            # FP16 GPU math
        
        # CPU SAVERS:
        cache='ram',         
        device=0,            
        workers=8,           
        
        # AUGMENTATIONS (The actual improvement):
        fliplr=0.0,          # Stops confusing the "flip" class
        copy_paste=0.3,      # Pastes scratches onto good images to artificially multiply data
        
        patience=20,         
        project='runs',      
        name='metal_nut_improved' 
    )