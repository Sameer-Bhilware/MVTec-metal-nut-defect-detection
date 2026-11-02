from ultralytics import YOLO

if __name__ == '__main__':
    # STRICT EDGE CONSTRAINT: Using the tiny YOLOv8n model
    model = YOLO('yolov8n.pt') 
    
    results = model.train(
        data=r'C:\games\InSiSo Technologies Assignment\data.yaml',
        epochs=200,          # Give it time to fully learn
        imgsz=640,           # Perfect for edge CPU inference
        batch=16,            # Good balance for 60 images (more weight updates)
        amp=True,            
        cache='ram',         
        device=0,            
        workers=8,           
        
        # --- THE WINNING TWEAKS ---
        optimizer='auto',    
        cos_lr=True,         # Smooth learning rate decay
        patience=50,         # Stop only if it hasn't improved for 50 epochs
        box=10.0,            # INCREASE PENALTY for bad boxes (helps scratches)
        cls=1.5,             # INCREASE PENALTY for wrong classes
        
        # --- STANDARD AUGMENTATION ---
        fliplr=0.5,          
        mosaic=1.0,          
        
        project='runs',      
        name='metal_nut_improved_v3' 
    )