from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Load the smallest YOLOv8 model (pretrained on COCO)
    model = YOLO('yolov8n.pt') 
    
    # 2. Train the model
    results = model.train(
        data=r'C:\games\InSiSo Technologies Assignment\data.yaml',
        epochs=100,          # Maximum number of epochs
        imgsz=640,           # Standard image size
        batch=8,             # Fits comfortably in your 4GB RTX 3050
        patience=20,         # Early stopping (stops if no improvement for 20 epochs)
        device=0,            # Forces the use of your GPU
        workers=2,           # Windows safe limit for dataloading
        project='runs',      # Folder to save results
        name='metal_nut_v1'  # Name of this training run
    )