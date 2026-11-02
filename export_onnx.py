from ultralytics import YOLO
import onnxruntime as ort
import numpy as np
import time
import platform

if __name__ == '__main__':
    # 1. Load the SAVED V3 model from the main folder
    model_path = r'C:\games\InSiSo Technologies Assignment\improved_v3_best.pt'
    model = YOLO(model_path)
    
    print("Exporting to ONNX...")
    onnx_path = model.export(format='onnx', imgsz=640, simplify=True)
    print(f"Model exported successfully to: {onnx_path}")
    
    # 2. Benchmark CPU speed
    print("\nBenchmarking CPU Inference Speed...")
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    
    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        sess.run(None, {input_name: dummy_input})
        
    # Timed runs
    times = []
    for _ in range(50):
        start = time.perf_counter()
        sess.run(None, {input_name: dummy_input})
        times.append((time.perf_counter() - start) * 1000)
        
    print(f"\n--- SPEED RESULTS ---")
    print(f"CPU: {platform.processor()}")
    print(f"Input Size: 640x640")
    print(f"Mean Latency: {np.mean(times):.2f} ms")
    print(f"P95 Latency:  {np.percentile(times, 95):.2f} ms")