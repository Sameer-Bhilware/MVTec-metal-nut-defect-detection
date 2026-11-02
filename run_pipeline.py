import subprocess
import sys

if __name__ == '__main__':
    print("==================================================")
    print(" STEP 1: Evaluating Champion Model on Test Set")
    print("==================================================")
    subprocess.run([sys.executable, "evaluate.py"], check=True)
    
    print("\n==================================================")
    print(" STEP 2: Exporting to ONNX & Benchmarking CPU")
    print("==================================================")
    subprocess.run([sys.executable, "export_onnx.py"], check=True)
    
    print("\nPipeline complete! Check the outputs above.")