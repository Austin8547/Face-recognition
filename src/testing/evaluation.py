import time
import numpy as np
import cv2
import os
import sys

# Ensure imports work regardless of execution directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.config.config import config
from src.utility.yolo_face import YOLOv8_face
from src.utility.align import norm_crop
from src.utility.arcface import ArcFaceONNX

# ── Config ────────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH    = config['model']['YOLO_ONNX_PATH']
ARCFACE_MODEL_PATH = config['model']['ARCFACE_MODEL_PATH']
IMAGE_SIZE         = config['model']['IMAGE_SIZE']
YOLO_CONF_THRESH   = config['model']['YOLO_CONF_THRESH']

def evaluate_pipeline():
    print("\n==============================================")
    print("      AI PIPELINE PERFORMANCE EVALUATION      ")
    print("==============================================\n")
    
    # ── 1. Load Models ──
    print("[1] Loading models onto GPU...")
    t0 = time.perf_counter()
    detector = YOLOv8_face(YOLO_MODEL_PATH, conf_thres=YOLO_CONF_THRESH, iou_thres=0.5)
    t_yolo_load = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    arcface = ArcFaceONNX(ARCFACE_MODEL_PATH)
    t_arc_load = time.perf_counter() - t0
    
    print(f"    ✓ YOLO Load Time:    {t_yolo_load*1000:.2f} ms")
    print(f"    ✓ ArcFace Load Time: {t_arc_load*1000:.2f} ms")
    
    # ── 2. Warmup ──
    # Create a solid black image (640x480) for consistent baseline testing
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_aligned = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    
    print("\n[2] Warming up CUDA execution providers...")
    for _ in range(5):
        detector.detect(dummy_frame)
        arcface.get_embedding(dummy_aligned)
            
    # ── 3. Benchmarking ──
    ITERATIONS = 100
    yolo_times = []
    arc_times = []
    
    print(f"\n[3] Running {ITERATIONS} iterations...")
    
    for _ in range(ITERATIONS):
        # YOLO Detection Benchmark
        t0 = time.perf_counter()
        detector.detect(dummy_frame)
        t_yolo = time.perf_counter() - t0
        yolo_times.append(t_yolo)
        
        # ArcFace Embedding Benchmark
        t0 = time.perf_counter()
        arcface.get_embedding(dummy_aligned)
        t_arc = time.perf_counter() - t0
        arc_times.append(t_arc)
        
    avg_yolo = np.mean(yolo_times) * 1000
    avg_arc = np.mean(arc_times) * 1000
    
    print("\n==============================================")
    print("             BENCHMARK RESULTS                ")
    print("==============================================")
    print(f" YOLO Detection Speed:      {avg_yolo:.2f} ms")
    print(f" ArcFace Embedding Speed:   {avg_arc:.2f} ms (per face)")
    print("----------------------------------------------")
    print(f" Total AI Time Per Frame:   {avg_yolo + avg_arc:.2f} ms")
    print(f" Theoretical Max FPS:       {1000 / (avg_yolo + avg_arc):.1f} FPS")
    print("==============================================\n")
    print("* Note: Max FPS assumes exactly 1 face in the frame.")
    print("* Additional faces add ~{:.2f} ms each to the total time.".format(avg_arc))

if __name__ == '__main__':
    evaluate_pipeline()
