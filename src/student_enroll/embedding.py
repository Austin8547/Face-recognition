import cv2
import numpy as np
import os
import sys
import pickle
import onnxruntime as ort

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utility.yolo_face import YOLOv8_face
from src.utility.align import norm_crop
from src.config.config import config
from src.utility.arcface import ArcFaceONNX


available = ort.get_available_providers()
print(f"Available ORT providers: {available}")
if 'DmlExecutionProvider' not in available:
    print("WARNING: DirectML not available — check onnxruntime-directml is installed")


YOLO_MODEL_PATH    = config["model"]["YOLO_ONNX_PATH"]
ARCFACE_MODEL_PATH = config["model"]["ARCFACE_MODEL_PATH"]
ENROLLMENT_DIR     = config["directory"]["ENROLLMENT_DIR"]
EMBEDDINGS_DIR     = config["directory"]["EMBEDDINGS_DIR"]
IMAGE_SIZE         = config["model"]["IMAGE_SIZE"]



# ── Landmark reshape helper ───────────────────────────────────────────────────
def kpts_to_5x2(kp_flat):
    """
    yolo_face returns landmarks as flat (15,) array: [x0,y0,conf0, x1,y1,conf1, ...]
    align.estimate_norm() expects shape (5, 2) — x,y only.
    """
    return kp_flat.reshape(5, 3)[:, :2].astype(np.float32)


# ── Core per-image function ───────────────────────────────────────────────────
def embed_image(img_bgr, detector, arcface):
    """
    Detect face -> align -> embed.
    Returns 512-d normalised embedding, or None if no face detected.
    Picks the highest-confidence detection when multiple faces appear.
    """
    boxes, scores, _, kpts = detector.detect(img_bgr)

    if len(boxes) == 0:
        return None

    best     = int(np.argmax(scores))
    landmark = kpts_to_5x2(kpts[best])
    aligned  = norm_crop(img_bgr, landmark, image_size=IMAGE_SIZE)
    return arcface.get_embedding(aligned)


def main():
    if not os.path.isdir(ENROLLMENT_DIR):
        print(f"Error: enrollment directory not found:\n  {ENROLLMENT_DIR}")
        return

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    print(f"\nLoading YOLOv8-face  : {YOLO_MODEL_PATH}")
    detector = YOLOv8_face(YOLO_MODEL_PATH, conf_thres=0.45, iou_thres=0.5)

    print(f"Loading ArcFace ONNX : {ARCFACE_MODEL_PATH}")
    arcface = ArcFaceONNX(ARCFACE_MODEL_PATH)

    student_dirs = sorted([
        d for d in os.listdir(ENROLLMENT_DIR)
        if os.path.isdir(os.path.join(ENROLLMENT_DIR, d))
    ])

    if not student_dirs:
        print("\nNo student folders found in enrollment directory.")
        return

    print(f"\nFound {len(student_dirs)} student(s): {student_dirs}\n")

    skipped  = []
    processed = []

    for student_name in student_dirs:
        student_path = os.path.join(ENROLLMENT_DIR, student_name)
        out_path     = os.path.join(EMBEDDINGS_DIR, f"{student_name}.pkl")

        # ── skip already embedded students ────────────────────────────
        if os.path.exists(out_path):
            print(f"[SKIP] {student_name} — embedding already exists.")
            skipped.append(student_name)
            continue

        image_files = sorted([
            f for f in os.listdir(student_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        if not image_files:
            print(f"[{student_name}] No images found, skipping.")
            continue

        print(f"Processing: {student_name}  ({len(image_files)} image(s))")
        embeddings = []

        for fname in image_files:
            fpath = os.path.join(student_path, fname)
            img   = cv2.imread(fpath)

            if img is None:
                print(f"  WARNING : Could not read {fname}, skipping.")
                continue

            emb = embed_image(img, detector, arcface)

            if emb is None:
                print(f"  WARNING : No face detected in {fname}, skipping.")
                continue

            embeddings.append(emb)
            print(f"  OK      : {fname}")

        if not embeddings:
            print(f"  No valid embeddings for '{student_name}', skipping.\n")
            continue

        with open(out_path, 'wb') as f:
            pickle.dump(embeddings, f)

        processed.append(student_name)
        print(f"  Saved {len(embeddings)}/{len(image_files)} embeddings -> {out_path}\n")

    # ── summary ───────────────────────────────────────────────────────
    print("=" * 50)
    print(f"Done.")
    print(f"  Newly processed : {len(processed)}  → {processed}")
    print(f"  Skipped         : {len(skipped)}  → {skipped}")
    print("=" * 50)


if __name__ == "__main__":
    main()