import cv2
import numpy as np
import os
import pickle
from yolo_face import YOLOv8_face
from align import norm_crop

# ── Config ────────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH    = 'weights/yolov8n-face.onnx'
ARCFACE_MODEL_PATH = 'weights/w600k_r50.onnx'
ENROLLMENT_DIR     = '/home/austin/autoattend/data/enrollment'
EMBEDDINGS_DIR     = '/home/austin/autoattend/data/embeddings'
IMAGE_SIZE         = 112                # ArcFace input size


# ── ArcFace wrapper ───────────────────────────────────────────────────────────
class ArcFaceONNX:
    def __init__(self, model_path):
        self.net = cv2.dnn.readNetFromONNX(model_path)

    def preprocess(self, aligned_face):
        img = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0          # ArcFace normalisation
        img = img.transpose(2, 0, 1)         # HWC → CHW
        blob = img[np.newaxis, ...]           # (1, 3, 112, 112)
        return blob

    def get_embedding(self, aligned_face):
        """Returns a normalised 512-d embedding (L2 norm = 1)."""
        blob = self.preprocess(aligned_face)
        self.net.setInput(blob)
        emb = self.net.forward()
        emb = emb.flatten()
        emb = emb / np.linalg.norm(emb)
        return emb


# ── Landmark reshape helper ───────────────────────────────────────────────────
def kpts_to_5x2(kp_flat):
    """
    yolo_face returns landmarks as flat (15,) array: [x0,y0,conf0, x1,y1,conf1, ...]
    align.estimate_norm() expects shape (5, 2) — x,y only.
    """
    kp = kp_flat.reshape(5, 3)
    return kp[:, :2].astype(np.float32)


# ── Core per-image function ───────────────────────────────────────────────────
def embed_image(img_bgr, detector, arcface):
    """
    Detect face → align → embed.
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


# ── Main pipeline ─────────────────────────────────────────────────────────────
def main():
    # Validate directories
    if not os.path.isdir(ENROLLMENT_DIR):
        print(f"Error: enrollment directory not found:\n  {ENROLLMENT_DIR}")
        return

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    # Load models
    print(f"Loading YOLOv8-face  : {YOLO_MODEL_PATH}")
    detector = YOLOv8_face(YOLO_MODEL_PATH, conf_thres=0.45, iou_thres=0.5)

    print(f"Loading ArcFace ONNX : {ARCFACE_MODEL_PATH}")
    arcface = ArcFaceONNX(ARCFACE_MODEL_PATH)

    student_dirs = sorted([
        d for d in os.listdir(ENROLLMENT_DIR)
        if os.path.isdir(os.path.join(ENROLLMENT_DIR, d))
    ])

    if not student_dirs:
        print("No student folders found in enrollment directory.")
        return

    print(f"\nFound {len(student_dirs)} student(s): {student_dirs}\n")

    for student_name in student_dirs:
        student_path = os.path.join(ENROLLMENT_DIR, student_name)
        image_files  = sorted([
            f for f in os.listdir(student_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        if not image_files:
            print(f"[{student_name}] No images found, skipping.")
            continue

        print(f"Processing: {student_name}  ({len(image_files)} images)")
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

        if not embeddings:
            print(f"  No valid embeddings for '{student_name}', skipping.\n")
            continue

        # Save as /home/austin/autoattend/data/embeddings/student1.pkl
        out_path = os.path.join(EMBEDDINGS_DIR, f"{student_name}.pkl")
        with open(out_path, 'wb') as f:
            pickle.dump(embeddings, f)

        print(f"  Saved {len(embeddings)}/{len(image_files)} embeddings → {out_path}\n")

    print("Done. All students processed.")


if __name__ == "__main__":
    main()