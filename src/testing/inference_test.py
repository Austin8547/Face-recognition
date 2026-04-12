import cv2
import numpy as np
import os
import pickle
import time
import onnxruntime as ort
from src.utility.yolo_face import YOLOv8_face
from src.utility.align import norm_crop
from src.utility.arcface import ArcFaceONNX




available = ort.get_available_providers()
print(f"Available ORT providers: {available}")
if 'DmlExecutionProvider' not in available:
    print("WARNING: DirectML not available — check onnxruntime-directml is installed")

# ── Config ────────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH      = r"C:\Users\Austin\Face-recognition\weights\yolov8n-face.onnx"
ARCFACE_MODEL_PATH   = r"C:\Users\Austin\Face-recognition\weights\w600k_r50.onnx"
EMBEDDINGS_DIR       = r"C:\Users\Austin\Face-recognition\data\embeddings"
IMAGE_SIZE           = 112
SIMILARITY_THRESHOLD = 0.4
WEBCAM_INDEX         = 0

# ── FPS optimisation knobs ────────────────────────────────────────────────────
WEBCAM_WIDTH  = 640   # force 640x480 — don't feed 1080p into YOLO
WEBCAM_HEIGHT = 480
FRAME_SKIP    = 2     # run detection every Nth frame, draw cached boxes on skipped frames
                      # 1 = every frame (slowest), 2 = good balance, 3 = fastest



# ── Helpers ───────────────────────────────────────────────────────────────────
def kpts_to_5x2(kp_flat):
    return kp_flat.reshape(5, 3)[:, :2].astype(np.float32)

def load_embeddings(embeddings_dir):
    db = {}
    if not os.path.exists(embeddings_dir):
        return db
    for fname in sorted(os.listdir(embeddings_dir)):
        if not fname.endswith('.pkl'):
            continue
        name = os.path.splitext(fname)[0]
        with open(os.path.join(embeddings_dir, fname), 'rb') as f:
            emb_list = pickle.load(f)
        db[name] = np.stack(emb_list)   # (N, 512)
    return db

def recognise(embedding, db):

    if not db:
        return 'Unknown', 0.0

    best_name  = 'Unknown'
    best_score = -1.0
    for name, stored_matrix in db.items():
        mean_score = float(np.dot(stored_matrix, embedding).mean())
        if mean_score > best_score:
            best_score = mean_score
            best_name  = name

    if best_score < SIMILARITY_THRESHOLD:
        return 'Unknown', best_score
    return best_name, best_score


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading YOLO detector...")
    detector = YOLOv8_face(YOLO_MODEL_PATH, conf_thres=0.45, iou_thres=0.5)

    print("Loading ArcFace model...")
    arcface = ArcFaceONNX(ARCFACE_MODEL_PATH)

    print("Loading embeddings database...")
    db = load_embeddings(EMBEDDINGS_DIR)
    print(f"System Ready — {len(db)} student(s) loaded\n")

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WEBCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)

    if not cap.isOpened():
        print(f"Error: Cannot open webcam at index {WEBCAM_INDEX}")
        print("Run this to find your camera index:")
        print('  python -c "import cv2; [print(f\'index {i}: {cv2.VideoCapture(i).isOpened()}\') for i in range(5)]"')
        return

    prev_time      = time.time()
    frame_idx      = 0
    cached_results = []   # list of (box, name, sim) — reused on skipped frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx    += 1
        run_detection = (frame_idx % FRAME_SKIP == 0)

        if run_detection:
            cached_results = []
            boxes, scores, _, kpts = detector.detect(frame)

            for box, kp in zip(boxes, kpts):
                # Align
                landmark = kpts_to_5x2(kp)
                aligned  = norm_crop(frame, landmark, image_size=IMAGE_SIZE)

                # Embed — always batch size 1, no DirectML BatchNorm crash
                embedding = arcface.get_embedding(aligned)

                # Recognise
                name, sim = recognise(embedding, db)
                cached_results.append((box, name, sim))

        # ── Draw cached results on every frame ─────────────────
        for box, name, sim in cached_results:
            x, y, w, h = box.astype(int)
            color = (0, 255, 0) if name != 'Unknown' else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{name} {sim:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ── FPS counter ────────────────────────────────────────
        curr_time = time.time()
        fps       = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 0), 1)

        cv2.imshow('AutoAttend - DirectML GPU', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
