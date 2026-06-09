import cv2
import numpy as np
import os
import pickle
import time
import onnxruntime as ort
from src.config import config
from src.utility.yolo_face import YOLOv8_face
from src.utility.align import norm_crop
from src.utility.attendance import AttendanceManager
from src.utility.arcface import ArcFaceONNX
from src.utility.database import get_connection, DatabaseManager
    

available = ort.get_available_providers()
print(f"Available ORT providers: {available}")
if 'DmlExecutionProvider' not in available:
    print("WARNING: DirectML not available — check onnxruntime-directml is installed")

# ── Config ────────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH      = config['model']['YOLO_ONNX_PATH']
ARCFACE_MODEL_PATH   = config['model']['ARCFACE_MODEL_PATH']
EMBEDDINGS_DIR       = config['directory']['EMBEDDINGS_DIR']
IMAGE_SIZE           = config['model']['IMAGE_SIZE']

SIMILARITY_THRESHOLD = config['model']['ARCFACE_SIM_THRESH']
YOLO_CONF_THRESH     = config['model']['YOLO_CONF_THRESH']
WEBCAM_INDEX         = config['hardware']['WEBCAM_INDEX']

WEBCAM_WIDTH  = 640
WEBCAM_HEIGHT = 480
FRAME_SKIP    = 2


# ── Helpers ───────────────────────────────────────────────────────────────────
def kpts_to_5x2(kp_flat):
    return kp_flat.reshape(5, 3)[:, :2].astype(np.float32)

def load_embeddings(embeddings_dir):
    db = {}
    if not os.path.exists(embeddings_dir):
        return db, np.empty((0, 512), dtype=np.float32), []

    for fname in sorted(os.listdir(embeddings_dir)):
        if not fname.endswith('.pkl'):
            continue
        name = os.path.splitext(fname)[0]
        with open(os.path.join(embeddings_dir, fname), 'rb') as f:
            emb_list = pickle.load(f)
        db[name] = np.stack(emb_list)

    all_embs  = []
    all_names = []
    for name, matrix in db.items():
        for emb in matrix:
            all_embs.append(emb)
            all_names.append(name)

    if all_embs:
        flat_matrix = np.stack(all_embs).astype(np.float32)
    else:
        flat_matrix = np.empty((0, 512), dtype=np.float32)

    return db, flat_matrix, all_names

def recognise(embedding, flat_matrix, all_names):
    if flat_matrix.shape[0] == 0:
        return 'Unknown', 0.0

    scores     = flat_matrix @ embedding
    best_idx   = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    if best_score < SIMILARITY_THRESHOLD:
        return 'Unknown', best_score
    return all_names[best_idx], best_score

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading YOLO detector...")
    detector = YOLOv8_face(YOLO_MODEL_PATH, conf_thres=YOLO_CONF_THRESH, iou_thres=0.5)

    print("Loading ArcFace model...")
    arcface = ArcFaceONNX(ARCFACE_MODEL_PATH)

    print("Loading embeddings database...")
    db, flat_matrix, all_names = load_embeddings(EMBEDDINGS_DIR)
    print(f"System Ready — {len(db)} student(s), {flat_matrix.shape[0]} embeddings\n")

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WEBCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)

    if not cap.isOpened():
        print(f"Error: Cannot open webcam at index {WEBCAM_INDEX}")
        return

    #  Attendance system init
    attendance = AttendanceManager()

    # Cooldown system + score cache (avoids per-frame DB reads)
    last_mark_time  = {}
    cached_scores   = {}   # name → last known daily_score
    COOLDOWN = 10  # seconds

    prev_time      = time.time()
    frame_idx      = 0
    cached_results = []
    avg_fps        = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame_idx += 1
        run_detection = (frame_idx % FRAME_SKIP == 0)

        if run_detection:
            cached_results = []
            current_time = time.time() # Grab time once per detection frame
            boxes, scores, _, kpts = detector.detect(frame)

            for box, kp in zip(boxes, kpts):
                # ── 1. Recognition ──
                landmark = kpts_to_5x2(kp)
                aligned  = norm_crop(frame, landmark, image_size=IMAGE_SIZE)
                embedding = arcface.get_embedding(aligned)
                name, sim = recognise(embedding, flat_matrix, all_names)

                # ── 2. DB Logic + Cooldown ──
                score = 0.0
                if name != "Unknown":
                    if name not in last_mark_time or (current_time - last_mark_time[name]) > COOLDOWN:
                        # Write to PostgreSQL; captures returned score in one round-trip
                        new_score = attendance.update_attendance(name)
                        last_mark_time[name] = current_time
                        if new_score is not None:
                            cached_scores[name] = new_score

                    # Use the cached score — no extra DB read per frame
                    score = cached_scores.get(name, 0.0)

                cached_results.append((box, name, sim, score))

        # ── Draw results (on every frame) ──
        for box, name, sim, score in cached_results:
            x, y, w, h = box.astype(int)
            color = (0, 255, 0) if name != 'Unknown' else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Show Name, Sim Score, and Daily Attendance Total
            label = f"{name} | {sim:.2f} | Score: {score}"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ── FPS counter ──
        curr_time = time.time()
        fps       = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        if avg_fps == 0.0:
            avg_fps = fps
        else:
            avg_fps = (avg_fps * 0.9) + (fps * 0.1)

        cv2.putText(frame, f"FPS: {avg_fps:.1f}  Faces: {len(cached_results)}", (20, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 0), 1)

        cv2.imshow('AutoAttend - DirectML GPU', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    attendance.close()
    print("[Attendance] DB connection closed.")

if __name__ == "__main__":
    main()