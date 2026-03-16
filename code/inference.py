import cv2
import numpy as np
import os
import pickle
import time
from yolo_face import YOLOv8_face
from align import norm_crop

# ── Config ────────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH      = 'weights/yolov8n-face.onnx'
ARCFACE_MODEL_PATH   = 'weights/w600k_r50.onnx'
EMBEDDINGS_DIR       = '/home/austin/autoattend/data/embeddings'
IMAGE_SIZE           = 112
SIMILARITY_THRESHOLD = 0.4
WEBCAM_INDEX         = 0


# ── ArcFace wrapper ───────────────────────────────────────────────────────────
class ArcFaceONNX:
    def __init__(self, model_path):
        self.net = cv2.dnn.readNetFromONNX(model_path)

    def get_embedding(self, aligned_face):
        img = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0
        img = img.transpose(2, 0, 1)[np.newaxis, ...]
        self.net.setInput(img)
        emb = self.net.forward().flatten()
        return emb / np.linalg.norm(emb)


# ── Helpers ───────────────────────────────────────────────────────────────────
def kpts_to_5x2(kp_flat):
    return kp_flat.reshape(5, 3)[:, :2].astype(np.float32)


def load_embeddings(embeddings_dir):
    db = {}
    for fname in sorted(os.listdir(embeddings_dir)):
        if not fname.endswith('.pkl'):
            continue
        name = os.path.splitext(fname)[0]
        with open(os.path.join(embeddings_dir, fname), 'rb') as f:
            emb_list = pickle.load(f)
        db[name] = np.stack(emb_list)   # (N, 512)
    return db


def recognise(embedding, db):
    best_name  = 'Unknown'
    best_score = -1.0
    for name, stored in db.items():
        score = float(np.mean(stored @ embedding))
        if score > best_score:
            best_score = score
            best_name  = name
    if best_score < SIMILARITY_THRESHOLD:
        return 'Unknown', best_score
    return best_name, best_score


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading models...")
    detector = YOLOv8_face(YOLO_MODEL_PATH, conf_thres=0.45, iou_thres=0.5)
    arcface  = ArcFaceONNX(ARCFACE_MODEL_PATH)

    print("Loading embeddings...")
    db = load_embeddings(EMBEDDINGS_DIR)
    print(f"Ready — {len(db)} student(s) loaded\n")

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, scores, _, kpts = detector.detect(frame)

        if len(boxes) > 0:
            for box, det_score, kp in zip(boxes, scores, kpts):
                x, y, w, h = box.astype(int)

                landmark  = kpts_to_5x2(kp)
                aligned   = norm_crop(frame, landmark, image_size=IMAGE_SIZE)
                embedding = arcface.get_embedding(aligned)
                name, sim = recognise(embedding, db)

                colour = (0, 255, 0) if name != 'Unknown' else (0, 0, 255)
                label  = f'{name}  {sim:.2f}'

                cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

        # FPS
        curr_time = time.time()
        fps       = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()