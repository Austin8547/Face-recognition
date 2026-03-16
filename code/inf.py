import cv2
import numpy as np
import os
import pickle
import time
import onnxruntime as ort  # Switching to ONNX Runtime for GPU
from yolo_face import YOLOv8_face
from align import norm_crop

# ── Config ────────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH      = 'weights/yolov8n-face.onnx'
ARCFACE_MODEL_PATH   = 'weights/w600k_r50.onnx'
EMBEDDINGS_DIR       = '/home/austin/autoattend/data/embeddings'
IMAGE_SIZE           = 112
SIMILARITY_THRESHOLD = 0.4
WEBCAM_INDEX         = 0

# ── ArcFace GPU wrapper ───────────────────────────────────────────────────────
class ArcFaceONNX:
    def __init__(self, model_path):
        # Explicitly target CUDA (GPU)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def get_embedding(self, aligned_face):
        # Preprocessing: BGR to RGB, Normalize, Transpose to CHW
        img = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0
        img = img.transpose(2, 0, 1)[np.newaxis, ...] # Shape (1, 3, 112, 112)
        
        # GPU Inference
        outputs = self.session.run(None, {self.input_name: img})
        emb = outputs[0].flatten()
        
        # L2 Normalization
        return emb / (np.linalg.norm(emb) + 1e-6)

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
        db[name] = np.stack(emb_list) # Matrix (60, 512)
    return db

def recognise(embedding, db):
    best_name  = 'Unknown'
    best_score = -1.0
    
    # Vectorized cosine similarity: Matrix @ Vector
    for name, stored_matrix in db.items():
        # stored_matrix is (60, 512), embedding is (512,)
        scores = np.dot(stored_matrix, embedding)
        mean_score = float(np.mean(scores))
        
        if mean_score > best_score:
            best_score = mean_score
            best_name  = name
            
    if best_score < SIMILARITY_THRESHOLD:
        return 'Unknown', best_score
    return best_name, best_score

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Initializing GPU Session...")
    # Note: Ensure your yolo_face.py is also updated to use ONNX Runtime if possible, 
    # but ArcFace is usually the bottleneck.
    detector = YOLOv8_face(YOLO_MODEL_PATH, conf_thres=0.45, iou_thres=0.5)
    arcface  = ArcFaceONNX(ARCFACE_MODEL_PATH)

    print("Loading database...")
    db = load_embeddings(EMBEDDINGS_DIR)
    print(f"System Ready — {len(db)} student(s) loaded")

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    # Optional: Increase resolution if camera supports it
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Face Detection
        boxes, scores, _, kpts = detector.detect(frame)

        for box, det_score, kp in zip(boxes, scores, kpts):
            x, y, w, h = box.astype(int)
            
            # Landmark Alignment
            landmark = kpts_to_5x2(kp)
            aligned  = norm_crop(frame, landmark, image_size=IMAGE_SIZE)
            
            # Feature Extraction (GPU)
            embedding = arcface.get_embedding(aligned)
            
            # Matching
            name, sim = recognise(embedding, db)

            # UI Logic
            color = (0, 255, 0) if name != 'Unknown' else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{name} {sim:.2f}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # FPS Calculation
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 0), 1)

        cv2.imshow('Austin AutoAttend - GPU Inference', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()