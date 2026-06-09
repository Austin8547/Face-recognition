

import sys
import os
import cv2
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utility.yolo_face import YOLOv8_face
from src.config.config import config


def main(model_path, camera_id=0, conf_thres=0.45, iou_thres=0.5):

    # Landmark colors (RetinaFace order)
    KPS_COLORS = [
        (255, 0,   0  ),   # L.Eye   - Blue
        (0,   255, 255),   # R.Eye   - Yellow
        (0,   0,   255),   # Nose    - Red
        (255, 255, 0  ),   # L.Mouth - Cyan
        (255, 0,   255),   # R.Mouth - Magenta
    ]
    KPS_LABELS = ['', '', '', '', '']

    # Load model
    detector = YOLOv8_face(model_path, conf_thres=conf_thres, iou_thres=iou_thres)

    # Open webcam
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return                             # ← return instead of exit()
                                           #   so the function exits cleanly
    prev_time = 0
    print("Press 'q' to quit | 's' to save frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect
        boxes, scores, classids, kpts = detector.detect(frame)

        # Draw
        face_count = 0
        if len(boxes) > 0:
            for box, score, kp in zip(boxes, scores, kpts):
                face_count += 1
                x, y, w, h = box.astype(int)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'{score:.2f}', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                for j in range(5):
                    px = int(kp[j * 3])
                    py = int(kp[j * 3 + 1])
                    cv2.circle(frame, (px, py), 5, KPS_COLORS[j], -1)
                    cv2.putText(frame, KPS_LABELS[j], (px + 5, py + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, KPS_COLORS[j], 1)

        # FPS
        curr_time = time.time()
        fps       = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        cv2.putText(frame, f'FPS: {fps:.1f}  Faces: {face_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow('YOLOv8-Face 5 Landmarks', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f'face_{int(time.time())}.jpg'
            cv2.imwrite(fname, frame)
            print(f'Saved: {fname}')

    cap.release()
    cv2.destroyAllWindows()


# ── Entry point — now just one clean line ────────────────────────────────────
if __name__ == "__main__":
    MODEL_PATH = config['model']['YOLO_ONNX_PATH']
    WEBCAM_INDEX = config['hardware']['WEBCAM_INDEX']
    main(model_path=MODEL_PATH, camera_id=WEBCAM_INDEX, conf_thres=0.45, iou_thres=0.5)