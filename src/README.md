# Face Recognition Attendance System (`src/`)

Welcome to the `src/` directory, which contains the entire source code and pipeline for the automated, face recognition-based attendance system. 

## 📂 Directory Structure

This module is organized to separate business logic, UI concerns, utilities, and configuration to maintain a clean codebase:

- **`face_recognition.py`**: The main entry point/runner for the real-time inference loop. It captures webcam data, runs YOLOv8 detection, aligns faces, extracts ArcFace embeddings, and tracks live attendance in the database.
- **`config/`**: Centralized configuration logic parsing constants, hyperparameters, and directory paths.
- **`student_enroll/`**: Scripts meant for new student enrollment, including data collection and generating embeddings (`.pkl` generation). 
- **`testing/`**: Isolated unit and integration tests (e.g., verifying model inference independently).
- **`utility/`**: The core workhorse modules of the application.
  - `yolo_face.py`: Custom YOLOv8 ONNX-based face detection logic.
  - `arcface.py`: ResNet ArcFace feature extraction (ONNX).
  - `align.py`: Facial landmark alignment and cropping functionalities.
  - `attendance.py`: Real-time session and time-window manager implementing cooldowns and idempotency for attendance logs. 
  - `database.py`: PostgreSQL interaction layer, preventing race conditions via atomic score recalculations and handling daily tracking rows.

## 🚀 Running the System

To avoid Python path resolution errors (like `ModuleNotFoundError`) when executing scripts inside the `src/` folder, **always run the modules from the root workspace directory** using the `-m` flag.

### Starting the Live Face Recognition Application
Make sure you have activated the virtual environment (e.g. `yolo_work`), then run:
```powershell
python -m src.face_recognition
```

##  System Architecture Pipeline

The system is optimized for speed and uses **DirectML execution providers** for GPU acceleration across ONNX models. 

1. **Capture**: A real-time stream is hooked via OpenCV.
2. **Detection**: YOLOv8 outputs bounding boxes and facial keypoints.
3. **Alignment**: The keypoints are evaluated, bringing the face into a canonical standardized crop.
4. **Extraction**: `ArcFaceONNX` generates a 512-dimensional feature embedding map for the aligned face.
5. **Recognition**: High-speed matrix multiplication compares the live face against known student databases in memory (`data/embeddings/*.pkl`).
6. **Logging**: Identified students are seamlessly passed into `AttendanceManager` which executes transactional logic on a PostgreSQL database (`attendence_db`). The database natively enforces complex daily scoring mechanisms depending on predefined class schedules without causing frame drops (FPS hits).

## 📊 Database Considerations

Ensure your `attendence_db` PostgreSQL instance is running on `localhost:5432`.
The logic enforces Foreign Key Integrity — only IDs present as `roll_no` in the top-level `students` table can be correctly matched and awarded attendance points. 
