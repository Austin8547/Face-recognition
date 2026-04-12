# AutoAttend: Face Recognition Attendance System

AutoAttend is an end-to-end automated attendance tracking system powered by computer vision. Utilizing YOLOv8 for robust face detection and ArcFace ONNX for state-of-the-art embedding extraction, this system seamlessly tracks and records student attendance in real-time.

It integrates directly with a PostgreSQL backend, evaluating customized daily timetables and time-window capture logic, ensuring perfect scaling without lag or dropping frames during heavy processing.

## 🗂️ Project Workspace Structure

Here is a breakdown of the top-level directory structure:

- **`src/`**: Contains the core system logic, Python scripts, utilities, and inference managers. (See [`src/README.md`](src/README.md) for deeper technical specifications of the modules).
- **`data/`**: Used for data persistence outside the database. E.g., `data/embeddings/` holds the pre-calculated feature maps (`.pkl` files) representing enrolled students.
- **`weights/`**: Stores the high-performance compiled model weights. Ensure that your valid YOLO ONNX and ArcFace ONNX weights are located here.
- **`yolo_work/`**: The dedicated Python virtual environment for isolated dependency management.

- **`documents/`**: Miscellaneous project notes and documentation records.

## 🛠️ Prerequisites & Setup

1. **Database Runtime**: Ensure you have PostgreSQL installed. The database `attendence_db` is required and should be running on `localhost:5432`. Provide the active credentials to your `src/config/config.py` (or environment).
2. **Virtual Environment**: 
   Activate your workspace's virtual environment:
  ss
   ```
3. **Hardware Acceleration**: 
   To harness maximum frame rates, the pipeline takes advantage of the DirectML execution provider (`onnxruntime-directml`). Make sure this is properly installed via pip within your environment.

## 🚀 Usage

To start capturing video from your camera and dynamically mapping faces to class attendance, run the face recognition app as a module from this root directory:

```powershell
python -m src.face_recognition
```

> **Note**: For enrolling a new student into the database or collecting initial facial snapshots, explore the tools present inside `src/student_enroll/`.

## ✨ Core Features

- **Blazing Fast AI**: Replaced heavy CNN loads with lightweight ONNX pipelines targeting the GPU via DirectML.
- **Intelligent Time Windows**: Only marks attendance within dynamically configured class blocks (e.g., Morning Section vs Evening Section).
- **Graceful DB Threading**: Applies cool-down intervals and idempotency to drastically minimize Database `UPDATE` locks and excessive reads.
- **Fully Automated Score Allocation**: PostgreSQL handles point assignments directly via declarative SQL definitions triggered by the Python pipeline.
