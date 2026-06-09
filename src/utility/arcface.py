import cv2
import numpy as np
import onnxruntime as ort


class ArcFaceONNX:
    """ArcFace ONNX wrapper with DirectML (GPU) support."""

    def __init__(self, model_path):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session    = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"ArcFace running on: {self.session.get_providers()[0]}")

    def get_embedding(self, aligned_face):
        """Returns a normalised 512-d embedding (L2 norm = 1)."""
        img = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img - 127.5) / 128.0
        img = img.transpose(2, 0, 1)[np.newaxis, ...]   # (1, 3, 112, 112)
        emb = self.session.run(None, {self.input_name: img})[0].flatten()
        return emb / (np.linalg.norm(emb) + 1e-6)
