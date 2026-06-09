import onnxruntime as ort
try:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        "/home/austin/autoattend/Face-recognition/weights/w600k_r50.onnx",
        sess_options=opts,
        providers=['TensorrtExecutionProvider']
    )
    print("TensorRT successfully initialized!")
except Exception as e:
    print(f"TensorRT failed: {e}")
