import cv2
import numpy as np
import math
import time
import onnxruntime as ort


class YOLOv8_face:
    def __init__(self, path, conf_thres=0.45, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold  = iou_thres
        self.class_names    = ['face']
        self.num_classes    = len(self.class_names)
        self.input_height   = 640
        self.input_width    = 640
        self.reg_max        = 16
        self.project        = np.arange(self.reg_max)
        self.strides        = (8, 16, 32)
        self.feats_hw       = [(math.ceil(self.input_height / self.strides[i]),
                                math.ceil(self.input_width  / self.strides[i]))
                               for i in range(len(self.strides))]
        self.anchors        = self.make_anchors(self.feats_hw)

        # ── FIX: precompute kpt pad/scale index masks once ───────────────────
        # Old code called np.tile(...) inside post_process on every single frame
        # — allocating a new array each call for values that never change.
        # These two arrays are built once here and reused every detection pass.
        # _kpt_pad_idx / _kpt_scale_idx store which columns get padw/padh/scale_w/scale_h
        # so post_process can do: kpt_pad[0::3] = padw etc. in-place without tile().
        self._kpt_pad   = np.zeros(15, dtype=np.float32)   # [padw,padh,0, padw,padh,0, ...]
        self._kpt_scale = np.ones(15,  dtype=np.float32)   # [sw,  sh,  1, sw,  sh,  1, ...]
        # Index positions: x=0,3,6,9,12  y=1,4,7,10,13  conf=2,5,8,11,14
        self._kpt_x_idx    = np.arange(0, 15, 3)   # [0,3,6,9,12]
        self._kpt_y_idx    = np.arange(1, 15, 3)   # [1,4,7,10,13]

        # ── ONNX Runtime with DirectML (GPU on Windows, no CUDA needed) ──────
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        self.session    = ort.InferenceSession(path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        active          = self.session.get_providers()[0]
        print(f"YOLO running on: {active}")

    def make_anchors(self, feats_hw, grid_cell_offset=0.5):
        anchor_points = {}
        for i, stride in enumerate(self.strides):
            h, w = feats_hw[i]
            x = np.arange(0, w) + grid_cell_offset
            y = np.arange(0, h) + grid_cell_offset
            sx, sy = np.meshgrid(x, y)
            anchor_points[stride] = np.stack((sx, sy), axis=-1).reshape(-1, 2)
        return anchor_points

    def softmax(self, x, axis=-1):
        # Subtract max before exp to prevent overflow (inf/nan).
        # Without this, large raw logits produce inf, making softmax return nan
        # and causing YOLO to output phantom boxes silently under bright light
        # or extreme angles. Cost: one extra .max() call — essentially free.
        x = x - x.max(axis=axis, keepdims=True)
        x_exp = np.exp(x)
        return x_exp / np.sum(x_exp, axis=axis, keepdims=True)

    def resize_image(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.input_width, self.input_height
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_height, int(self.input_width / hw_scale)
                img  = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_width - neww) * 0.5)
                img  = cv2.copyMakeBorder(img, 0, 0, left, self.input_width - neww - left,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))
            else:
                newh, neww = int(self.input_height * hw_scale), self.input_width
                img  = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top  = int((self.input_height - newh) * 0.5)
                img  = cv2.copyMakeBorder(img, top, self.input_height - newh - top, 0, 0,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:
            img = cv2.resize(srcimg, (self.input_width, self.input_height),
                             interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left

    def detect(self, srcimg):
        input_img, newh, neww, padh, padw = self.resize_image(
            cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB)
        )
        scale_h = srcimg.shape[0] / newh
        scale_w = srcimg.shape[1] / neww

        # Normalise and convert to NCHW float32
        blob = (input_img.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]

        # GPU inference via DirectML
        outputs = self.session.run(None, {self.input_name: blob})

        det_bboxes, det_conf, det_classid, landmarks = self.post_process(
            outputs, scale_h, scale_w, padh, padw
        )
        return det_bboxes, det_conf, det_classid, landmarks

    def post_process(self, preds, scale_h, scale_w, padh, padw):
        bboxes, scores, landmarks = [], [], []

        # FIX: build pad/scale vectors from precomputed index arrays — no np.tile() per frame
        kpt_pad   = np.zeros(15, dtype=np.float32)
        kpt_scale = np.ones(15,  dtype=np.float32)
        kpt_pad[self._kpt_x_idx]   = padw
        kpt_pad[self._kpt_y_idx]   = padh
        kpt_scale[self._kpt_x_idx] = scale_w
        kpt_scale[self._kpt_y_idx] = scale_h
        # Shape to (1,15) for broadcast against (N,15) kpts arrays
        kpt_pad   = kpt_pad.reshape(1, 15)
        kpt_scale = kpt_scale.reshape(1, 15)

        for i, pred in enumerate(preds):
            stride = int(self.input_height / pred.shape[2])
            pred   = pred.transpose((0, 2, 3, 1))

            box  = pred[..., :self.reg_max * 4]
            cls  = 1 / (1 + np.exp(-pred[..., self.reg_max * 4:-15])).reshape((-1, 1))
            kpts = pred[..., -15:].reshape((-1, 15))

            tmp       = box.reshape(-1, 4, self.reg_max)
            # Uses stable softmax — no nan/inf risk from large logits
            bbox_pred = self.softmax(tmp, axis=-1)
            bbox_pred = np.dot(bbox_pred, self.project).reshape((-1, 4))

            bbox = self.distance2bbox(self.anchors[stride], bbox_pred,
                                      max_shape=(self.input_height, self.input_width)) * stride

            kpts[:, 0::3] = (kpts[:, 0::3] * 2.0 +
                             (self.anchors[stride][:, 0].reshape((-1, 1)) - 0.5)) * stride
            kpts[:, 1::3] = (kpts[:, 1::3] * 2.0 +
                             (self.anchors[stride][:, 1].reshape((-1, 1)) - 0.5)) * stride
            kpts[:, 2::3] = 1 / (1 + np.exp(-kpts[:, 2::3]))

            bbox -= np.array([[padw, padh, padw, padh]])
            bbox *= np.array([[scale_w, scale_h, scale_w, scale_h]])

            # FIX: use precomputed vectors instead of np.tile() called inline
            kpts -= kpt_pad
            kpts *= kpt_scale

            bboxes.append(bbox)
            scores.append(cls)
            landmarks.append(kpts)

        bboxes    = np.concatenate(bboxes,    axis=0)
        scores    = np.concatenate(scores,    axis=0)
        landmarks = np.concatenate(landmarks, axis=0)

        bboxes_wh         = bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]
        classIds    = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        mask        = confidences > self.conf_threshold
        bboxes_wh   = bboxes_wh[mask]
        confidences = confidences[mask]
        classIds    = classIds[mask]
        landmarks   = landmarks[mask]

        indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(),
                                   self.conf_threshold, self.iou_threshold)
        if len(indices) > 0:
            indices     = np.array(indices).flatten()
            mlvl_bboxes = bboxes_wh[indices]
            confidences = confidences[indices]
            classIds    = classIds[indices]
            landmarks   = landmarks[indices]
            return mlvl_bboxes, confidences, classIds, landmarks
        else:
            return np.array([]), np.array([]), np.array([]), np.array([])

    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)
