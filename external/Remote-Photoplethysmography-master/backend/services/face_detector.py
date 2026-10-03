"""Face detection service using MediaPipe FaceLandmarker Tasks API."""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)


# Path to the face landmarker model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "face_landmarker.task")


class FaceDetector:
    """Detects faces and returns 478 facial landmarks using MediaPipe FaceLandmarker."""

    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: str | None = None,
    ):
        """Initialize face detector.

        Args:
            max_num_faces: Maximum number of faces to detect.
            min_detection_confidence: Minimum confidence for detection.
            min_tracking_confidence: Minimum confidence for tracking.
            model_path: Path to face_landmarker.task model file.
        """
        model = model_path or MODEL_PATH

        if not os.path.exists(model):
            raise FileNotFoundError(
                f"Face landmarker model not found at {model}. "
                "Download from: https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model),
            running_mode=RunningMode.IMAGE,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray) -> dict | None:
        """Detect face landmarks in a frame.

        Args:
            frame: BGR image as numpy array.

        Returns:
            Dict with 'landmarks' (list of (x, y) pixel coords),
            'bbox' (x, y, w, h), and 'face_center' (cx, cy),
            or None if no face detected.
        """
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect
        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return None

        face_lms = result.face_landmarks[0]
        landmarks = []
        xs, ys = [], []

        for lm in face_lms:
            px = int(lm.x * w)
            py = int(lm.y * h)
            landmarks.append((px, py))
            xs.append(px)
            ys.append(py)

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        return {
            "landmarks": landmarks,
            "bbox": (x_min, y_min, x_max - x_min, y_max - y_min),
            "face_center": ((x_min + x_max) / 2, (y_min + y_max) / 2),
        }

    def close(self):
        """Release resources."""
        self.landmarker.close()
