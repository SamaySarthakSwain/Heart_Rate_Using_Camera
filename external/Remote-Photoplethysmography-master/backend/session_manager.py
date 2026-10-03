"""Session manager orchestrating the full rPPG processing pipeline."""

import time
import uuid
import base64
import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum

from backend.services.face_detector import FaceDetector
from backend.services.roi_extractor import ROIExtractor
from backend.services.signal_buffer import SignalBuffer
from backend.services.chrom_rppg import chrom_rppg
from backend.services.pos_rppg import pos_rppg
from backend.services.signal_filter import process_rppg_signal
from backend.services.hr_estimator import HREstimator
from backend.services.signal_quality import SignalQualityEvaluator
from backend.utils.smoothing_utils import HRSmoother
from backend.utils.motion_utils import MotionDetector


class SessionState(str, Enum):
    ACTIVE = "active"
    ENDED = "ended"


@dataclass
class SessionSummary:
    """Post-session summary metrics."""
    average_hr: float = 0.0
    median_hr: float = 0.0
    std_hr: float = 0.0
    confidence_score: float = 0.0
    signal_stability: float = 0.0
    duration_seconds: float = 0.0
    total_frames: int = 0
    valid_frames: int = 0


@dataclass
class SessionData:
    """Data container for a single measurement session."""
    session_id: str = ""
    state: SessionState = SessionState.ACTIVE
    start_time: float = 0.0
    end_time: float = 0.0
    total_frames: int = 0
    valid_frames: int = 0
    hr_history: list = field(default_factory=list)
    quality_history: list = field(default_factory=list)
    brightness_history: list = field(default_factory=list)


class Session:
    """A single rPPG measurement session with full pipeline orchestration."""

    def __init__(self, session_id: str | None = None):
        """Initialize a new session.

        Args:
            session_id: Optional session ID. Generated if not provided.
        """
        self.data = SessionData(
            session_id=session_id or str(uuid.uuid4()),
            start_time=time.time(),
        )

        # Initialize pipeline components
        self.face_detector = FaceDetector()
        self.roi_extractor = ROIExtractor()
        self.signal_buffer = SignalBuffer(max_size=300, min_size=90)
        self.hr_estimator = HREstimator()
        self.quality_evaluator = SignalQualityEvaluator()
        self.hr_smoother = HRSmoother(max_hr_change=10.0, ema_alpha=0.15, history_size=30)
        self.motion_detector = MotionDetector(motion_threshold=15.0, history_size=30)

        # Latest results cache
        self._latest_hr: float | None = None
        self._latest_quality: dict = {}
        self._latest_motion: dict = {}

    @property
    def session_id(self) -> str:
        return self.data.session_id

    @property
    def is_active(self) -> bool:
        return self.data.state == SessionState.ACTIVE

    @property
    def elapsed_seconds(self) -> float:
        if self.data.state == SessionState.ENDED:
            return self.data.end_time - self.data.start_time
        return time.time() - self.data.start_time

    def process_frame(self, frame: np.ndarray) -> dict:
        """Process a single video frame through the full rPPG pipeline.

        Args:
            frame: BGR image as numpy array.

        Returns:
            Dict with 'hr_bpm', 'confidence', 'motion', 'lighting',
            'face_detected', 'buffer_ready', 'elapsed'.
        """
        if not self.is_active:
            return self._empty_frame_result()

        self.data.total_frames += 1

        # Step 1: Detect face
        detection = self.face_detector.detect(frame)
        if detection is None:
            return self._make_result(face_detected=False)

        # Step 2: Track motion
        cx, cy = detection["face_center"]
        motion_result = self.motion_detector.update(cx, cy)
        self._latest_motion = motion_result

        # Step 3: Extract ROI RGB
        roi_result = self.roi_extractor.extract(frame, detection["landmarks"])
        if roi_result is None:
            return self._make_result(face_detected=True)

        r, g, b = roi_result["rgb"]
        brightness = roi_result["brightness"]

        # Step 4: Add to signal buffer
        self.signal_buffer.add_sample(r, g, b, brightness)
        self.data.brightness_history.append(brightness)
        self.data.valid_frames += 1

        # Step 5: Attempt rPPG processing if buffer ready
        if not self.signal_buffer.is_ready():
            return self._make_result(face_detected=True, buffer_ready=False)

        window = self.signal_buffer.get_window()
        if window is None:
            return self._make_result(face_detected=True, buffer_ready=False)

        # Step 6: Run CHROM + POS algorithms
        chrom_signal = chrom_rppg(window["R"], window["G"], window["B"])
        pos_signal = pos_rppg(window["R"], window["G"], window["B"])

        # Step 7: Estimate HR with fusion
        hr_result = self.hr_estimator.estimate_fused(
            chrom_signal, pos_signal, window["fps"]
        )

        if hr_result["valid"]:
            smoothed_hr = self.hr_smoother.update(hr_result["hr_bpm"])
            if smoothed_hr is not None:
                self.data.hr_history.append(smoothed_hr)

        # Use stability-gated HR for display (only when stable)
        stable_hr = self.hr_smoother.get_stable_hr()
        if stable_hr is not None:
            self._latest_hr = stable_hr
        elif self._latest_hr is None and self.data.hr_history:
            # Fallback: show raw smoothed value until stable
            pass

        # Step 8: Evaluate signal quality
        quality = self.quality_evaluator.evaluate(
            snr=hr_result["snr"],
            motion_score=motion_result["motion_score"],
            brightness=brightness,
            hr_stability=self.hr_smoother.get_stability(),
        )
        self._latest_quality = quality
        self.data.quality_history.append(quality["confidence"])

        return self._make_result(
            face_detected=True,
            buffer_ready=True,
            hr_bpm=self._latest_hr,
            quality=quality,
        )

    def end_session(self) -> SessionSummary:
        """End the session and compute summary metrics.

        Returns:
            SessionSummary with final metrics.
        """
        self.data.state = SessionState.ENDED
        self.data.end_time = time.time()

        summary = SessionSummary(
            duration_seconds=self.elapsed_seconds,
            total_frames=self.data.total_frames,
            valid_frames=self.data.valid_frames,
        )

        if self.data.hr_history:
            hr_arr = np.array(self.data.hr_history)
            summary.average_hr = float(np.mean(hr_arr))
            summary.median_hr = float(np.median(hr_arr))
            summary.std_hr = float(np.std(hr_arr))

        if self.data.quality_history:
            summary.confidence_score = float(np.mean(self.data.quality_history))

        summary.signal_stability = self.hr_smoother.get_stability()

        # Clean up
        self.face_detector.close()

        return summary

    def get_status(self) -> dict:
        """Get current session status."""
        return {
            "session_id": self.session_id,
            "state": self.data.state.value,
            "elapsed_seconds": self.elapsed_seconds,
            "total_frames": self.data.total_frames,
            "valid_frames": self.data.valid_frames,
            "current_hr": self._latest_hr,
            "quality": self._latest_quality,
            "motion": self._latest_motion,
            "buffer_size": self.signal_buffer.get_size(),
            "buffer_ready": self.signal_buffer.is_ready(),
        }

    def _make_result(
        self,
        face_detected: bool = False,
        buffer_ready: bool = False,
        hr_bpm: float | None = None,
        quality: dict | None = None,
    ) -> dict:
        motion = self._latest_motion
        return {
            "face_detected": face_detected,
            "buffer_ready": buffer_ready,
            "hr_bpm": hr_bpm or self._latest_hr,
            "confidence": quality["confidence"] if quality else (self._latest_quality.get("confidence", 0.0)),
            "quality_label": quality["quality_label"] if quality else self._latest_quality.get("quality_label", "poor"),
            "motion_detected": motion.get("motion_detected", False),
            "motion_score": motion.get("motion_score", 1.0),
            "elapsed_seconds": self.elapsed_seconds,
            "total_frames": self.data.total_frames,
        }

    def _empty_frame_result(self) -> dict:
        return self._make_result()


class SessionManager:
    """Manages multiple concurrent sessions."""

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def create_session(self) -> Session:
        """Create and register a new session."""
        session = Session()
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def end_session(self, session_id: str) -> SessionSummary | None:
        """End a session and return its summary."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        return session.end_session()

    def remove_session(self, session_id: str):
        """Remove a session from the manager."""
        self._sessions.pop(session_id, None)

    def list_sessions(self) -> list[str]:
        """List all session IDs."""
        return list(self._sessions.keys())
