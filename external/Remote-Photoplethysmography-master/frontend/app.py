"""
Streamlit rPPG Heart Rate Monitor — 6-Panel Scientific Dashboard.

Real-time webcam-based heart rate monitoring using CHROM + POS algorithms.
Organized into: Header → Acquisition → Signal Intelligence → HR Result →
Scientific Visualization → Session Analytics.
"""

import time
import io
import os
import sys

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.face_detector import FaceDetector
from backend.services.roi_extractor import ROIExtractor
from backend.services.signal_buffer import SignalBuffer
from backend.services.chrom_rppg import chrom_rppg
from backend.services.pos_rppg import pos_rppg
from backend.services.hr_estimator import HREstimator
from backend.services.signal_quality import SignalQualityEvaluator
from backend.utils.smoothing_utils import HRSmoother
from backend.utils.motion_utils import MotionDetector
from backend.utils.fft_utils import compute_psd

# --- Configuration ---
MIN_SESSION_SECONDS = 60
FRAME_SKIP = 2

# --- Page Config ---
st.set_page_config(
    page_title="Advanced rPPG Heart Rate Monitor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ═══════════════════════════════════════════════════════════════════
#  CSS Theme — Biomedical Dark
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
    }

    /* ── Header ── */
    .app-header {
        text-align: center; padding: 1.5rem 0 0.8rem;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 1.2rem;
    }
    .app-header h1 {
        background: linear-gradient(135deg, #e74c3c, #ff6b6b, #ee5a24);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 2.4rem; font-weight: 800; letter-spacing: -0.5px; margin: 0;
    }
    .app-header p { color: #8892b0; font-size: 0.95rem; margin-top: 4px; }

    /* ── Status Badges ── */
    .status-badge {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 5px 16px; border-radius: 20px;
        font-size: 0.78rem; font-weight: 600; letter-spacing: 0.5px;
    }
    .status-initializing { background: rgba(128,128,128,0.15); color: #aaa; border: 1px solid rgba(128,128,128,0.3); }
    .status-detecting    { background: rgba(52,152,219,0.15); color: #3498db; border: 1px solid rgba(52,152,219,0.3); }
    .status-stabilizing  { background: rgba(243,156,18,0.15); color: #f39c12; border: 1px solid rgba(243,156,18,0.3); }
    .status-measuring    { background: rgba(46,204,113,0.15); color: #2ecc71; border: 1px solid rgba(46,204,113,0.3); }
    .status-complete     { background: rgba(231,76,60,0.15); color: #e74c3c; border: 1px solid rgba(231,76,60,0.3); }
    .status-ready        { background: rgba(162,155,254,0.15); color: #a29bfe; border: 1px solid rgba(162,155,254,0.3); }

    /* ── Section Headers ── */
    .section-header {
        color: #ccd6f6; font-size: 1.05rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 2px;
        padding: 0.8rem 0 0.5rem; margin-top: 1rem;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px; padding: 1.2rem 1rem;
        text-align: center; transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(255,255,255,0.15);
        transform: translateY(-2px);
    }
    .metric-label {
        color: #8892b0; font-size: 0.7rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 6px;
    }
    .metric-value {
        font-size: 2.2rem; font-weight: 800; margin: 0; line-height: 1.1;
    }
    .metric-unit { color: #8892b0; font-size: 0.78rem; font-weight: 400; margin-top: 3px; }

    /* ── Quality Bar ── */
    .quality-bar {
        width: 100%; height: 8px;
        background: rgba(255,255,255,0.1);
        border-radius: 4px; overflow: hidden; margin-top: 8px;
    }
    .quality-fill {
        height: 100%; border-radius: 4px;
        transition: width 0.5s ease, background 0.5s ease;
    }

    /* ── HR Display ── */
    .hr-panel {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px; padding: 2rem; text-align: center;
    }
    .hr-big { color: #ff6b6b; font-size: 4.5rem; font-weight: 800; line-height: 1; }
    .hr-unit { color: #8892b0; font-size: 1rem; margin-top: 4px; }
    .hr-stabilizing { color: #f39c12; font-size: 1.1rem; font-weight: 600; }
    .algo-row {
        display: flex; justify-content: center; gap: 2rem;
        margin-top: 1rem; padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.06);
    }
    .algo-item { text-align: center; }
    .algo-name { color: #8892b0; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; }
    .algo-val { color: #ccd6f6; font-size: 1.1rem; font-weight: 700; }

    /* ── Summary Card ── */
    .summary-card {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px; padding: 2rem; margin-top: 0.5rem;
    }
    .summary-title {
        color: #ccd6f6; font-size: 1.3rem; font-weight: 700;
        margin-bottom: 1.5rem; text-align: center;
    }
    .summary-metric { text-align: center; padding: 0.8rem; }
    .summary-metric .value { font-size: 1.8rem; font-weight: 800; line-height: 1.1; }
    .summary-metric .label {
        color: #8892b0; font-size: 0.7rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 1.2px; margin-top: 4px;
    }

    /* ── Guidance Overlay ── */
    .guidance {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px; padding: 1rem;
    }
    .guidance-item {
        color: #8892b0; font-size: 0.8rem; padding: 3px 0;
    }
    .guidance-item span { margin-right: 6px; }

    /* ── Animations ── */
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.08); }
    }
    .pulse-icon { animation: pulse 1s ease-in-out infinite; display: inline-block; }

    /* ── Hide Streamlit chrome ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #e74c3c, #c0392b) !important;
        color: white !important; border: none !important;
        border-radius: 12px !important; padding: 0.6rem 2rem !important;
        font-weight: 600 !important; font-size: 0.95rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  Pipeline State
# ═══════════════════════════════════════════════════════════════════

class PipelineState:
    """Full rPPG processing state with caches for visualization."""

    def __init__(self):
        self.face_detector = FaceDetector()
        self.roi_extractor = ROIExtractor()
        self.signal_buffer = SignalBuffer(max_size=300, min_size=50)
        self.hr_estimator = HREstimator()
        self.quality_evaluator = SignalQualityEvaluator()
        self.hr_smoother = HRSmoother(max_hr_change=10.0, ema_alpha=0.15, history_size=30)
        self.motion_detector = MotionDetector(motion_threshold=15.0, history_size=30)

        # Current metrics
        self.hr_bpm: float | None = None
        self.stable_hr: float | None = None
        self.hr_is_stable = False
        self.confidence = 0.0
        self.quality_label = "waiting"
        self.motion_detected = False
        self.motion_score = 1.0
        self.face_detected = False
        self.buffer_ready = False
        self.total_frames = 0
        self.valid_frames = 0

        # Per-component scores
        self.snr = 0.0
        self.snr_score = 0.0
        self.lighting_score = 0.0
        self.stability_score = 0.0
        self.brightness = 0.0
        self.fps = 0.0

        # Algorithm transparency
        self.chrom_hr = 0.0
        self.pos_hr = 0.0
        self.fused_hr = 0.0

        # ROI details
        self.roi_details: dict = {}

        # Visualization caches
        self.last_green_signal: np.ndarray | None = None
        self.last_chrom_signal: np.ndarray | None = None
        self.last_pos_signal: np.ndarray | None = None
        self.last_freqs: np.ndarray | None = None
        self.last_psd: np.ndarray | None = None
        self.last_peak_freq: float = 0.0

        # History for summary
        self.hr_history: list[float] = []
        self.quality_history: list[float] = []

        # Pipeline status string
        self.pipeline_status = "initializing"

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through the full pipeline, return annotated frame."""
        self.total_frames += 1

        # 1. Detect face
        detection = self.face_detector.detect(frame)
        if detection is None:
            self.face_detected = False
            self.pipeline_status = "detecting"
            return self._draw_overlay(frame)

        self.face_detected = True

        # 2. Motion tracking
        cx, cy = detection["face_center"]
        motion_result = self.motion_detector.update(cx, cy)
        self.motion_detected = motion_result["motion_detected"]
        self.motion_score = motion_result["motion_score"]

        # 3. Extract ROI
        roi_result = self.roi_extractor.extract(frame, detection["landmarks"])
        if roi_result is None:
            self.pipeline_status = "detecting"
            return self._draw_overlay(frame)

        r, g, b = roi_result["rgb"]
        self.brightness = roi_result["brightness"]
        self.roi_details = roi_result.get("roi_details", {})
        self.valid_frames += 1

        # 4. Buffer
        self.signal_buffer.add_sample(r, g, b, self.brightness)

        if not self.signal_buffer.is_ready():
            self.buffer_ready = False
            self.pipeline_status = "buffering"
            frame = self.roi_extractor.draw_rois(frame, detection["landmarks"])
            return self._draw_overlay(frame)

        self.buffer_ready = True
        window = self.signal_buffer.get_window()
        if window is None:
            frame = self.roi_extractor.draw_rois(frame, detection["landmarks"])
            return self._draw_overlay(frame)

        self.fps = window["fps"]

        # 5. CHROM + POS
        chrom_signal = chrom_rppg(window["R"], window["G"], window["B"])
        pos_signal = pos_rppg(window["R"], window["G"], window["B"])

        # Cache for visualization
        self.last_green_signal = window["G"].copy()
        self.last_chrom_signal = chrom_signal.copy()
        self.last_pos_signal = pos_signal.copy()

        # 6. HR estimation
        hr_result = self.hr_estimator.estimate_fused(chrom_signal, pos_signal, window["fps"])
        if hr_result["valid"]:
            smoothed = self.hr_smoother.update(hr_result["hr_bpm"])
            if smoothed is not None:
                self.hr_bpm = smoothed
                self.hr_history.append(smoothed)
            self.chrom_hr = hr_result.get("chrom_hr", 0.0)
            self.pos_hr = hr_result.get("pos_hr", 0.0)
            self.fused_hr = hr_result.get("hr_bpm", 0.0)
            self.last_peak_freq = hr_result.get("peak_freq", 0.0)
            self.snr = hr_result.get("snr", 0.0)

            # Cache PSD for FFT visualization
            from backend.services.signal_filter import process_rppg_signal
            filtered = process_rppg_signal(chrom_signal, window["fps"])
            freqs, psd = compute_psd(filtered, window["fps"])
            self.last_freqs = freqs
            self.last_psd = psd

        # Update stability-gated HR
        self.stable_hr = self.hr_smoother.get_stable_hr()
        self.hr_is_stable = self.hr_smoother.is_stable()

        # 7. Quality
        quality = self.quality_evaluator.evaluate(
            snr=hr_result["snr"],
            motion_score=motion_result["motion_score"],
            brightness=self.brightness,
            hr_stability=self.hr_smoother.get_stability(),
        )
        self.confidence = quality["confidence"]
        self.quality_label = quality["quality_label"]
        self.snr_score = quality.get("snr_score", 0.0)
        self.lighting_score = quality.get("lighting_score", 0.0)
        self.stability_score = quality.get("stability_score", 0.0)
        self.quality_history.append(quality["confidence"])

        # Pipeline status
        if self.hr_is_stable:
            self.pipeline_status = "measuring"
        else:
            self.pipeline_status = "stabilizing"

        # Draw ROI regions on frame
        frame = self.roi_extractor.draw_rois(frame, detection["landmarks"])
        return self._draw_overlay(frame)

    def get_summary(self) -> dict:
        hr_arr = np.array(self.hr_history) if self.hr_history else np.array([0.0])
        q_arr = np.array(self.quality_history) if self.quality_history else np.array([0.0])
        return {
            "average_hr": float(np.mean(hr_arr)),
            "median_hr": float(np.median(hr_arr)),
            "min_hr": float(np.min(hr_arr)) if len(hr_arr) > 1 else 0.0,
            "max_hr": float(np.max(hr_arr)) if len(hr_arr) > 1 else 0.0,
            "std_hr": float(np.std(hr_arr)),
            "confidence_score": float(np.mean(q_arr)),
            "signal_stability": float(max(0, 1.0 - np.std(hr_arr) / 20.0)) if len(hr_arr) > 1 else 0.0,
            "total_frames": self.total_frames,
            "valid_frames": self.valid_frames,
            "avg_snr": float(self.snr),
            "avg_lighting": float(self.lighting_score),
            "avg_motion": float(self.motion_score),
        }

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        if self.face_detected:
            cv2.circle(frame, (20, 25), 6, (0, 255, 100), -1)
            cv2.putText(frame, "Face OK", (35, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (20, 25), 6, (0, 0, 255), -1)
            cv2.putText(frame, "No Face", (35, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)

        if self.hr_bpm and self.hr_bpm > 0:
            cv2.putText(frame, f"HR: {self.hr_bpm:.0f} BPM", (w - 200, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 100, 255), 2, cv2.LINE_AA)

        if self.motion_detected:
            cv2.putText(frame, "MOTION", (w // 2 - 40, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2, cv2.LINE_AA)

        bar_w = int(w * self.confidence)
        bar_color = (0, 255, 100) if self.confidence >= 0.7 else (0, 200, 255) if self.confidence >= 0.4 else (0, 0, 255)
        cv2.rectangle(frame, (0, h - 6), (bar_w, h), bar_color, -1)
        return frame

    def reset(self):
        self.signal_buffer.clear()
        self.hr_smoother.reset()
        self.motion_detector.reset()
        self.hr_bpm = None
        self.stable_hr = None
        self.hr_is_stable = False
        self.confidence = 0.0
        self.quality_label = "waiting"
        self.motion_detected = False
        self.face_detected = False
        self.buffer_ready = False
        self.total_frames = 0
        self.valid_frames = 0
        self.hr_history = []
        self.quality_history = []
        self.chrom_hr = 0.0
        self.pos_hr = 0.0
        self.fused_hr = 0.0
        self.snr = 0.0
        self.snr_score = 0.0
        self.lighting_score = 0.0
        self.stability_score = 0.0
        self.brightness = 0.0
        self.fps = 0.0
        self.roi_details = {}
        self.last_green_signal = None
        self.last_chrom_signal = None
        self.last_pos_signal = None
        self.last_freqs = None
        self.last_psd = None
        self.last_peak_freq = 0.0
        self.pipeline_status = "initializing"

    def close(self):
        self.face_detector.close()


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def format_time(seconds: float) -> str:
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"

def quality_color(v: float) -> str:
    if v >= 0.7: return "#2ecc71"
    if v >= 0.4: return "#f39c12"
    return "#e74c3c"

def status_class(status: str) -> str:
    return {
        "initializing": "status-initializing",
        "detecting": "status-detecting",
        "buffering": "status-detecting",
        "stabilizing": "status-stabilizing",
        "measuring": "status-measuring",
        "complete": "status-complete",
        "ready": "status-ready",
    }.get(status, "status-initializing")

def status_label(status: str) -> str:
    return {
        "initializing": "● Initializing Camera",
        "detecting": "● Detecting Face",
        "buffering": "● Extracting ROI",
        "stabilizing": "● Stabilizing Signal",
        "measuring": "● Measuring",
        "complete": "● Measurement Complete",
        "ready": "● Ready",
    }.get(status, "● Initializing")


# ═══════════════════════════════════════════════════════════════════
#  Session State Init
# ═══════════════════════════════════════════════════════════════════

if "pipeline" not in st.session_state:
    st.session_state.pipeline = PipelineState()
if "session_active" not in st.session_state:
    st.session_state.session_active = False
if "session_summary" not in st.session_state:
    st.session_state.session_summary = None
if "start_time" not in st.session_state:
    st.session_state.start_time = 0.0

pipeline = st.session_state.pipeline


# ═══════════════════════════════════════════════════════════════════
#  PANEL 1 — Application Header
# ═══════════════════════════════════════════════════════════════════

status = pipeline.pipeline_status if st.session_state.session_active else ("complete" if st.session_state.session_summary else "ready")
st.markdown(f"""
<div class="app-header">
    <h1>🫀 Advanced rPPG Heart Rate Monitor</h1>
    <p>Real-Time Remote Photoplethysmography System</p>
    <div style="margin-top:10px;">
        <span class="status-badge {status_class(status)}">{status_label(status)}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Control Buttons ──
c1, c2, c3 = st.columns([1, 1, 3])
with c1:
    start_clicked = st.button("🟢 Start Session", disabled=st.session_state.session_active, width='stretch')
with c2:
    stop_clicked = st.button("🔴 End Session", disabled=not st.session_state.session_active, width='stretch')

if start_clicked:
    pipeline.reset()
    st.session_state.session_active = True
    st.session_state.session_summary = None
    st.session_state.start_time = time.time()
    st.rerun()

if stop_clicked:
    elapsed = time.time() - st.session_state.start_time
    summary = pipeline.get_summary()
    summary["duration_seconds"] = elapsed
    summary["hr_history"] = list(pipeline.hr_history)
    st.session_state.session_summary = summary
    st.session_state.session_active = False
    pipeline.pipeline_status = "complete"
    st.rerun()


# ═══════════════════════════════════════════════════════════════════
#  PANEL 2 — Acquisition Panel
# ═══════════════════════════════════════════════════════════════════

if st.session_state.session_active or st.session_state.session_summary is None:
    st.markdown('<div class="section-header">📹 Acquisition — Camera + ROI</div>', unsafe_allow_html=True)

    col_cam, col_roi = st.columns([3, 2], gap="large")

    with col_cam:
        video_placeholder = st.empty()

    with col_roi:
        roi_placeholder = st.empty()

    if st.session_state.session_active:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            video_placeholder.error("⚠️ Cannot open webcam. Check camera permissions.")
        else:
            frame_count = 0
            while st.session_state.session_active:
                ret, frame = cap.read()
                if not ret:
                    video_placeholder.warning("Lost webcam feed.")
                    break

                frame_count += 1
                if frame_count % FRAME_SKIP == 0:
                    annotated = pipeline.process_frame(frame)
                else:
                    annotated = frame

                video_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    width='stretch',
                )

                # ── Right Column: ROI Info + Guidance ──
                elapsed = time.time() - st.session_state.start_time
                roi_html = f"""
                <div class="metric-card" style="margin-bottom:10px;">
                    <div class="metric-label">Skin Signal Extraction</div>
                    <div style="text-align:left; margin-top:8px;">"""

                if pipeline.roi_details:
                    for name, d in pipeline.roi_details.items():
                        roi_html += f"""
                        <div style="color:#ccd6f6;font-size:0.8rem;padding:3px 0;">
                            <span style="color:{'#ff6bff' if 'forehead' in name else '#ffdd57'}">{name.replace('_',' ').title()}</span>:
                            R={d['r']:.0f} G={d['g']:.0f} B={d['b']:.0f}
                            <span style="color:#8892b0;">({d['pixel_count']}px)</span>
                        </div>"""
                else:
                    roi_html += '<div style="color:#8892b0;font-size:0.8rem;">Waiting for ROI...</div>'

                roi_html += f"""
                    </div>
                    <div style="color:#8892b0;font-size:0.75rem;margin-top:8px;border-top:1px solid rgba(255,255,255,0.06);padding-top:8px;">
                        FPS: {pipeline.fps:.1f} &nbsp;|&nbsp; Brightness: {pipeline.brightness:.0f} &nbsp;|&nbsp; Buffer: {pipeline.signal_buffer.get_size()}
                    </div>
                </div>
                <div class="guidance">
                    <div class="metric-label" style="margin-bottom:6px;">User Guidance</div>
                    <div class="guidance-item"><span>🎯</span> Center your face in the frame</div>
                    <div class="guidance-item"><span>🚫</span> Avoid sudden movements</div>
                    <div class="guidance-item"><span>💡</span> Ensure even lighting on face</div>
                    <div class="guidance-item"><span>⏱️</span> Elapsed: {format_time(elapsed)}</div>
                </div>"""
                roi_placeholder.markdown(roi_html, unsafe_allow_html=True)

                time.sleep(0.03)
            cap.release()

    elif not st.session_state.session_summary:
        video_placeholder.markdown("""
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                    border-radius:12px;padding:3rem;text-align:center;">
            <div style="font-size:4rem;margin-bottom:1rem;">📹</div>
            <div style="color:#ccd6f6;font-size:1rem;">Camera feed will appear here</div>
            <div style="color:#8892b0;font-size:0.85rem;margin-top:8px;">
                Click <strong>Start Session</strong> to begin
            </div>
        </div>
        """, unsafe_allow_html=True)
        roi_placeholder.markdown("""
        <div class="guidance">
            <div class="metric-label" style="margin-bottom:6px;">Quick Start</div>
            <div class="guidance-item"><span>1️⃣</span> Click Start Session</div>
            <div class="guidance-item"><span>2️⃣</span> Stay still for 60+ seconds</div>
            <div class="guidance-item"><span>3️⃣</span> Click End Session for results</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  PANELS 3–5 — Live Metrics (only during active session)
# ═══════════════════════════════════════════════════════════════════

if st.session_state.session_active:

    # ── PANEL 3: Signal Intelligence ──
    st.markdown('<div class="section-header">📡 Signal Intelligence</div>', unsafe_allow_html=True)

    si1, si2, si3, si4 = st.columns(4)

    with si1:
        qc = quality_color(pipeline.confidence)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Signal Quality</div>
            <div class="metric-value" style="color:{qc};font-size:1.8rem;">{pipeline.confidence*100:.0f}%</div>
            <div class="metric-unit">{pipeline.quality_label.upper()}</div>
            <div class="quality-bar"><div class="quality-fill" style="width:{pipeline.confidence*100}%;background:{qc};"></div></div>
        </div>""", unsafe_allow_html=True)

    with si2:
        ms = pipeline.motion_score
        mc = "#2ecc71" if ms > 0.7 else "#f39c12" if ms > 0.3 else "#e74c3c"
        ml = "Stable" if ms > 0.7 else "Minor Movement" if ms > 0.3 else "Excessive"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Motion Score</div>
            <div class="metric-value" style="color:{mc};font-size:1.8rem;">{ms*100:.0f}%</div>
            <div class="metric-unit">{ml}</div>
            <div class="quality-bar"><div class="quality-fill" style="width:{ms*100}%;background:{mc};"></div></div>
        </div>""", unsafe_allow_html=True)

    with si3:
        ls = pipeline.lighting_score
        lc = "#2ecc71" if ls > 0.7 else "#f39c12" if ls > 0.3 else "#e74c3c"
        ll = "Optimal" if ls > 0.7 else "Acceptable" if ls > 0.3 else "Poor"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Lighting Score</div>
            <div class="metric-value" style="color:{lc};font-size:1.8rem;">{ls*100:.0f}%</div>
            <div class="metric-unit">{ll}</div>
            <div class="quality-bar"><div class="quality-fill" style="width:{ls*100}%;background:{lc};"></div></div>
        </div>""", unsafe_allow_html=True)

    with si4:
        ss = pipeline.stability_score
        sc = "#2ecc71" if ss > 0.7 else "#f39c12" if ss > 0.3 else "#e74c3c"
        sl = f"Stable" if pipeline.hr_is_stable else "⚠ Stabilizing…"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Stability Score</div>
            <div class="metric-value" style="color:{sc};font-size:1.8rem;">{ss*100:.0f}%</div>
            <div class="metric-unit">{sl}</div>
            <div class="quality-bar"><div class="quality-fill" style="width:{ss*100}%;background:{sc};"></div></div>
        </div>""", unsafe_allow_html=True)

    # ── PANEL 4: Heart Rate Result ──
    st.markdown('<div class="section-header">❤️ Heart Rate Result</div>', unsafe_allow_html=True)

    hr_display = pipeline.stable_hr if pipeline.hr_is_stable else None
    hr_text = f"{hr_display:.0f}" if hr_display and hr_display > 0 else "--"

    hr_html = f"""
    <div class="hr-panel">"""

    if pipeline.hr_is_stable and hr_display:
        hr_html += f"""
        <div class="hr-big"><span class="pulse-icon">❤️</span> {hr_text}</div>
        <div class="hr-unit">BPM — Estimated Heart Rate</div>"""
    elif pipeline.buffer_ready:
        hr_html += f"""
        <div class="hr-stabilizing">⏳ Stabilizing signal…</div>
        <div style="color:#8892b0;font-size:0.85rem;margin-top:6px;">Hold still for accurate measurement</div>
        <div style="color:#ccd6f6;font-size:1.5rem;font-weight:700;margin-top:8px;">{pipeline.hr_bpm:.0f} BPM</div>
        <div style="color:#8892b0;font-size:0.7rem;">(raw estimate — not yet stable)</div>""" if pipeline.hr_bpm else """
        <div class="hr-stabilizing">⏳ Processing signal…</div>"""
    else:
        hr_html += """
        <div class="hr-stabilizing">📊 Collecting signal data…</div>
        <div style="color:#8892b0;font-size:0.85rem;margin-top:6px;">Building signal buffer</div>"""

    # Algorithm fusion transparency
    if pipeline.chrom_hr > 0 and pipeline.pos_hr > 0:
        hr_html += f"""
        <div class="algo-row">
            <div class="algo-item"><div class="algo-name">CHROM</div><div class="algo-val">{pipeline.chrom_hr:.1f}</div></div>
            <div class="algo-item"><div class="algo-name">POS</div><div class="algo-val">{pipeline.pos_hr:.1f}</div></div>
            <div class="algo-item"><div class="algo-name">Fused</div><div class="algo-val" style="color:#ff6b6b;">{pipeline.fused_hr:.1f}</div></div>
            <div class="algo-item"><div class="algo-name">SNR</div><div class="algo-val">{pipeline.snr:.1f}</div></div>
        </div>"""

    hr_html += "</div>"
    st.markdown(hr_html, unsafe_allow_html=True)

    # ── PANEL 5: Scientific Visualization ──
    if pipeline.buffer_ready and pipeline.last_green_signal is not None:
        st.markdown('<div class="section-header">🔬 Scientific Visualization</div>', unsafe_allow_html=True)

        viz1, viz2 = st.columns(2)

        with viz1:
            st.markdown("**rPPG Signal Waveform** (Green Channel)")
            g_sig = pipeline.last_green_signal
            if g_sig is not None and len(g_sig) > 0:
                g_detrended = g_sig - np.mean(g_sig)
                df_wave = pd.DataFrame({"Signal": g_detrended})
                st.line_chart(df_wave, height=200, width='stretch')

        with viz2:
            st.markdown("**Frequency Spectrum** (FFT Power)")
            if pipeline.last_freqs is not None and pipeline.last_psd is not None:
                freqs = pipeline.last_freqs
                psd = pipeline.last_psd
                # Show in BPM range 40-200
                mask = (freqs >= 0.5) & (freqs <= 3.5)
                if np.any(mask):
                    bpm_axis = freqs[mask] * 60
                    power = psd[mask]
                    df_fft = pd.DataFrame({"Power": power}, index=np.round(bpm_axis, 1))
                    df_fft.index.name = "BPM"
                    st.area_chart(df_fft, height=200, width='stretch')

        # Algorithm comparison
        if pipeline.last_chrom_signal is not None and pipeline.last_pos_signal is not None:
            st.markdown("**Algorithm Comparison** — CHROM vs POS")
            chrom_s = pipeline.last_chrom_signal
            pos_s = pipeline.last_pos_signal
            min_len = min(len(chrom_s), len(pos_s))
            df_algo = pd.DataFrame({
                "CHROM": chrom_s[:min_len] / (np.max(np.abs(chrom_s[:min_len])) + 1e-8),
                "POS": pos_s[:min_len] / (np.max(np.abs(pos_s[:min_len])) + 1e-8),
            })
            st.line_chart(df_algo, height=200, width='stretch')


# ═══════════════════════════════════════════════════════════════════
#  PANEL 6 — Session Analytics (post-session)
# ═══════════════════════════════════════════════════════════════════

if st.session_state.session_summary and not st.session_state.session_active:
    summary = st.session_state.session_summary
    avg_hr = summary.get("average_hr", 0)
    median_hr = summary.get("median_hr", 0)
    min_hr = summary.get("min_hr", 0)
    max_hr = summary.get("max_hr", 0)
    std_hr = summary.get("std_hr", 0)
    confidence = summary.get("confidence_score", 0)
    stability = summary.get("signal_stability", 0)
    duration = summary.get("duration_seconds", 0)
    total_frames = summary.get("total_frames", 0)
    valid_frames = summary.get("valid_frames", 0)
    hr_history = summary.get("hr_history", [])

    st.markdown('<div class="section-header">📊 Session Analytics</div>', unsafe_allow_html=True)

    # ── Main HR Result ──
    st.markdown(f"""
    <div class="hr-panel" style="margin-bottom:1rem;">
        <div class="hr-big">❤️ {avg_hr:.0f}</div>
        <div class="hr-unit">Average Heart Rate (BPM)</div>
    </div>""", unsafe_allow_html=True)

    # ── Heart Rate Statistics ──
    st.markdown('<div class="summary-card"><div class="summary-title">Heart Rate Statistics</div>', unsafe_allow_html=True)
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        st.markdown(f'<div class="summary-metric"><div class="value" style="color:#ff6b6b;">{avg_hr:.1f}</div><div class="label">Average BPM</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="summary-metric"><div class="value" style="color:#a29bfe;">{median_hr:.1f}</div><div class="label">Median BPM</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="summary-metric"><div class="value" style="color:#2ecc71;">{min_hr:.1f}</div><div class="label">Min BPM</div></div>', unsafe_allow_html=True)
    with s4:
        st.markdown(f'<div class="summary-metric"><div class="value" style="color:#e74c3c;">{max_hr:.1f}</div><div class="label">Max BPM</div></div>', unsafe_allow_html=True)
    with s5:
        st.markdown(f'<div class="summary-metric"><div class="value" style="color:#ccd6f6;">±{std_hr:.1f}</div><div class="label">Std Dev</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Signal Quality Summary ──
    st.markdown('<div class="summary-card"><div class="summary-title">Signal Quality Summary</div>', unsafe_allow_html=True)
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.markdown(f'<div class="summary-metric"><div class="value" style="color:{quality_color(confidence)};">{confidence*100:.0f}%</div><div class="label">Confidence</div></div>', unsafe_allow_html=True)
    with q2:
        st.markdown(f'<div class="summary-metric"><div class="value" style="color:{quality_color(stability)};">{stability*100:.0f}%</div><div class="label">Stability</div></div>', unsafe_allow_html=True)
    with q3:
        st.markdown(f'<div class="summary-metric"><div class="value timer-value" style="font-size:1.5rem;">{format_time(duration)}</div><div class="label">Duration</div></div>', unsafe_allow_html=True)
    with q4:
        pct = (valid_frames / total_frames * 100) if total_frames > 0 else 0
        st.markdown(f'<div class="summary-metric"><div class="value" style="color:#ccd6f6;font-size:1.5rem;">{pct:.0f}%</div><div class="label">Valid Frames</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── HR History Chart ──
    if hr_history and len(hr_history) > 2:
        st.markdown('<div class="summary-card"><div class="summary-title">Heart Rate Over Time</div>', unsafe_allow_html=True)
        df_hr = pd.DataFrame({"Heart Rate (BPM)": hr_history})
        st.line_chart(df_hr, height=250, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Export Options ──
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        if hr_history:
            csv_data = pd.DataFrame({
                "sample": range(1, len(hr_history) + 1),
                "hr_bpm": hr_history,
            }).to_csv(index=False)
            st.download_button(
                "📥 Download HR Data (CSV)",
                data=csv_data,
                file_name="rppg_session_hr.csv",
                mime="text/csv",
                width='stretch',
            )
    with ex2:
        report = f"""rPPG Session Report
{'='*40}
Duration: {format_time(duration)}
Average HR: {avg_hr:.1f} BPM
Median HR: {median_hr:.1f} BPM
Min HR: {min_hr:.1f} BPM
Max HR: {max_hr:.1f} BPM
Std Dev: ±{std_hr:.1f} BPM
Confidence: {confidence*100:.0f}%
Stability: {stability*100:.0f}%
Valid Frames: {pct:.0f}%
"""
        st.download_button(
            "📄 Download Report (TXT)",
            data=report,
            file_name="rppg_session_report.txt",
            mime="text/plain",
            width='stretch',
        )
    with ex3:
        if st.button("🔄 New Session", width='stretch'):
            pipeline.reset()
            st.session_state.session_active = False
            st.session_state.session_summary = None
            st.rerun()
