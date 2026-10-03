"""Session API routes for the rPPG backend."""

import base64
import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.session_manager import SessionManager


router = APIRouter(prefix="/api/session", tags=["session"])

# Global session manager instance
session_manager = SessionManager()


# --- Request/Response Models ---

class StartSessionResponse(BaseModel):
    session_id: str
    message: str


class FrameRequest(BaseModel):
    frame_base64: str  # Base64-encoded JPEG image


class FrameResponse(BaseModel):
    face_detected: bool
    buffer_ready: bool
    hr_bpm: float | None
    confidence: float
    quality_label: str
    motion_detected: bool
    motion_score: float
    elapsed_seconds: float
    total_frames: int


class SessionStatusResponse(BaseModel):
    session_id: str
    state: str
    elapsed_seconds: float
    total_frames: int
    valid_frames: int
    current_hr: float | None
    buffer_size: int
    buffer_ready: bool


class EndSessionResponse(BaseModel):
    session_id: str
    average_hr: float
    median_hr: float
    std_hr: float
    confidence_score: float
    signal_stability: float
    duration_seconds: float
    total_frames: int
    valid_frames: int


# --- Endpoints ---

@router.post("/start", response_model=StartSessionResponse)
async def start_session():
    """Start a new rPPG measurement session."""
    session = session_manager.create_session()
    return StartSessionResponse(
        session_id=session.session_id,
        message="Session started successfully",
    )


@router.post("/{session_id}/frame", response_model=FrameResponse)
async def process_frame(session_id: str, request: FrameRequest):
    """Process a single video frame.

    Accepts a base64-encoded JPEG image and returns real-time metrics.
    """
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.is_active:
        raise HTTPException(status_code=400, detail="Session has ended")

    # Decode base64 JPEG to numpy array
    try:
        img_bytes = base64.b64decode(request.frame_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid frame data: {str(e)}")

    # Process through pipeline
    result = session.process_frame(frame)

    return FrameResponse(**result)


@router.get("/{session_id}/status", response_model=SessionStatusResponse)
async def get_session_status(session_id: str):
    """Get current session status and metrics."""
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    status = session.get_status()
    return SessionStatusResponse(
        session_id=status["session_id"],
        state=status["state"],
        elapsed_seconds=status["elapsed_seconds"],
        total_frames=status["total_frames"],
        valid_frames=status["valid_frames"],
        current_hr=status["current_hr"],
        buffer_size=status["buffer_size"],
        buffer_ready=status["buffer_ready"],
    )


@router.post("/{session_id}/end", response_model=EndSessionResponse)
async def end_session(session_id: str):
    """End a session and return summary metrics."""
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.is_active:
        raise HTTPException(status_code=400, detail="Session already ended")

    summary = session_manager.end_session(session_id)
    if summary is None:
        raise HTTPException(status_code=500, detail="Failed to end session")

    return EndSessionResponse(
        session_id=session_id,
        average_hr=summary.average_hr,
        median_hr=summary.median_hr,
        std_hr=summary.std_hr,
        confidence_score=summary.confidence_score,
        signal_stability=summary.signal_stability,
        duration_seconds=summary.duration_seconds,
        total_frames=summary.total_frames,
        valid_frames=summary.valid_frames,
    )
