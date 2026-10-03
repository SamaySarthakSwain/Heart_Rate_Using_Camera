import logging
import os
import base64
import json
import time

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from our_project.api.pipeline import RealTimePipeline
from our_project.rppg.consistency import PhysiologicalConsistencyChecker

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Device-Agnostic AI/rPPG Dashboard",
    description="Real-time heart rate estimation via rPPG from a webcam stream.",
    version="1.0.0",
)

STATIC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "website"
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def get_dashboard():
    """Serve the single-page frontend."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health", tags=["ops"])
async def health_check():
    """Render health-check endpoint — must return 200 for the service to be marked healthy."""
    return JSONResponse({"status": "ok"})


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected: %s", websocket.client)

    pipeline = RealTimePipeline(method="chrom", buffer_size=150)
    consistency_checker = PhysiologicalConsistencyChecker()

    try:
        while True:
            data = await websocket.receive_text()

            # Expect a base64-encoded JPEG data-URL
            if not data.startswith("data:image"):
                continue

            try:
                encoded_data = data.split(",", 1)[1]
                nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as decode_err:
                logger.warning("Frame decode error: %s", decode_err)
                continue

            if frame is None:
                continue

            current_time = time.time()
            hr, sqi, status, raw_wave = pipeline.process_frame(frame, current_time)

            if hr is not None:
                is_valid, final_hr, reason = consistency_checker.validate(
                    hr, current_time, sqi
                )
                if not is_valid:
                    final_hr = None
            else:
                final_hr = None
                reason = status

            # Safe motion score extraction
            motion_score = 0.0
            try:
                history = pipeline.fusion.motion_analyzer.landmark_history
                if history:
                    motion_score = round(float(history[-1][0][0]), 2)
            except Exception:
                pass

            response = {
                "hr": round(final_hr, 1) if final_hr is not None else "--",
                "sqi": round(sqi, 2) if hr is not None else 0.0,
                "status": reason,
                "wave_value": float(raw_wave),
                "motion_score": motion_score,
            }
            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected: %s", websocket.client)
    except Exception as exc:
        logger.exception("Unexpected WebSocket error: %s", exc)


# ---------------------------------------------------------------------------
# Entrypoint (local dev only — Render uses the CMD in the Dockerfile)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info("Starting rPPG Dashboard on port %d", port)
    uvicorn.run(
        "our_project.api.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
