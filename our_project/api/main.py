from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import cv2
import numpy as np
import base64
import time
import json
import os

from our_project.api.pipeline import RealTimePipeline
from our_project.rppg.consistency import PhysiologicalConsistencyChecker

app = FastAPI(title="Device-Agnostic AI/rPPG Dashboard")

# Ensure static directory exists
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "website")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def get_dashboard():
    with open(os.path.join(STATIC_DIR, "index.html"), "r") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    pipeline = RealTimePipeline(method="chrom", buffer_size=150)
    consistency_checker = PhysiologicalConsistencyChecker()
    
    try:
        while True:
            # Receive frame from client
            data = await websocket.receive_text()
            
            # Decode base64 image
            if data.startswith("data:image"):
                encoded_data = data.split(',')[1]
                nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Process frame
                current_time = time.time()
                hr, sqi, status, raw_wave = pipeline.process_frame(frame, current_time)
                
                # Check consistency if we have a reading
                if hr is not None:
                    is_valid, final_hr, reason = consistency_checker.validate(hr, current_time, sqi)
                else:
                    final_hr = None
                    reason = status
                    
                # Send result back
                response = {
                    "hr": round(final_hr, 1) if final_hr else "--",
                    "sqi": round(sqi, 2) if hr else 0.0,
                    "status": reason,
                    "wave_value": float(raw_wave),
                    "motion_score": round(pipeline.fusion.motion_analyzer.landmark_history[-1][0][0] if len(pipeline.fusion.motion_analyzer.landmark_history) > 0 else 0, 2) # debug value
                }
                await websocket.send_text(json.dumps(response))
                
    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    print("Starting WebRTC Dashboard on http://localhost:8088")
    uvicorn.run("our_project.api.main:app", host="0.0.0.0", port=8088, reload=True)
