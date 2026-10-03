"""FastAPI main application for the rPPG backend."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.session_routes import router as session_router

app = FastAPI(
    title="rPPG Heart Rate Monitor API",
    description="Real-time remote photoplethysmography backend using CHROM + POS algorithms",
    version="1.0.0",
)

# CORS middleware — allow Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(session_router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "rPPG Backend"}
