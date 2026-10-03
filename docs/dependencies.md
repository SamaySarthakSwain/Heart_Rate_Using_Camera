# Dependency Analysis

This document outlines the dependencies found across the reference repositories and our proposed dependency stack.

## 1. rPPG-Toolbox
- `h5py==2.10.0`, `mat73==0.59`: Data loading for specific formats.
- `yacs==0.1.8`: Configuration management.
- `scipy==1.10.0`, `numpy==1.22.0`, `pandas==1.1.5`: Core numerics.
- `scikit_image==0.17.2`, `opencv_python==4.5.2.54`: Image processing.
- `matplotlib==3.1.2`, `tensorboardX==2.4.1`: Visualization.
- `PyYAML==6.0`, `scikit_learn==1.0.2`, `tqdm==4.66.3`: Utilities.
*(Note: Requires PyTorch, though not explicitly in `requirements.txt`, setup scripts handle it).*

## 2. rppg-heart-rate
- `numpy==1.26.4`, `scipy==1.13.1`, `pandas==2.2.3`: Numerics.
- `opencv-python==4.10.0.84`, `mediapipe==0.10.14`: Vision & Face tracking.
- `torch==2.4.1`, `torchvision==0.19.1`: Deep Learning.
- `onnx==1.16.2`, `onnxruntime==1.19.2`: Inference optimization.
- `fastapi==0.115.6`, `uvicorn==0.32.1`, `websockets==15.0.1`, `python-multipart==0.0.20`, `jinja2==3.1.4`: Web Dashboard.
- `matplotlib==3.9.2`: Visualization.

## 3. Remote-Photoplethysmography
- `fastapi==0.115.6`, `uvicorn[standard]==0.34.0`, `pydantic==2.10.4`, `requests==2.32.3`, `python-multipart==0.0.20`: Backend.
- `numpy==1.26.4`, `scipy==1.14.1`: Numerics.
- `mediapipe==0.10.21`, `opencv-python-headless==4.10.0.84`: Vision.
- `streamlit==1.41.1`, `streamlit-webrtc==0.47.9`, `av==13.1.0`: Frontend streaming.
- `pytest==8.3.4`: Testing.

---

## Proposed Dependency Stack for Our Project

We will use modern, stable versions of the following packages.

### Core Signal & Vision
- `numpy>=1.26.0`
- `scipy>=1.13.0`
- `opencv-python>=4.10.0`
- `mediapipe>=0.10.14` (crucial for facial landmark extraction)

### Machine Learning & Deep Learning
- `torch>=2.4.0`, `torchvision>=0.19.0` (for neural rPPG models)
- `scikit-learn>=1.4.0` (for classical ML regression/fusion)
- `xgboost>=2.0.0` (for ML-based fusion and feature evaluation)
- `onnx>=1.16.0`, `onnxruntime>=1.19.0` (for optimized inference)

### Web & API (Real-Time Deployment)
- `fastapi>=0.115.0`, `uvicorn[standard]>=0.32.0` (Backend orchestration)
- `websockets>=14.0` (Real-time data streaming)
*(Note: We will build a React/TS frontend instead of Streamlit for better UI/UX and client-side processing control as requested in Phase 27).*

### Data & Configuration
- `pandas>=2.2.0`
- `pyyaml>=6.0.1`

### Testing & Quality
- `pytest>=8.3.0`
