# Repository Analysis

This document provides a deep analysis of the three reference repositories provided for the rPPG research project.

## 1. rPPG-Toolbox (BMI Lab, IIT Hyderabad / UbiCompLab)

### Overview
- **Architecture**: A comprehensive, modular research framework for both unsupervised and deep-learning rPPG methods. Built primarily for benchmarking and training, not real-time inference.
- **Python version / Dependencies**: Uses Python 3.x, heavily depends on PyTorch, OpenCV, SciPy, NumPy, and specific data loading libraries (`mat73`, `h5py`).
- **Configuration**: Uses YAML files (`configs/`) for defining experiments, datasets, and hyperparameters.
- **Data Pipeline**: Supports a wide array of public datasets (SCAMPS, UBFC, PURE, BP4D+, UBFC-Phys, MMPD). Includes robust preprocessing steps like face cropping, dynamic detection, data chunking, and generative data augmentation (motion augmentation).
- **Face/ROI**: Mentions face detection (`DYNAMIC_DETECTION`, `DO_CROP_FACE`), but primarily relies on standard bounding boxes rather than dense landmark meshes for ROI selection.
- **Algorithms**: 
  - Unsupervised: GREEN, ICA, CHROM, POS, LGI, PBV.
  - Neural: DeepPhys, PhysNet, TS-CAN, EfficientPhys, BigSmall.
- **Real-time Pipeline**: Weak/None. It provides inference scripts (`main.py --config_file`), but it's geared towards offline video processing.
- **Limitations**: Setup can be heavy. Focuses on evaluation/benchmarking rather than live deployment. 
- **License**: RAIL (Responsible AI License) - restricts commercial use and certain applications.

## 2. rppg-heart-rate (HarshTomar1234)

### Overview
- **Architecture**: A production-grade system split into classical processing and a deep-learning model (PhysNet). It includes a web dashboard (FastAPI backend + WebSockets).
- **Dependencies**: Python 3.10+, FastAPI, MediaPipe (for face detection), OpenCV, PyTorch, ONNX.
- **Data Pipeline**: Custom dataset loaders (e.g., `ubfc_loader.py`). Evaluated on UBFC-rPPG. Uses a Leave-One-Subject-Out (LOSO) cross-validation for PhysNet.
- **Face/ROI**: Uses MediaPipe Face Mesh (478 landmarks) for precise forehead and cheek ROI extraction.
- **Algorithms**:
  - Unsupervised: CHROM, POS, Green-channel, with an auto-selection mechanism.
  - Neural: PhysNet 3D-CNN (implemented, but not yet integrated into the live app).
- **Real-time Pipeline**: Strong. The `app/main.py` provides a live WebSocket-based dashboard. Includes signal quality assessment (SNR-based confidence).
- **Limitations**: Model only trained on 15 subjects from UBFC. High lighting and motion sensitivity. PhysNet isn't fully integrated into the live app yet.
- **License**: Apache 2.0.

## 3. Remote-Photoplethysmography (Aadik1ng)

### Overview
- **Architecture**: A focused, advanced real-time monitor using Streamlit for the frontend (WebRTC) and FastAPI for the backend.
- **Dependencies**: Streamlit, WebRTC, FastAPI, MediaPipe, SciPy.
- **Data Pipeline**: Not primarily geared towards massive model training; more focused on robust real-time signal processing.
- **Face/ROI**: Uses MediaPipe FaceMesh for stable ROI extraction (forehead and cheeks).
- **Algorithms**: 
  - Dual-Algorithm Fusion: CHROM and POS.
- **Real-time Pipeline**: Excellent. Uses "High-Fidelity Accuracy Stack" including Harmonic Rejection Logic, Stability Gating, Timestamp Interpolation, and Motion Outlier Rejection.
- **Limitations**: Heavily focused on classical methods; no deep learning models evident from the README. No explicit license found.
- **License**: None explicitly provided (assumed all rights reserved).
