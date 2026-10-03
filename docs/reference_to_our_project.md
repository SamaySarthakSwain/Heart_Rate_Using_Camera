# Reference to Our Project Map

This document tracks how we will utilize the reference repositories and what our expected novel contribution will be.

| Reference Component | Source Repo | What it Does | Why it is Useful | How We Improve/Adapt It | Our Expected Contribution |
|---------------------|-------------|--------------|------------------|-------------------------|---------------------------|
| **Deep Learning Models** | rPPG-Toolbox | Provides PhysNet, TS-CAN architectures. | Established baselines for rPPG. | We will implement them cleanly without the heavy toolbox overhead, optimizing them for real-time inference (ONNX). | A lightweight, fast inference pipeline for these heavy models. |
| **MediaPipe FaceMesh** | rppg-heart-rate | Extracts forehead and cheeks using 478 landmarks. | More robust than Haar cascades; tracks face reliably. | Implement **Adaptive Multi-ROI**. Instead of static averaging, we will calculate SNR/motion per ROI and dynamically weight them. | Adaptive quality-weighted ROI fusion. |
| **CHROM / POS algos** | All three | Classical color-space projection to extract pulse. | Robust baselines that don't require training data. | Use as base estimators for our **Multi-Method Fusion**. | Dynamically selecting/weighting CHROM/POS/DL based on lighting/motion. |
| **Harmonic Rejection** | Remote-PPG | Prevents "2x HR" errors by validating FFT peaks. | Solves a common artifact where breathing/harmonics overpower the true HR. | We will implement our own physiological consistency module that considers temporal context and harmonics. | Robust physiological consistency that rejects false peaks. |
| **Timestamp Interpolation** | Remote-PPG | Normalizes variable webcam frame rates. | Essential for accurate FFT which assumes a constant sampling rate. | Build into our core **Camera Abstraction Layer** to ensure cross-device consistency. | Device-agnostic frame and timestamp normalization. |
| **Web Dashboard** | rppg-heart-rate | Streams video/data via FastAPI + WebSockets. | Good starting point for real-time visualization. | We will build a production-grade React frontend that handles local privacy constraints (processing on device where possible). | A professional, privacy-preserving full-stack web application. |
| **Confidence Scoring** | rppg-heart-rate | Provides SNR-based confidence. | Gives the user an idea of signal reliability. | Expand into a comprehensive **Signal Quality Index (SQI)** covering motion, illumination, and SNR. | Uncertainty-aware HR estimation with explicit error boundaries. |
