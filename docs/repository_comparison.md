# Repository Comparison

| Feature | rPPG-Toolbox | rppg-heart-rate | Remote-Photoplethysmography |
|---------|---------------|---------------|---------------|
| **Face Detection** | Standard Bounding Box | MediaPipe | MediaPipe |
| **Face Tracking** | Dynamic/Median bounding box | MediaPipe Mesh | MediaPipe Mesh |
| **ROI** | Standard crop | Forehead + Cheeks | Forehead + Cheeks |
| **GREEN** | Yes | Yes | No |
| **CHROM** | Yes | Yes | Yes |
| **POS** | Yes | Yes | Yes |
| **ICA** | Yes | No | No |
| **Deep Learning** | Yes (Extensive) | Yes (PhysNet) | No |
| **PhysNet** | Yes | Yes | No |
| **TS-CAN** | Yes | No | No |
| **EfficientPhys** | Yes | No | No |
| **Motion Handling** | Motion Augmentation training | Basic (not heavily robust) | Advanced (Motion Outlier Rejection) |
| **Illumination Handling** | Not explicitly emphasized | Identified as limitation | Stability Gating |
| **Signal Quality** | Not explicitly emphasized | SNR-based confidence | Multi-factor (SNR, Motion, Stability) |
| **Dataset Support** | SCAMPS, UBFC, PURE, BP4D+, etc. | UBFC | Not explicitly listed |
| **Training** | Comprehensive YAML configs | Scripts for PhysNet LOSO | None / Not applicable |
| **Real-Time** | Offline/Batch | Yes (FastAPI + WebSockets) | Yes (Streamlit + WebRTC) |
| **Mobile Support** | Not directly | Dashboard accessible via browser| WebRTC browser support |
| **Web Support** | None | Yes (React/FastAPI) | Yes (Streamlit) |
| **Evaluation** | MAE, RMSE, Pearson | MAE, RMSE, Bland-Altman | Unit tests, script evaluation |
| **Limitations** | Heavy, non-real-time | Lighting/motion sensitivity | Lack of DL, No explicit license |
