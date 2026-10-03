# Device-Agnostic Real-Time rPPG System: Research & Architecture Report

## 1. Executive Summary
This report details the architectural design and theoretical justification for our device-agnostic, real-time remote photoplethysmography (rPPG) monitoring system. Built to operate efficiently on standard RGB cameras, the engine explicitly prioritizes robustness to motion and variable illumination without relying on heavily overfitted, dataset-specific deep learning end-to-end models. 

By utilizing a **Multi-Method Fusion** approach—blending Deep Learning feature extraction (PhysNet) with battle-tested deterministic classical signal processing (CHROM, POS)—the system achieves high generalizability across devices.

## 2. Architectural Decisions & Rationale

### 2.1 Camera Abstraction Layer
- **Decision:** Normalize all incoming video streams to a targeted 30 FPS and 640x480 resolution while actively monitoring timestamps.
- **Rationale:** Different webcams and smartphones suffer from variable frame rates and auto-exposure lag. If timestamp drift is not corrected, frequency-domain FFT peaks shift artificially, destroying the Heart Rate estimate.

### 2.2 Facial Landmark Stability
- **Decision:** Utilize MediaPipe FaceMesh (478 dense landmarks) instead of bounding-box trackers (like Haar Cascades or standard YOLO).
- **Rationale:** Bounding boxes jitter pixel-by-pixel frame-to-frame. Even a 2-pixel jitter in the bounding box creates a massive artificial spike in the spatial RGB mean, entirely overpowering the micro-color changes of the human pulse. MediaPipe's sub-pixel topological consistency solves this.

## 3. Handling of Motion and Illumination Artifacts

### 3.1 Adaptive Multi-ROI vs Single-ROI Averaging
- **Traditional Approach (Single ROI):** Averaging the entire bounding box of the face.
  - *Failure Mode:* Includes background pixels, hair, and shadows under the chin. Highly susceptible to talking/smiling artifacts.
- **Our Approach (Adaptive Multi-ROI):** We extract the forehead, left cheek, and right cheek independently.
  - *Motion Penalty:* When the user talks, the Euclidean displacement of the lower facial landmarks spikes. Our `AdaptiveROIFusion` module calculates a `motion_score` and dynamically drops the weight of the cheeks to near zero, relying solely on the stable forehead.
  - *Illumination Penalty:* If half the face is in shadow (luminance < 40), that specific ROI is dynamically suppressed.

## 4. Multi-Method Fusion and Uncertainty Estimation

Instead of forcing a single algorithm to be a silver bullet, our system computes HR across `CHROM`, `POS`, `GREEN`, and `PhysNet` simultaneously.
- **Signal Quality Index (SQI):** Each method's output is graded using frequency-domain Signal-to-Noise Ratio (SNR) in the human physiological band (0.7 - 3.0 Hz).
- **ML Fusion:** An XGBoost regressor (implemented in `method_fusion.py`) takes the HR and SQI of all methods, alongside the global motion/illumination scores, to predict the true HR.
- **Uncertainty Layer:** `uncertainty.py` calculates the weighted variance across the methods. If the algorithms drastically disagree (e.g., POS says 70 BPM, PhysNet says 120 BPM), the system outputs a low Confidence Percentage (< 50%) rather than a hallucinated average.

## 5. Physiological Consistency
- **Absolute Limits:** Hard boundaries at 42-180 BPM.
- **Temporal Continuity:** The human heart cannot jump from 60 BPM to 140 BPM in 1 second. The `PhysiologicalConsistencyChecker` rejects biologically implausible rate-of-change spikes, ensuring clinical safety in the readout.

## 6. Future Steps for Physical Dataset Collection
Because public datasets (UBFC, PURE) lack sufficient extreme-edge-case diversity, the next phase of this project must involve physical dataset collection:
1. **Cross-Device Matrix:** Collect synchronous video from a Macbook Webcam, a cheap $15 USB camera, an iPhone, and a cheap Android device to train the fusion model to ignore sensor-specific chromatic aberration.
2. **Skin Tone Uniformity:** Ensure the dataset perfectly balances the Fitzpatrick skin type scale (Types I-VI) to prevent melanin-bias in the DL models.
3. **Explicit Artifact Protocols:** Record subjects explicitly talking, eating, shivering, and transitioning from dark rooms to sunlight while wearing a ground-truth ECG/Pulse Oximeter.
