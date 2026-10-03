# Research Gap Analysis

This document identifies the existing state of rPPG implementations, what is partially explored, and potential novel areas for our research.

## ALREADY EXISTING
1. **Classical rPPG Implementations**: CHROM, POS, GREEN, and ICA are well-implemented across multiple repositories.
2. **Deep Learning Baselines**: Models like PhysNet, TS-CAN, DeepPhys are available in the rPPG-Toolbox.
3. **Basic Real-time Pipelines**: WebRTC and WebSocket streaming with live HR estimation exist (rppg-heart-rate, Remote-Photoplethysmography).
4. **MediaPipe ROI Extraction**: Extracting forehead and cheeks using MediaPipe 478-landmark mesh is a standard approach in real-time systems.
5. **Basic Filtering & FFT**: Bandpass filtering (0.7-3.0 Hz) and FFT for peak frequency detection.

## PARTIALLY EXPLORED
1. **Signal Quality Indexes (SQI)**: Repositories use basic SNR or stability gating, but a robust, multi-factor, experimentally validated SQI that adapts the pipeline is not fully realized.
2. **Motion Robustness**: "Remote-Photoplethysmography" has "Motion Outlier Rejection", and "rPPG-Toolbox" has motion-augmented training datasets. However, dynamic real-time motion compensation and ROI re-selection are limited.
3. **Cross-Subject Generalization**: "rppg-heart-rate" performs a Leave-One-Subject-Out (LOSO) cross-validation on 15 subjects, but it admits poor performance (MAE ~31 BPM) due to small dataset size.
4. **Timestamp Interpolation**: Used in "Remote-Photoplethysmography" to fix variable webcam frame rates, but not standard across all DL approaches.

## POTENTIALLY NOVEL
1. **Adaptive Multi-ROI Fusion based on Dynamic Quality**: Most systems average the forehead and cheeks. Dynamically weighting or selecting the *best* ROI per frame based on local SNR/illumination is a strong novel direction.
2. **Multi-Method rPPG Fusion**: Combining CHROM, POS, and DL representations dynamically (e.g., using attention mechanisms or confidence weighting) rather than just falling back to one.
3. **Cross-Device and Cross-Resolution Generalization**: Explicitly researching and quantifying performance drops across different camera qualities, framerates (e.g., 15 vs 24 vs 30 fps), and devices, and building an abstraction layer to normalize this.
4. **Physiological Consistency & Uncertainty Modeling**: Real-time uncertainty estimation that prevents unrealistic HR jumps using contextual waveform analysis rather than basic Kalman filtering or hard clamping.
5. **Illumination-Aware rPPG**: Detecting lighting changes and actively correcting the signal or routing to a low-light specific model/algorithm.
