# Datasets

This document details the publicly available datasets we intend to use for training and validation in our rPPG research project. 

Because many of these datasets contain sensitive physiological and biometric information, direct downloads are legally restricted. You must register and agree to the Data Use Agreements (DUAs) for each dataset to legally obtain them.

## 1. UBFC-rPPG
- **Subjects**: 42
- **Videos**: 42 (uncompressed, standard RGB)
- **FPS**: ~30 FPS
- **Resolution**: 640x480
- **Reference Signal**: CMS50E pulse oximeter (PPG waveform, Heart Rate, SpO2)
- **Lighting**: Indoor, relatively controlled, though some variation exists.
- **Motion**: Minimal. Subjects were asked to play a time-sensitive math game, inducing mild cognitive stress and slight natural movement.
- **Camera/Device**: Logitech C920 HD Pro webcam.
- **Acquisition Process**:
  - Visit the [UBFC-rPPG website](https://sites.google.com/view/ybenezeth/ubfcrppg) or contact the authors directly.
  - Sign the DUA acknowledging non-commercial research use.
- **Limitations**: Demographics are relatively narrow; lighting is mostly stable. It is not sufficient alone for testing extreme lighting or motion robustness.

## 2. SCAMPS (Synthetics for Camera Measurement of Physiological Signals)
- **Subjects**: ~2,800 synthetic subjects
- **Videos**: 2,800
- **FPS**: 30 FPS
- **Resolution**: 720x720
- **Reference Signal**: Synthetically generated underlying PPG and respiration waveforms.
- **Lighting**: Highly varied (simulated).
- **Motion**: Varied synthetic head motions and facial action units.
- **Acquisition Process**:
  - Available via [SCAMPS GitHub/Website](https://github.com/danmcduff/scampsdataset).
  - May require a DUA form submission to access the raw `.mat` files or MP4s.
- **Limitations**: It is synthetic. While excellent for pre-training deep models (PhysNet/TS-CAN) to learn generalized spatial features, models must still be fine-tuned or heavily evaluated on real human skin to ensure true clinical/physiological validity.

## 3. PURE (Pulse Rate Detection Dataset)
- **Subjects**: 10 (8 male, 2 female)
- **Videos**: 60 (6 different motion tasks per subject)
- **FPS**: 30 FPS
- **Resolution**: 640x480
- **Reference Signal**: Pulox PO-400 finger pulse oximeter (sampled at 60Hz).
- **Lighting**: Ambient indoor lighting (through window).
- **Motion**: Explicit motion tasks (steady, talking, slow translation, fast translation, small rotation, medium rotation).
- **Camera/Device**: eco274CVGE camera.
- **Acquisition Process**:
  - Contact the authors at TU Ilmenau. Information is [here](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure).
  - Submit the signed EULA.
- **Limitations**: Small subject pool (N=10), but highly valuable for evaluating the **Motion Robustness** module because of the explicit motion protocols.

## 4. MMPD (Multi-Domain Mobile Video Physiology Dataset)
- **Subjects**: 33
- **Videos**: 825
- **FPS**: 30 FPS
- **Resolution**: 320x240 (compressed for mobile)
- **Reference Signal**: HKG-07C+ pulse oximeter.
- **Lighting**: 4 lighting conditions (LED-low, LED-high, Incandescent, Natural).
- **Motion**: 3 motion states (Stationary, Head Rotation, Talking).
- **Camera/Device**: Samsung Galaxy S20 (Smartphone camera).
- **Acquisition Process**:
  - Follow the instructions on the [MMPD GitHub](https://github.com/McJackTang/MMPD_rPPG_dataset).
- **Limitations**: Low resolution, but highly critical for our **Cross-Device Generalization** (smartphone vs webcam) and explicitly varying lighting conditions.

## Experimental Protocol: No Data Leakage
To ensure strict scientific validity:
- We will use **Subject-Independent Cross-Validation**.
- For cross-dataset evaluation (e.g., Train on SCAMPS + UBFC, Test on PURE), the entirety of the test dataset will be completely unseen during the training and validation loops.
- The test set will never be used for hyperparameter tuning. Tuning will rely strictly on the validation set.
