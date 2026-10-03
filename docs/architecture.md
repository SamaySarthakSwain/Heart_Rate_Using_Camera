# System Architecture and Implementation Plan

## Proposed Architecture

Our system will be designed to be highly modular, device-agnostic, and uncertainty-aware.

```mermaid
graph TD
    A[ANY RGB CAMERA] --> B[Camera Abstraction Layer]
    B --> C[Capability Detection]
    C --> D[Timestamp/Frame Normalization]
    
    D --> E[Face Detection & Landmarks]
    E --> F[Multi-ROI Extraction]
    
    F --> G[Motion Analysis]
    F --> H[Illumination Analysis]
    
    G --> I[Adaptive ROI Selection & Fusion]
    H --> I
    
    I --> J[CHROM / POS / GREEN Extraction]
    I --> K[DL Model Inference]
    
    J --> L[Signal Quality Index (SQI)]
    K --> L
    
    L --> M[Feature / Method Fusion ML]
    M --> N[Heart Rate Estimator]
    
    N --> O[Confidence & Uncertainty Analysis]
    O --> P[Physiological Consistency Check]
    
    P --> Q[Final Validated Result]
    Q --> R[Real-time Web Dashboard]
```

## Implementation Plan (Phases 1-35)

As requested, we will execute strictly in the defined order.

**CURRENT STATUS:**
- [x] STEP 1: Analyze all three repositories.
- [x] STEP 2: Analyze licenses.
- [x] STEP 3: Analyze dependencies.
- [x] STEP 4: Create architecture.
- [ ] STEP 5: Set up environment.
- [ ] STEP 6: Run reference repositories.
- [ ] STEP 7: Set up datasets.
- [ ] STEP 8: Implement standardized evaluation.
- [ ] STEP 9: Reproduce GREEN/CHROM/POS.
- ... (Continues up to STEP 30)

We are currently at **STEP 5**. I am waiting for your explicit confirmation on this analysis and architecture before we proceed to environment setup, directory scaffolding, and running the reference repos.
