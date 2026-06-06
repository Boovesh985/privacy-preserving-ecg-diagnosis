#  Privacy-Preserving ECG Diagnosis with Homomorphic Encryption & Explainable AI

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)

> **End-to-end privacy-preserving ECG arrhythmia classification** using CKKS Homomorphic Encryption and SHAP-based explainability — with novel cross-dataset generalization analysis.

Extends: *Cenitta et al., "Explainable AI With Homomorphic Encryption for Secure Cloud-Based ECG Analysis in Heart Disease Diagnosis," IEEE Access 2025* ([DOI: 10.1109/ACCESS.2025.3614655](https://doi.org/10.1109/ACCESS.2025.3614655))

---

##  Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Novel Contributions](#novel-contributions)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Live Demo](#live-demo)
- [License](#license)

---

## Overview

This project implements a **secure, cloud-based ECG diagnostic framework** that enables heart disease classification on encrypted physiological data — the cloud never sees the patient's raw signal.

### Key Capabilities

| Capability | Description |
|---|---|
| **1D-CNN Classifier** | 5-class arrhythmia detection (N, S, V, F, Q) achieving **95.66% accuracy** on MIT-BIH |
| **CKKS Encryption** | Full homomorphic encryption pipeline using TenSEAL (Ring-LWE lattice cryptography) |
| **Encrypted Inference** | CNN inference via matrix-fused convolution operations on ciphertext |
| **SHAP Explainability** | 45-dimensional clinical feature attribution using DeepExplainer / KernelExplainer |
| **Interactive Dashboard** | Streamlit app with 4-step pipeline walkthrough (Plaintext → Encrypt → Infer → Explain) |

---

## Architecture

```
┌─────────────────────┐     ┌──────────────────────────┐     ┌──────────────────────┐
│    CLIENT DEVICE     │     │      CLOUD SERVER         │     │   CLINICIAN SIDE     │
│                      │     │                           │     │                      │
│  Raw ECG (187 pts)   │────▶│  Encrypted CNN Inference  │────▶│  Decrypt + Softmax   │
│  CKKS Encrypt        │     │  (W_conv × ciphertext)    │     │  SHAP Explanation    │
│  (128-bit security)  │     │  No decryption needed     │     │  Clinical Decision   │
└─────────────────────┘     └──────────────────────────┘     └──────────────────────┘
```

**Homomorphic Pipeline:**
1. **Client encrypts** ECG waveform → CKKS ciphertext (~340 KB payload)
2. **Cloud computes** 3-layer 1D-CNN via fused matrix multiplication on ciphertext
3. **Client decrypts** logits → softmax probabilities → arrhythmia class
4. **SHAP explains** prediction using 45 clinical ECG features (P-wave, QRS, ST-segment, RR intervals, DWT coefficients)

---

## Novel Contributions

This work introduces **4 novel contributions** absent from all prior HE-ECG literature:

| Claim | What We Measure | Key Finding |
|---|---|---|
| **C1** | SHAP rank-correlation: MIT-BIH → PTB-XL | Mean Spearman ρ = 0.365 |
| **C2** | Risk-adaptive CKKS tiers by arrhythmia class | 3 tiers: 80 / 128 / 192-bit |
| **C3** | SHAP reliability under wearable SNR noise | Stable down to SNR ≥ 5 dB (ρ > 0.96) |
| **C4** | Cross-dataset encrypted accuracy | Domain shift gap: +74.56% |

### C1 — Cross-Dataset SHAP Stability
First paper to measure SHAP feature rank-correlation consistency under ECG domain shift (clinical → wearable).

### C2 — Risk-Adaptive CKKS Encryption
First class-conditional CKKS parameter selection — Normal beats use lightweight 80-bit encryption for fast screening, while life-critical Ventricular arrhythmias use 192-bit protection.

### C3 — Noise Robustness Threshold
First SHAP reliability threshold for wearable ECG deployment — explanations remain reliable (ρ ≥ 0.96) even at SNR = 5 dB.

### C4 — Cross-Dataset Encrypted Inference
First paper to report CKKS-encrypted inference accuracy across two independent ECG datasets, quantifying the domain shift challenge.

---

## Project Structure

```
privacy-preserving-ecg-diagnosis/
│
├── phase-1-95acu/                    # Core research code
│   ├── app.py                        # Local Streamlit dashboard (full HE inference)
│   ├── utils.py                      # CKKS context, matrix operations, SHAP utilities
│   ├── precompute_matrices.py        # Pre-compute HE weight matrices for deployment
│   ├── train_shap_model.py           # Train 45-feature CNN proxy for SHAP
│   ├── export_matrices.py            # Export HE matrices to pickle
│   ├── feature_names.json            # 45 clinical ECG feature labels
│   ├── requirements.txt              # Python dependencies
│   │
│   ├── phase-1-95.ipynb              # Phase 1: 1D-CNN training & evaluation
│   ├── phase2-paper-faithful.ipynb   # Phase 2: Paper-faithful CKKS implementation
│   ├── phase-3-fixed.ipynb           # Phase 3: SHAP integration & dashboard
│   ├── phase4_novel_ptbxl_shap_adaptive_ckks.ipynb  # Phase 4: Novel contributions
│   ├── notebook33cd81492c.ipynb      # Supplementary analysis notebook
│   ├── notebookb28e144f78.ipynb      # Supplementary analysis notebook
│   │
│   ├── phase1_1dcnn_final.keras      # Trained 1D-CNN model (5-class, 95.66% acc)
│   ├── phase1_best.keras             # Best checkpoint during training
│   ├── cnn_feat_best.keras           # 45-feature proxy CNN for SHAP
│   ├── scaler_feat.pkl               # Feature scaler (sklearn)
│   ├── scaler_raw.pkl                # Raw signal scaler
│   │
│   ├── X_test_cnn.npy                # Test set: raw ECG waveforms (187 samples)
│   ├── X_test_feat_scaled.npy        # Test set: 45-dim scaled features
│   ├── y_test.npy                    # Test set: ground truth labels
│   ├── shap_background_feat.npy      # SHAP background distribution
│   │
│   ├── phase1_confusion_matrix.png   # Confusion matrix visualization
│   ├── phase1_dataset_overview.png   # Dataset class distribution
│   ├── phase1_training_curves.png    # Training/validation curves
│   │
│   ├── test_he.py                    # HE matrix extraction test
│   ├── test_matrix.py                # Matrix operation unit test
│   ├── test_ts.py                    # TenSEAL round-trip test
│   └── read_pdf.py                   # PDF text extraction utility
│
├── hf-deploy/                        # Hugging Face Spaces deployment
│   ├── app.py                        # Streamlit app (optimized for free-tier)
│   ├── utils.py                      # Deployment utilities (demo-mode inference)
│   ├── requirements.txt              # Pinned dependencies for HF Spaces
│   ├── packages.txt                  # System-level dependencies (cmake, g++)
│   ├── README.md                     # HF Spaces metadata card
│   ├── .gitattributes                # Git LFS tracking for HF
│   ├── upload_to_hf.py              # HF Hub API upload script
│   ├── deploy_to_hf.bat             # Windows deployment batch script
│   ├── he_matrices.npz              # Pre-computed HE weight matrices
│   ├── phase4_novel_results.json    # Phase 4 quantitative results
│   ├── figure1_shap_stability_cross_dataset.png
│   ├── figure2_snr_shap_stability.png
│   └── figure3_adaptive_ckks_latency.png
│
├── Explainable_AI_With_Homomorphic_Encryption_for_Secure_Cloud-Based_ECG_Analysis_in_Heart_Disease_Diagnosis.pdf
│
├── README.md                         # This file
├── LICENSE                           # MIT License
└── .gitignore                        # Git ignore rules
```

---

## Datasets

| Dataset | Source | Samples | Classes |
|---|---|---|---|
| **MIT-BIH Arrhythmia** | [PhysioNet](https://physionet.org/content/mitdb/1.0.0/) | 109,446 | N, S, V, F, Q |
| **PTB-XL** | [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/) | 21,837 | Mapped to 5-class |

- **MIT-BIH** is used as the primary training/testing dataset (AAMI standard 5-class split)
- **PTB-XL** is used for cross-dataset domain shift analysis (Phase 4 novelty)

---

## Installation

### Prerequisites
- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/Boovesh985/privacy-preserving-ecg-diagnosis.git
cd privacy-preserving-ecg-diagnosis

# Install dependencies
pip install -r phase-1-95acu/requirements.txt
```

### System Dependencies (for TenSEAL)

**Linux / macOS:**
```bash
sudo apt install cmake g++ libgmp-dev libprotobuf-dev protobuf-compiler
```

**Windows:**
TenSEAL wheels include pre-built binaries — `pip install tenseal` should work directly.

---

## Usage

### Run the Local Dashboard (Full HE Inference)

```bash
cd phase-1-95acu
streamlit run app.py
```

This runs the complete pipeline with **real CKKS encrypted inference** — including homomorphic matrix-vector multiplication on ciphertext.

### Run Individual Components

```bash
# Pre-compute HE weight matrices
python phase-1-95acu/precompute_matrices.py

# Train the SHAP feature proxy model
python phase-1-95acu/train_shap_model.py

# Test TenSEAL round-trip encryption
python phase-1-95acu/test_ts.py

# Test HE matrix extraction
python phase-1-95acu/test_he.py
```

### Jupyter Notebooks

Open the notebooks in order to follow the full research pipeline:

1. `phase-1-95.ipynb` — 1D-CNN model training and evaluation
2. `phase2-paper-faithful.ipynb` — Paper-faithful CKKS implementation
3. `phase-3-fixed.ipynb` — SHAP integration and dashboard development
4. `phase4_novel_ptbxl_shap_adaptive_ckks.ipynb` — Novel contributions (cross-dataset, adaptive CKKS)

---

## Results

### Phase 1 — 1D-CNN Classification

| Metric | Value |
|---|---|
| **Overall Accuracy** | 95.66% |
| **Architecture** | 3× Conv1D + AvgPool + Dense |
| **Input** | 187-sample normalized ECG waveform |
| **Classes** | Normal, Supraventricular, Ventricular, Fusion, Unknown |

### CKKS Encryption Performance

| Metric | Value |
|---|---|
| **Encryption Scheme** | CKKS (Cheon-Kim-Kim-Song) |
| **Security Level** | 128-bit (RLWE) |
| **Poly Modulus Degree** | 8192 |
| **Ciphertext Size** | ~340 KB |
| **Total Latency** | ~420 ms (encrypt + infer + decrypt) |

### SHAP Explainability

The framework decomposes predictions into **45 clinical ECG features**, including:
- Morphological: P-wave amplitude, QRS duration, ST-segment mean, T-wave slope
- Temporal: RR interval, RR variability, QT/QTc intervals
- Statistical: Mean, variance, skewness, kurtosis, energy, entropy
- Spectral: 20 Discrete Wavelet Transform (DWT) coefficients

---

## Live Demo

The application is deployed on Hugging Face Spaces:

🔗 **[Launch Live Demo](https://huggingface.co/spaces/Boovesh985/secure-ecg-analysis)**

> *Note: The HF Spaces version uses plaintext CNN inference for speed on free-tier hardware, while encryption/decryption operations are real CKKS.*

---

## Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | TensorFlow / Keras |
| Homomorphic Encryption | TenSEAL (CKKS via Microsoft SEAL) |
| Explainability | SHAP (DeepExplainer + KernelExplainer) |
| Dashboard | Streamlit |
| Data Processing | NumPy, scikit-learn |
| Visualization | Matplotlib |

---

## Citation

If you use this work, please cite the base paper:

```bibtex
@article{cenitta2025explainable,
  title={Explainable AI With Homomorphic Encryption for Secure Cloud-Based ECG Analysis in Heart Disease Diagnosis},
  author={Cenitta, D. and others},
  journal={IEEE Access},
  year={2025},
  doi={10.1109/ACCESS.2025.3614655}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
