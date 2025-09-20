# LAM-Soft: Semi-Supervised Colonoscopy Depth Estimation via Lighting Adjustment Model-Driven Soft Labels

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

> **Project**: LAM-Soft: Semi-Supervised Colonoscopy Depth Estimation via Lighting Adjustment Model-Driven Soft Labels
> **Authors**: Xiheng Wu, Zhaolin Zhang, Zhe Xu, Entong Liu, Dongya Chen, Bishi He

Welcome to the official repository for **LAM-Soft**, the code implementation accompanying our cutting-edge research manuscript. LAM-Soft is a novel semi-supervised framework designed to tackle the critical challenges of monocular depth estimation in colonoscopy, specifically addressing **complex lighting variations** (e.g., mucus reflections, exposure inconsistencies) and the **synthetic-to-real domain gap**. Our goal is to enable highly accurate polyp size measurement, a crucial factor for reliable colorectal cancer (CRC) diagnosis.

By introducing a dynamic **Lighting Adjustment Model (LAM)** and **confidence-aware learning**, LAM-Soft achieves state-of-the-art performance on real clinical data. **The associated research paper is currently under preparation and will be submitted to a top-tier conference or journal in medical imaging or computer vision.**

## 🌟 Key Innovations

*   **Lighting Adjustment Model (LAM)**: A pioneering component that dynamically models near-field light attenuation and surface reflections. It transforms rigid pseudo-labels into illumination-sensitive soft constraints, effectively bridging the gap between synthetic data and real colonoscopy images.
*   **Confidence-Aware Semi-Supervised Learning**: Employs a teacher-student architecture that leverages hard labels from synthetic data and LAM-adjusted soft labels from clinical data. An integrated confidence decoder automatically filters out low-quality frames (e.g., motion-blurred or contaminated images), ensuring robust and noise-resistant training.
*   **Clinically Validated Superior Performance**: Achieves SOTA results on the C3VD dataset and reduces the mean absolute error for polyp size estimation on the SUN database by 47% compared to the PPSNet baseline.
*   **Lightweight & Real-Time**: Engineered for efficiency, the model supports real-time processing at up to 71 FPS, making it ideal for seamless integration into clinical colonoscopy workflows.

## 📊 Performance Benchmarks

### Depth Estimation Performance on C3VD Dataset

| Model | L1 Error (mm) | δ1.05 Accuracy |
| :--- | :--- | :--- |
| **LAM-Soft (Ours)** | **0.1789** | **57.19%** |
| PPSNet (Baseline) | 0.2158 | 48.31% |
| DepthAnything | 0.2254 | 45.81% |

### Polyp Size Estimation Performance on SUN Database

| Model | Mean Absolute Error (mm) | Improvement vs. PPSNet |
| :--- | :--- | :--- |
| **LAM-Soft (Ours)** | **0.819** | **-47%** |
| PPSNet (Baseline) | 1.558 | - |

## 🚀 Quick Start

### 1. Environment Setup

We recommend setting up a dedicated Python environment using `conda`.

```bash
# Clone the repository
git clone https://github.com/your_username/LAM-Soft.git
cd LAM-Soft

# Create and activate a conda environment
conda create -n lamsoft python=3.12
conda activate lamsoft

# Install PyTorch (adjust the CUDA version as needed)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install other dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

This project requires the following datasets:
*   **C3VD Dataset**: Synthetic dataset for supervised training.
*   **Clinical Colonoscopy Dataset**: Real clinical data for semi-supervised learning (you will need to prepare this yourself).
*   **SUN Colonoscopy Video Database**: For evaluating polyp size estimation performance.

Download and extract the datasets into the `datasets/` directory at the project root, structured as follows:
```
LAM-Soft/datasets
├── C3VD
│   ├── cecum_t1_a
│   ├── cecum_t1_b
│   ├── cecum_t2_a
│   ├── cecum_t2_b
│   ├── cecum_t2_c
│   ├── cecum_t3_a
│   ├── cecum_t4_a
│   ├── cecum_t4_b
│   ├── desc_t4_a
│   ├── desc_t4_a_p1
│   ├── desc_t4_a_p2
│   ├── seq1
│   ├── seq2
│   ├── seq3
│   ├── seq4
│   ├── sigmoid_t1_a
│   ├── sigmoid_t2_a
│   ├── sigmoid_t3_a
│   ├── sigmoid_t3_b
│   ├── trans_t1_a
│   ├── trans_t1_b
│   ├── trans_t2_a
│   ├── trans_t2_b
│   ├── trans_t2_c
│   ├── trans_t3_a
│   ├── trans_t3_b
│   ├── trans_t4_a
│   ├── trans_t4_b
│   └── cfhq190l_10x10mm_checkerboard_images
└── ClinicalData
    └── RawFrames
```



---

**Disclaimer**: This software is a research preview and is provided for academic and educational purposes only. **DO NOT** use it for clinical diagnosis or treatment without consulting qualified medical professionals and conducting thorough clinical validation.

---
