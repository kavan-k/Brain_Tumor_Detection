# Brain Tumor Detection & Segmentation
**University of Michigan-Dearborn | Advance AI Term Project**

This repository contains the implementation of a deep learning pipeline for brain tumor detection and segmentation using the **BraTS 2020 Dataset**.

## 🧠 Model Architecture
- **Preprocessing:** Normalization, resizing, and data augmentation via `torchvision`.
- **Core Model:** Implementation of High-Resolution Image Synthesis (AEGAN/SRGAN principles).
- **Framework:** PyTorch

## 📊 Dataset
The project utilizes the **BraTS 2020** (Brain Tumor Segmentation) dataset. Due to size constraints and licensing, the raw `.nii` data files are not included in this repository.
- **Data Source:** [MICCAI BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/)
- **Structure:** Expected in `/brats_data/` (ignored by Git).

## 🚀 How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
