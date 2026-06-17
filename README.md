````markdown
# SLIMIA-IPP: Inverse Protocol Prediction from Spheroid Microscopy Imaging

Official implementation of the paper:

**Inverse Protocol Prediction from Spheroid Microscopy Imaging via Morphology-Aware Structured Learning**

**Authors:** Prateek Mittal, Ayush Srivastava, and Joohi Chauhan  
**Affiliation:** Vision Exploration and Data Analytics (VEDAs) Lab, Department of Computer Science and Engineering, Motilal Nehru National Institute of Technology Allahabad

---

## Overview

Microscopy images are traditionally used to measure experimental outcomes such as spheroid size, morphology, and viability. However, an important inverse problem remains largely unexplored: can a microscopy image reveal the experimental protocol that generated it?

This repository introduces **Inverse Protocol Prediction (IPP)**, a structured multi-label learning task that aims to infer experimental culture conditions directly from a single bright-field spheroid image.

The proposed framework combines automated spheroid segmentation, morphometric feature extraction, multimodal representation learning, and dependency-aware prediction to recover experimental protocol attributes including cell line, culture medium, formation method, seeding density, timepoint, microscope, and magnification.

In addition to protocol reconstruction, the repository includes experiments on domain adaptation, model interpretability, temporal morphology forecasting, and cross-dataset generalization.

---

## Motivation

Large-scale biological imaging studies often suffer from incomplete, inconsistent, or missing metadata. Experimental conditions may be incorrectly recorded, lost during data transfer, or inconsistently reported across laboratories.

Inverse Protocol Prediction addresses this challenge by treating microscopy images as signatures of their underlying experimental conditions.

Potential applications include:

- Experimental reproducibility auditing
- Metadata validation and quality control
- Detection of protocol inconsistencies
- Dataset curation and annotation verification
- Automated laboratory workflow validation
- Morphology-driven biological analysis

---

## Method Overview

The complete framework consists of four major stages:

### 1. Spheroid Segmentation

Deep learning segmentation models generate accurate spheroid masks from bright-field microscopy images.

Implemented architectures include:

- U-Net++
- Attention U-Net
- DeepLabV3
- DeepLabV3+
- SegNet
- RefineNet
- Swin-UNet
- TransUNet

### 2. Morphometric Feature Extraction

Morphological descriptors are extracted from predicted segmentation masks, including:

- Area
- Perimeter
- Circularity
- Solidity
- Eccentricity
- Convexity
- Major Axis Length
- Minor Axis Length
- Compactness

### 3. Multimodal Feature Learning

Morphological descriptors are fused with deep visual representations extracted from image encoders.

Implemented architectures include:

- ConvNeXt-Tiny
- ViT-B/16
- CoAtNet
- Image–Shape Fusion Transformer
- Hierarchical Multi-Task Transformer (HMTT)

### 4. Inverse Protocol Prediction

A structured prediction framework reconstructs experimental protocol attributes from learned image representations.

Predicted attributes include:

- Cell Line
- Culture Medium
- Formation Method
- Seeding Density
- Timepoint
- Biological Replicate
- Microscope
- Magnification

---

## Repository Structure

```text
SLIMIA-IPP/
│
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── .gitignore
│
├── data/
│   ├── README.md
│   └── dataset_links.txt
│
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_segmentation_training.ipynb
│   ├── 03_morphometry_extraction.ipynb
│   ├── 04_ipp_all_models.ipynb
│   ├── 07_ablation_study.ipynb
│   ├── 08_domain_adversarial.ipynb
│   ├── 09_gradcam_analysis.ipynb
│   ├── 10_temporal_prediction.ipynb
│   └── 11_cross_dataset_validation.ipynb
│
├── src/
│   ├── segmentation/
│   ├── morphometry/
│   ├── ipp/
│   ├── temporal/
│   ├── domain_adaptation/
│   └── utils/
│
├── configs/
├── checkpoints/
├── results/
├── paper/
└── docs/
````

---

## Main Results

### Segmentation Performance

| Model           | Dice Score | IoU    |
| --------------- | ---------- | ------ |
| RefineNet       | 0.9665     | 0.9437 |
| Attention U-Net | 0.9604     | 0.9361 |
| U-Net++         | 0.9582     | 0.9316 |
| TransUNet       | 0.9579     | 0.9260 |

### Inverse Protocol Prediction

| Model              | Accuracy |
| ------------------ | -------- |
| CoAtNet            | 98.04%   |
| ConvNeXt-Tiny      | 97.83%   |
| ViT-B/16           | 97.81%   |
| Fusion Transformer | 97.39%   |
| HMTT               | 97.07%   |

### Biological Protocol Recovery

Average Macro-F1 across biological protocol attributes:

**0.915**

### Cross-Dataset Validation

The framework was further evaluated on:

* RxRx1
* Cell Tracking Challenge (CTC)

to assess robustness under severe domain shifts and heterogeneous acquisition conditions.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/VEDAs-Lab/Inverse-Inference.git
cd SLIMIA-IPP
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Alternatively:

```bash
conda env create -f environment.yml
conda activate slimia-ipp
```

---

## Dataset

Experiments were conducted using the SLiMIA dataset.

Please obtain the dataset from the original authors and place it inside the `data/` directory.

Expected structure:

```text
data/
├── images/
├── masks/
├── metadata.csv
└── splits/
```

Additional dataset preparation instructions are provided in:

```text
data/dataset_links.txt
```

---

## Reproducing Experiments

### Train Segmentation Models

```bash
python src/segmentation/train.py --config configs/segmentation.yaml
```

### Extract Morphometric Features

```bash
python src/morphometry/feature_extractor.py
```

### Train IPP Models

```bash
python src/ipp/train.py --config configs/coatnet.yaml
```

### Run Domain-Adversarial Training

```bash
python src/domain_adaptation/dann_training.py
```

### Generate Grad-CAM Visualizations

```bash
python src/utils/gradcam.py
```

### Train Temporal Prediction Models

```bash
python src/temporal/train.py
```

---

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{mittal2026inverse,
  title={Inverse Protocol Prediction from Spheroid Microscopy Imaging via Morphology-Aware Structured Learning},
  author={Mittal, Prateek and Srivastava, Ayush and Chauhan, Joohi},
  journal={bioRxiv},
  pages={2026--03},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
```

---

## Acknowledgements

This work was conducted at the Vision Exploration and Data Analytics (VEDAs) Lab, Department of Computer Science and Engineering, Motilal Nehru National Institute of Technology Allahabad.

We thank the creators of the SLiMIA dataset and the broader computational microscopy community for supporting open and reproducible research.

---

## License

This repository is released under the MIT License. See the LICENSE file for additional details.

```
```
