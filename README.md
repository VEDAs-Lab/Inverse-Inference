# Inverse Protocol Prediction from Spheroid Microscopy Imaging via Morphology-Aware Structured Learning
## (Accepted at the 29th INTERNATIONAL CONFERENCE ON MEDICAL IMAGE COMPUTING AND COMPUTER ASSISTED INTERVENTION (MICCAI), 2026)

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

If you find this work useful in your research or you use any of the modules above, please cite the following papers:

```bibtex
@InProceedings{chauhan2026inverse,
  title={Inverse Protocol Prediction from Spheroid Microscopy Imaging via Morphology-Aware Structured Learning},
  author={Chauhan, Joohi and Mittal, Prateek and Srivastava, Ayush},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
  year = {2026},
  publisher = {Springer Nature Switzerland},
}
```

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

---
