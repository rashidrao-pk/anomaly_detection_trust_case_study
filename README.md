# 🦠 Can I Trust My Anomaly Detection System? A Case Study Based on Explainable AI

### Explainable and Trustworthy Anomaly Detection using VAE-GAN

<div align="center">
<a href="https://link.springer.com/chapter/10.1007/978-3-031-63803-9_13">
  <img src="https://img.shields.io/badge/Paper-Springer-red">
</a>
<img src="https://img.shields.io/badge/XAI%202024-Published-success">
<img src="https://img.shields.io/badge/version-v1.0.0-blue" alt="Version">
<a href="https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/blob/main/LICENSE">
<img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
</a>
<a href="https://github.com/rashidrao-pk/anomaly_detection_trust_case_study">
<img src="https://img.shields.io/github/stars/rashidrao-pk/anomaly_detection_trust_case_study?style=social" alt="GitHub Stars">
</a>
<a href="https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/watchers">
<img src="https://img.shields.io/github/watchers/rashidrao-pk/anomaly_detection_trust_case_study?style=flat" alt="GitHub Watchers">
</a>
<img src="https://img.shields.io/github/repo-size/rashidrao-pk/anomaly_detection_trust_case_study" alt="Repository Size">
<img src="https://img.shields.io/github/last-commit/rashidrao-pk/anomaly_detection_trust_case_study" alt="Last Commit">
</div>


## 📖 Overview

This repository contains the official implementation and experimental artifacts for the paper:

> **“Can I Trust My Anomaly Detection System? A Case Study Based on Explainable AI”**  
> Published at the **2nd World Conference on eXplainable Artificial Intelligence (XAI 2024)**  
> 📍 Valletta, Malta — July 17–19, 2024

📄 Paper Link:  
[Springer Chapter](https://link.springer.com/chapter/10.1007/978-3-031-63803-9_13)

🌐 Conference Website:  
[XAI World Conference 2024](https://xaiworldconference.com/2024/)

---
## ✨ Highlights

- ✅ VAE-GAN based anomaly detection
- ✅ Explainability using SHAP and LIME
- ✅ Trustworthiness evaluation using XAI
- ✅ Quantitative explanation validation
- ✅ Experiments on MVTec AD datasets
- ✅ Industrial inspection use case


## 🎯 Motivation

Industrial anomaly detection systems based on Deep Learning are increasingly deployed in safety-critical environments such as:

- 🏭 Industrial quality inspection  
- 🤖 Robotics and automation  
- ⚙️ Smart manufacturing pipelines  

Despite achieving high anomaly detection accuracy, these systems often behave as **black boxes**, making it difficult to understand:

- **Why** a sample is classified as anomalous
- Whether the model focuses on the **correct visual regions**
- If the anomaly decision is truly **trustworthy**

This work investigates the robustness and reliability of AI-based anomaly detection systems by combining:

- **VAE-GAN** for visual anomaly detection
- **LIME** and **SHAP** for explainability
- Ground-truth comparison using **optimal Jaccard similarity**

---

# 🧠 Proposed Framework

Our framework combines:

- **Variational Autoencoder Generative Adversarial Networks (VAE-GAN)**
- **Explainable AI (XAI) techniques**
- **Anomaly localization and trust analysis**

to analyze whether anomaly detection systems identify anomalies for the **right reason**.

---

## How it works? 

<div align="center">
  <img src="imgs/anomaly_detection_xai.gif" width="90%">
</div>

# ⚙️ Dependencies and Installation

## Requirements

- Python 3.9+
- TensorFlow
- NumPy
- OpenCV
- Matplotlib
- SHAP
- LIME

Optional:
- NVIDIA GPU + CUDA support

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/rashidrao-pk/anomaly_detection_trust_case_study.git

cd anomaly_detection_trust_case_study

pip install -r requirements.txt
```

---

# 📊 Supplementary Material

Experiments were conducted on the [MVTec Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) using:

- 🔩 Screw
- 🌰 Hazelnut

### Generated Results

| Dataset | PDF Results | HTML Results |
|---|---|---|
| Screw | [PDF](https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/blob/main/results/screw/imgs_screw_full.pdf) | [HTML](https://htmlpreview.github.io/?https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/blob/main/results/screw/imgs_screw_full.html) |
| Hazelnut | [PDF](https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/blob/main/results/hazelnut/imgs_hazelnut_full.pdf) | [HTML](https://htmlpreview.github.io/?https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/blob/main/results/hazelnut/imgs_hazelnut_full.html) |

---

# 📁 Repository Structure

```text
├── models/                         # Pre-trained VAE-GAN models
├── results/                        # Generated experimental results
├── imgs/                           # Figures and repository assets
├── utils.py                        # Utility functions
├── models.py                       # VAE-GAN architecture
├── AD_VAE_GAN_SCREW.ipynb          # Screw dataset experiments
├── AD_VAE_GAN_HAZELNUT.ipynb       # Hazelnut dataset experiments
└── requirements.txt
```

---

# 🔬 Main Contributions

This research contributes to the field of trustworthy anomaly detection by:

### ✅ Explainable Anomaly Detection
Combining **VAE-GAN** with **LIME** and **SHAP** to explain anomaly predictions.

### ✅ Reliability Analysis
Evaluating whether detected anomalies correspond to the actual defective regions.

### ✅ Trustworthiness Evaluation
Demonstrating that anomaly detectors can sometimes classify samples correctly for the **wrong visual reasons**.

### ✅ Quantitative Explanation Validation
Using an **optimal Jaccard similarity-based methodology** to compare explanation maps against ground-truth annotations.

---

# 📄 Paper

## Citation

```bibtex
@InProceedings{10.1007/978-3-031-63803-9_13,
  author    = {Rashid, Muhammad and Amparore, Elvio and Ferrari, Enrico and Verda, Damiano},
  editor    = {Longo, Luca and Lapuschkin, Sebastian and Seifert, Christin},
  title     = {Can I Trust My Anomaly Detection System? A Case Study Based on Explainable AI},
  booktitle = {Explainable Artificial Intelligence},
  year      = {2024},
  publisher = {Springer Nature Switzerland},
  address   = {Cham},
  pages     = {243--254}
}
```

---

# 🔑 Keywords

- Anomaly Detection
- Explainable AI (XAI)
- Variational Autoencoder (VAE)
- Generative Adversarial Networks (GANs)
- Trustworthy AI
- Industrial Inspection
- Explainability for Computer Vision

---

# 👨‍💻 Authors

- **Muhammad Rashid**, University of Turin, Italy
- **Elvio Amparore**, University of Turin, Italy
- **Enrico Ferrari**, RuleX Innovation labs, Genova, Italy
- **Damiano Verda**, RuleX Innovation labs, Genova, Italy

---

# 📜 License

This project is released under the MIT License.

---

# ⚠️ Limitations

- Tested primarily on MVTec AD datasets
- TensorFlow-based implementation
- Explanation quality depends on segmentation quality
- Computationally expensive for large-resolution images

# 🚀 Future Work

- PyTorch implementation
- Real-time robotic inspection
- ShapBPT integration
- Improved explanation metrics
- Vision-language anomaly explanations

# 🤝 Contributors Wanted

We are actively welcoming contributions from students, researchers, and developers interested in **explainable AI**, **visual anomaly detection**, and **trustworthy machine learning**.

Good places to start:

- [Add unit tests for anomaly scoring utilities](https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/issues/5) — beginner to intermediate
- [Add a lightweight reproducibility tutorial](https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/issues/7) — beginner
- [Add publication-quality comparison plots](https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/issues/10) — beginner to intermediate

More advanced opportunities include adding [PatchCore](https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/issues/1) and [EfficientAD](https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/issues/2), extending [XAI evaluation](https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/issues/3), and comparing [SHAP, LIME, Grad-CAM, and Integrated Gradients](https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/issues/4).

Browse [all open issues](https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/issues) to find a task matching your interests.

Before starting, please comment on the relevant issue so we can coordinate the work and avoid duplication. Focused pull requests, documentation improvements, reproducibility fixes, and new experimental results are all welcome.

Contributors will be acknowledged in the repository and relevant release notes. Research-paper authorship, where applicable, depends on substantial intellectual and experimental contributions and follows standard authorship guidelines.

We appreciate community contributions toward building more trustworthy and explainable AI systems.

---

# ⭐ Support the Project

If you find this repository useful in your research, please consider:

- ⭐ Starring the repository
- 🍴 Forking the project
- 📚 Citing our paper 

Your support helps improve and expand future research in Explainable AI and anomaly detection.