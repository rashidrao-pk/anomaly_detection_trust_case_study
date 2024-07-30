# Explaining Anomaly Detection based on VAE-GAN Model 🦠⚠️✅🫱🏻‍🫲🏼
      
<img src="https://img.shields.io/badge/version-v0.0.0-rc0" alt="Version">
      <a href ="https://github.com/DmitryRyumin/anomaly_detection_trust_case_study/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
      </a>
<a href="https://github.com/rashidrao-pk/">
<img src="https://img.shields.io/github/contributors/rashidrao-pk/anomaly_detection_trust_case_study" alt="GitHub contributors">
</a>
<img src="https://img.shields.io/github/repo-size/rashidrao-pk/anomaly_detection_trust_case_study" alt="GitHub repo size">
      <a href="https://github.com/rashidrao-pk/">
        <img src="https://img.shields.io/github/commit-activity/t/rashidrao-pk/anomaly_detection_trust_case_study" alt="GitHub commit activity (branch)">
      </a>


<a href="https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/graphs/contributors">
<img src="https://img.shields.io/github/contributors/rashidrao-pk/anomaly_detection_trust_case_study" alt="GitHub contributors">
</a>
<a href="https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/issues?q=is%3Aissue+is%3Aclosed">
<img src="https://img.shields.io/github/issues-closed/rashidrao-pk/anomaly_detection_trust_case_study" alt="GitHub closed issues">
</a>
<a href="https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/issues">
<img src="https://img.shields.io/github/issues/rashidrao-pk/anomaly_detection_trust_case_study" alt="GitHub issues">
</a>
<a href="https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/pulls?q=is%3Apr+is%3Aclosed">
<img src="https://img.shields.io/github/issues-pr-closed/rashidrao-pk/anomaly_detection_trust_case_study" alt="GitHub closed pull requests">
</a>
<a href="https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/pulls">
<img src="https://img.shields.io/github/issues-pr/rashidrao-pk/anomaly_detection_trust_case_study" alt="GitHub pull requests">
</a>

<img src="https://img.shields.io/github/last-commit/rashidrao-pk/anomaly_detection_trust_case_study" alt="GitHub last commit">
<a href="https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/watchers">
<img src="https://img.shields.io/github/watchers/rashidrao-pk/anomaly_detection_trust_case_study?style=flat" alt="GitHub watchers">
</a>
<a href="https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/forks">
<img src="https://img.shields.io/github/forks/rashidrao-pk/anomaly_detection_trust_case_study?style=flat" alt="GitHub forks">
</a>
<a href="https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/stargazers">
<img src="https://img.shields.io/github/stars/rashidrao-pk/anomaly_detection_trust_case_study?style=flat" alt="GitHub Repo stars">
</a>
<img src="https://api.visitorbadge.io/api/combined?path=https%3A%2F%2Fgithub.com%2Frashidrao-pk&label=Visitors&countColor=%23263759&style=flat" alt="Visitors">

This Repositry contains codes to for our accepted paper <b>['Can I trust my anomaly detection system? A case study based on eXaplainable AI'](https://link.springer.com/chapter/10.1007/978-3-031-63803-9_13)</b> into <b>[The 2nd World Conference on eXplainable Artificial Intelligence](https://xaiworldconference.com/2024/)</b> [17-19 July 2024].

<center> <img src='imgs/logo.png' width="25%" height="25%" ></center>
<p> we investigate the robustness of the Anomaly Detection process followed by AI 🤖 based Quality Control Inspection being adopetd in Industries 🏭.</p>

## Dependencies and Installation 🔧
- Python 3.9.18
- Tensorflow
- Option: NVIDIA GPU + CUDA

Clone the repositry and install all the required libraries by running following lines:

```
git clone https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/
cd anomaly_detection_trust_case_study
pip install -r requirements.txt
```

## Supplementary Material 📊
Following are the two Generated files for the results analyzed in the paper <a href='https://www.mvtec.com/company/research/datasets/mvtec-ad'>MVTech dataset </a> [Screw🔩 and Hazelnut 🌰], file containing results for;
1. `Screw Dataset` is uploaded as <a href='https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/blob/main/results/screw/imgs_screw_full.pdf'>**PDF** </a> and <a href='https://htmlpreview.github.io/?https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/blob/main/results/screw/imgs_screw_full.html'>**HTML** </a> file.
2. `Hazelnut Dataset` is uploaded as <a href='https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/blob/main/results/hazelnut/imgs_hazelnut_full.pdf'>**PDF**</a> and <a href='https://htmlpreview.github.io/?https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/blob/main/results/screw/imgs_screw_full.html'>**HTML** </a> file.

## Structure of the Artifact 💻

This artifact is structured as follows:

- the [`results/`](https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/blob/main/results) folder contains the results after running the artifact.
- the [`models/`](https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/blob/main/models) folder contains the models trained and used for testing purposes.
- two notebooks [`AD_VAE_GAN_SCREW.ipynb`](https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/blob/main/AD_VAE_GAN_SCREW.ipynb) and [`VAE_GAN_AD_HAZELNUT.ipynb`](https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/blob/main/AD_VAE_GAN_HAZELNUT.ipynb) which are main files to have all the working to reproduce the results for the proposed approach.
- [`models.py`](https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/blob/main/models.py) contains the codes for VAE-GAN model used in the proposed appoach and [`utils.py`](https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/blob/main/utils.py) contains all the functions required to run both notebooks ( `AD_VAE_GAN_SCREW.ipynb` & `VAE_GAN_AD_HAZELNUT.ipynb`).

## Contributions 📃
In this research, we:

1. *Review an explainable Anomaly Detection system architecture that combines VAE-GAN
models with the LIME and SHAP explanation methods;*
2. *Quantify the capacity of the Anomaly Detection system in performing anomaly detection
using anomaly scores;*
3. *Use XAI methods to determine if anomalies are actually detected for the
right reason by comparing with a ground truth. Results show that it is not
uncommon to find samples that were classified as anomalous, but for the
wrong reason. We adopt a methodology based on optimal Jaccard score to
detect such samples.*

## Paper PDF:
Paper can be found at [LINK]() uploaded on <a href=''> <img src="https://cdn.jsdelivr.net/gh/DmitryRyumin/NewEraAI-Papers@main/images/arxiv-logo.svg" width="45" alt="" />
</a>
### Authors ✍️

| Sr. No. | Author Name | Affiliation | Google Scholar | 
| :--:    | :--:        | :--:        | :--:           | 
| 1. | Muhammad Rashid | University of Torino, Computer Science Department, C.so Svizzera 185, 10149 Torino, Italy | [Muhammad Rashid](https://scholar.google.com/citations?user=F5u_Z5MAAAAJ&hl=en) | 
| 2. | Elvio G. Amparore | University of Torino, Computer Science Department, C.so Svizzera 185, 10149 Torino, Italy | [Elvio G. Amparore](https://scholar.google.com/citations?user=Hivlp1kAAAAJ&hl=en&oi=ao) | 
| 3. | Enrico Ferrari | Rulex Innovation Labs, Rulex Inc., Via Felice Romani 9, 16122 Genova, Italy | [Enrico Ferrari](https://scholar.google.com/citations?user=QOflGNIAAAAJ&hl=en&oi=ao) | 
| 4. | Damiano Verda | Rulex Innovation Labs, Rulex Inc., Via Felice Romani 9, 16122 Genova, Italy | [Damiano Verda](https://scholar.google.com/citations?user=t6o9YSsAAAAJ&hl=en&oi=ao) |


### Cite Us
```
@InProceedings{10.1007/978-3-031-63803-9_13, author="Rashid, Muhammad and Amparore, Elvio and Ferrari, Enrico and Verda, Damiano", editor="Longo, Luca and Lapuschkin, Sebastian and Seifert, Christin", title="Can I Trust My Anomaly Detection System? A Case Study Based on Explainable AI", booktitle="Explainable Artificial Intelligence",
year="2024", publisher="Springer Nature Switzerland",
address="Cham", pages="243--254"}
```


### Keywords 🔍
Anomaly detection · variational autoencoder · eXplainable
AI

### Copyright Notice:
MIT license
Author: Muhammad Rashid (muhammad.rashid@unito.it)
University of Turin, Italy.

## Contributors

<a href="https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/graphs/contributors">
  <img src="http://contributors.nn.ci/api?repo=rashidrao-pk/anomaly_detection_trust_case_study" alt="" />
</a>
<br>

> [!NOTE]
> Contributions to improve the completeness of this list are greatly appreciated. If you come across any overlooked papers, please **feel free to [*create pull requests*](https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/pulls), [*open issues*](https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/issues) or contact me via [*email*](mailto:muhammad.rashid@unito.it)**. Your participation is crucial to making this repository even better.

