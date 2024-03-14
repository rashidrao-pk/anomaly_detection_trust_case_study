# Explaining Anomaly Detection based on VAE-GAN Models 🦠⚠️✅🫱🏻‍🫲🏼

This Repositry contains codes to reproduce Results for our submission <b>'Can I trust my anomaly detection system? A case study'</b> 
<p> we investigate the robustness of the process followed by AI 🤖 based Quality Control Inspection being adopetd in Industries 🏭 </p>

### Dependencies and Installation 🔧
- Python 3.9.18
- Tensorflow
- Option: NVIDIA GPU + CUDA

Clone the repositry and install all the required libraries by running following line:

```
git clone https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/
cd anomaly_detection_trust_case_study
pip install -r requirements.txt
```

### Structure of the artifact 💻

This artifact is structured as follows:

- the [`results/`](https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/blob/main/results) folder, which contains the results after running the artifact.
- the [`models/`](https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/blob/main/models) folder, which contains the models trained and used for testing purposes.

- two notebooks [`AD_VAE_GAN_SCREW.ipynb`](https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/AD_VAE_GAN_SCREW.ipynb) and [`VAE_GAN_AD_hazelnut.ipynb`](https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/AD_VAE_GAN_HAZELNUT.ipynb)

### Paper contribution 📃
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

### Authors ✍️
[Muhammad Rashid<sup>1</sup>](https://scholar.google.com/citations?user=F5u_Z5MAAAAJ&hl=en), [Elvio G. Amparore<sup>1</sup>](https://scholar.google.com/citations?user=Hivlp1kAAAAJ&hl=en&oi=ao), [Enrico Ferrari<sup>2</sup>](https://scholar.google.com/citations?user=QOflGNIAAAAJ&hl=en&oi=ao), [Damiano Verda<sup>2</sup>](https://scholar.google.com/citations?user=t6o9YSsAAAAJ&hl=en&oi=ao)
1. University of Torino, Computer Science Department, C.so Svizzera 185, 10149 Torino, Italy
2. Rulex Innovation Labs, Rulex Inc., Via Felice Romani 9, 16122 Genova, Italy
### Keywords 🔍
Anomaly detection · variational autoencoder · eXplainable
AI
### Supplementary Material 📊
Following are the two Genearted File for the results analyzed in the paper <a href='https://www.mvtec.com/company/research/datasets/mvtec-ad'>MVTech dataset </a> [Screw🔩 and Hazelnut 🌰] and results using;
1. `Screw Dataset` is uploaded <a href='https://htmlpreview.github.io/?https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/blob/main/results/imgs_screw_full.html'>**here** </a>
2. `Hazelnut Dataset` is uploaded <a href='https://htmlpreview.github.io/?https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/blob/main/results/imgs_hazelnut_full.html'>**here**</a>


