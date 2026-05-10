# Contributing

Thank you for your interest in contributing to this repository.

This project contains the implementation and experimental artifacts for:

> **Can I Trust My Anomaly Detection System? A Case Study Based on Explainable AI**

The repository focuses on VAE-GAN-based anomaly detection and the use of eXplainable AI methods such as LIME and SHAP to evaluate whether anomaly detection models identify anomalies for the right reasons.

## Ways to Contribute

You can contribute by:

- Improving documentation
- Fixing typos or broken links
- Improving code readability
- Reporting bugs
- Adding reproducibility instructions
- Improving notebooks
- Adding comments or explanations to the code
- Suggesting better evaluation or visualization methods
- Reporting issues with dependencies or installation
- Extending the work to other datasets or XAI methods

## Repository Structure

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

## Getting Started

Clone the repository:

```bash
git clone https://github.com/rashidrao-pk/anomaly_detection_trust_case_study.git
cd anomaly_detection_trust_case_study
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Before Contributing

Before submitting a contribution, please make sure that:

- Your code runs without errors.
- Notebook outputs are clear and relevant.
- New code is documented where necessary.
- File paths are relative whenever possible.
- Large generated files are not added unless required.
- The contribution is relevant to anomaly detection, VAE-GAN, XAI, or reproducibility.
- Reporting Bugs

### When reporting a bug, please include:**

- A clear description of the problem
- Steps to reproduce the issue
- Your operating system
- Python version
- TensorFlow version
- Error message or traceback
- Screenshots, if helpful

### Example:

```text
Problem:
The notebook fails when loading the trained model.

Steps to reproduce:
1. Run AD_VAE_GAN_SCREW.ipynb
2. Execute the model loading cell
3. Error appears

Environment:
- Python 3.9
- TensorFlow version:
- OS:
```


## Suggesting Improvements

Suggestions are welcome. Please describe:

- What you want to improve
- Why it is useful
- Whether it affects code, documentation, experiments, or results
- Any references or examples, if available

## Pull Request Guidelines
When opening a pull request:

- Use a clear title.
- Describe what you changed.
- Mention the issue number, if related.
- Keep changes focused.
- Avoid mixing unrelated changes in one pull request.
- Make sure the repository still runs after your changes.

**Example pull request title:**
```text
Improve documentation for running VAE-GAN experiments on MVTec Screw
```


## Code Style

Please follow these guidelines:

- Use clear variable names.
- Add comments for non-trivial logic.
- Avoid unnecessary complexity.
- Keep notebooks clean and readable.
- Do not hard-code local machine paths.
- Use relative paths where possible.

## Documentation Style

When improving documentation:

- Use clear and simple language.
- Explain technical terms where useful.
- Keep Markdown formatting consistent.
- Check links before submitting.
- Avoid adding unsupported claims.

## Citation

If you use this repository in your research, please cite:

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

## License

By contributing to this repository, you agree that your contributions will be licensed under the MIT License.

## Contact

For questions, suggestions, or collaboration, please contact:

Muhammad Rashid
- Email: muhammad.rashid@unito.it
- GitHub: https://github.com/rashidrao-pk

