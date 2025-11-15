# NIDS Out-of-Distribution Detection with Transformer Enhancements

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]()

This repository contains **PyTorch implementations** and experiments for improving **Out-of-Distribution (OOD) detection** in **Network Intrusion Detection Systems (NIDS)**. It extends the work of **Corsini & Yang (IEEE CNS 2023, CyberOOD)** with:

1. **Transformer-based flow encodings** for richer temporal/semantic representations.
2. **Ensemble calibrated OOD detection** with Mahalanobis, energy scores, contrastive OOD head, and lightweight online adaptation.

---

## **Motivation**

Machine-learning-based NIDS can misclassify novel or zero-day attacks with high confidence. Standard OOD detectors often fail for subtle or “on-manifold” malicious traffic. This project proposes improvements to increase **recall for unknown attacks** while maintaining low false positives and manageable inference latency.

---

## **Repository Structure**

nids_ood_research/
│
├── src/ # Python scripts
│ ├── preprocess_cicids2017.py # Dataset preprocessing
│ ├── dataloader.py # PyTorch dataset/dataloader
│ ├── baseline_run.py # Baseline CyberOOD training/testing
│ ├── improved_run.py # Transformer + ensemble OOD improvements
│ ├── plot_roc.py # ROC comparison plots
│
├── notebooks/ # Jupyter / Colab notebook
│ └── nids_comparison_colab.ipynb # End-to-end experiments
│
├── processed_cicids2017/ # Preprocessed dataset (ignored in git)
├── checkpoints/ # Saved models (ignored in git)
└── results/ # ROC scores and plots (ignored in git)



---

## **Getting Started**

1. **Clone the repository:**

```bash
git clone https://github.com/<yourusername>/nids_ood_research.git
cd nids_ood_research

2. Install dependencies:

pip install -r requirements.txt


3. Download CIC-IDS2017 dataset from official site and place CSVs in CIC-IDS2017-CSV/


4. Run the Colab notebook for preprocessing, training, and evaluation:

notebooks/nids_comparison_colab.ipynb


Scripts Overview:
=================
preprocess_cicids2017.py – Converts raw CSVs into NumPy arrays for training/testing.
dataloader.py – PyTorch Dataset & DataLoader for NIDS flows.
baseline_run.py – Reproduces CyberOOD baseline OOD detection.
improved_run.py – Transformer-based embeddings + ensemble OOD detectors.
plot_roc.py – ROC curves comparison of baseline vs improved models.


Evaluation Metrics:
===================
OOD Detection: AUROC, Recall@1% FPR
Classification: Accuracy, Macro-F1 for known attack classes
Operational: Inference latency (ms/flow), online adaptation time



Results:
========

Expected improvements:
======================
Model				            OOD AUROC	    Recall@1% FPR
Baseline CyberOOD		      ~0.85		        ~0.60
Transformer + Ensemble		~0.92		        ~0.75

ROC curves are generated using plot_roc.py.


References:
===========
1. Andrea Corsini, Shanchieh Jay Yang, “Are Existing Out-of-Distribution Techniques Suitable for Network Intrusion Detection?”, IEEE CNS 2023 (CyberOOD) GitHub
2. A. Manocchio et al., “FlowTransformer: Transformer framework for flow-based NIDS”, ESWA / Computers & Security, 2024
3. CIC-IDS2017 Dataset – Official Link


License:
========
This project is licensed under the MIT License – see LICENSE  for details.

