# NIDS Out-of-Distribution Detection with Transformer Enhancements

This repository contains PyTorch implementations and experiments for improving Out-of-Distribution (OOD) detection in Network Intrusion Detection Systems (NIDS). It extends the work of Corsini & Yang (IEEE CNS 2023, CyberOOD) with:

1. Transformer-based flow encodings for richer temporal/semantic representations.
2. Ensemble calibrated OOD detection with Mahalanobis, energy scores, contrastive OOD head, and lightweight online adaptation.

## Repository Structure

```
nids_ood_research/
├── README.md
├── LICENSE
├── requirements.txt
├── src/
│   ├── preprocess_cicids2017.py
│   ├── dataloader.py
│   ├── baseline_run.py
│   ├── improved_run.py
│   └── plot_roc.py
└── notebooks/
    └── nids_comparison_colab.ipynb
```

## Quick start

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Download CIC-IDS2017 CSVs and place them in `CIC-IDS2017-CSV/`.
3. Run the Colab notebook or the scripts in `src/` as described in the paper.

## References

- Andrea Corsini, Shanchieh Jay Yang, "Are Existing Out-of-Distribution Techniques Suitable for Network Intrusion Detection?", IEEE CNS 2023 (CyberOOD)
- CIC-IDS2017 Dataset
