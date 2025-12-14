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
│   ├── 0. All_in_one_Colab_NSR.ipynb
│   ├── 1. install_requirements.py
│   ├── 2. preprocessing.py
│   ├── 3. downloader.py
│   └── 4. checking_number_of_classes.py
│   ├── 5. baseline.py
│   ├── 6. improved.py
│   └── 7. visualization.py
├── src/processed_cicids2018/
│       └── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
│       └── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX
│       └── Friday-WorkingHours-Morning.pcap_ISCX
│       └── Monday-WorkingHours.pcap_ISCX
│       └── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX
│       └── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX
│       └── Tuesday-WorkingHours.pcap_ISCX
└── notebooks/
    └── 1.txt
```

## Quick start

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Download CIC-IDS2018 CSVs and place them in `CIC-IDS2018-CSV/`.
3. Run the Colab notebook or the scripts in `src/` as described in the paper.

## References

- Andrea Corsini, Shanchieh Jay Yang, "Are Existing Out-of-Distribution Techniques Suitable for Network Intrusion Detection?", IEEE CNS 2023 (CyberOOD)
- CIC-IDS2018 Dataset
