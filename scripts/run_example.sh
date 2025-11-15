#!/usr/bin/env bash
set -e

# Example pipeline (assumes preprocessed data exists)
python src/pretrain_contrastive.py --data data/benign --save checkpoints/transformer_pretrain.pth
python src/train.py --data data/labeled --pretrained checkpoints/transformer_pretrain.pth --save checkpoints/transformer_ft.pth
python src/fit_ood_stats.py --embeddings data/embeddings_train.npy --labels data/labels_train.npy --out ood_stats.npz
python src/test.py --data data/test --model checkpoints/transformer_ft.pth --ood_stats ood_stats.npz
