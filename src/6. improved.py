# improved_pipeline_memory_friendly.py
# Fully corrected + memory safe + syntax clean
# Author: Adapted for Farhan (NUST) – 2025

import os
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, accuracy_score, f1_score,
    roc_curve
)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ---------------------------
# CONFIG
# ---------------------------
DATA_DIR = "/content/drive/MyDrive/Colab Notebooks/nids_ood_research/src/preprocess_cicids2018"
RESULT_DIR = "/content/drive/MyDrive/Colab Notebooks/nids_ood_research/src"
os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_DEVICE = True

# hyperparameters
LATENT_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 64        # lowered for memory
EPOCHS = 15
LR = 1e-3
WEIGHTED_CE = True
USE_FOCAL = True
FOCAL_GAMMA = 2.0
OVERSAMPLE = True      # via WeightedRandomSampler (SAFE)
USE_MAHALANOBIS = False
SEED = 42

MODEL_PATH = os.path.join(RESULT_DIR, "improved_model_final.pth")
TRAIN_EMBS_FILE = os.path.join(RESULT_DIR, "train_embs.npy")
TRAIN_LABELS_FILE = os.path.join(RESULT_DIR, "train_labels.npy")
SCORES_FILE = os.path.join(RESULT_DIR, "scores_improved.npy")
LABELS_FILE = os.path.join(RESULT_DIR, "labels_improved.npy")
LOGITS_FILE = os.path.join(RESULT_DIR, "logits_improved.npy")
PREDS_FILE = os.path.join(RESULT_DIR, "preds_improved.npy")

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ---------------------------
# Utils
# ---------------------------
def safe_load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    return np.load(path)

def softmax_np(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

# ---------------------------
# Dataset Wrapper
# ---------------------------
class NPYDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------------------
# Models
# ---------------------------
class EncoderMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, latent_dim)
        )
    def forward(self, x):
        return self.net(x)

class Classifier(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, num_classes=2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim//2),
            nn.ReLU(),
            nn.Linear(latent_dim//2, num_classes)
        )
    def forward(self, z):
        return self.head(z)

# ---------------------------
# Losses
# ---------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt)**self.gamma) * ce
        return loss.mean() if self.reduction == "mean" else loss.sum()

# ---------------------------
# Mahalanobis (optional)
# ---------------------------
class MahalanobisStats:
    def fit(self, embs, labels):
        classes = np.unique(labels)
        self.means = {}
        all_embs = []

        for c in classes:
            zc = embs[labels == c]
            self.means[c] = zc.mean(axis=0)
            all_embs.append(zc)

        all_embs = np.concatenate(all_embs, axis=0)
        cov = np.cov(all_embs, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6
        self.inv_cov = np.linalg.pinv(cov)

    def mahal(self, Z):
        scores = []
        for zi in Z:
            ds = []
            for c, mu in self.means.items():
                diff = zi - mu
                ds.append(diff @ self.inv_cov @ diff.T)
            scores.append(min(ds))
        return np.array(scores)

# ---------------------------
# Metrics
# ---------------------------
def compute_metrics_binary_from_scores(scores, labels):
    binary_labels = (labels != 0).astype(int)
    auroc = roc_auc_score(binary_labels, scores)

    fpr, tpr, thr = roc_curve(binary_labels, scores)
    idx = np.where(fpr <= 0.01)[0]
    recall_at_1p = tpr[idx[-1]] if len(idx) > 0 else tpr[0]

    return auroc, recall_at_1p

# ---------------------------
# Sampler for Memory-Safe Oversampling
# ---------------------------
def make_sampler(y_train):
    if not OVERSAMPLE:
        return None

    y = np.array(y_train)
    class_counts = np.bincount(y)
    class_counts = np.maximum(class_counts, 1)

    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

# ---------------------------
# Training & Evaluation
# ---------------------------
def train_one_epoch(encoder, clf, loader, optimizer, loss_fn):
    encoder.train(); clf.train()
    running_loss = 0
    total = 0

    pbar = tqdm(loader, desc="Train", ncols=120, leave=False)
    for xb, yb in pbar:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        z = encoder(xb)
        logits = clf(z)
        loss = loss_fn(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        total += xb.size(0)

        pbar.set_postfix(loss=running_loss / total)

    return running_loss / total

def evaluate(encoder, clf, loader):
    encoder.eval(); clf.eval()

    logits_list = []
    embs_list = []
    labels_list = []

    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="Eval", ncols=120, leave=False):
            xb = xb.to(DEVICE)

            z = encoder(xb)
            logits = clf(z)

            logits_list.append(logits.cpu().numpy().astype(np.float32))
            embs_list.append(z.cpu().numpy().astype(np.float32))
            labels_list.append(yb.numpy())

    logits = np.concatenate(logits_list, axis=0)
    embs = np.concatenate(embs_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    preds = np.argmax(logits, axis=1)
    probs = softmax_np(logits)

    return logits, probs, preds, labels, embs

# ---------------------------
# Main Pipeline
# ---------------------------
def run_pipeline():

    print("\nDEVICE:", DEVICE)

    # load data
    X_train = safe_load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = safe_load(os.path.join(DATA_DIR, "Y_train.npy"))
    X_test  = safe_load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test  = safe_load(os.path.join(DATA_DIR, "Y_test.npy"))

    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    # scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_ds = NPYDataset(X_train, y_train)
    test_ds = NPYDataset(X_test, y_test)

    sampler = make_sampler(y_train)

    if sampler is not None:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = int(y_train.max()) + 1

    # model
    input_dim = X_train.shape[1]
    encoder = EncoderMLP(input_dim).to(DEVICE)
    clf = Classifier(latent_dim=LATENT_DIM, num_classes=num_classes).to(DEVICE)

    # loss
    if WEIGHTED_CE:
        class_counts = np.bincount(y_train, minlength=num_classes)
        class_counts = np.maximum(class_counts, 1)
        weights = (1 / class_counts)
        weights = weights / weights.sum() * num_classes
        weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
        ce_loss = nn.CrossEntropyLoss(weight=weights)
    else:
        ce_loss = nn.CrossEntropyLoss()

    loss_fn = FocalLoss(gamma=FOCAL_GAMMA) if USE_FOCAL else ce_loss

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(clf.parameters()), lr=LR
    )

    # training loop
    print("\nStarting training...\n")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(encoder, clf, train_loader, optimizer, loss_fn)

        logits_val, probs_val, preds_val, labels_val, embs_val = evaluate(
            encoder, clf, test_loader
        )

        energy_scores = np.logaddexp.reduce(logits_val, axis=1)
        auroc, recall_1p = compute_metrics_binary_from_scores(energy_scores, labels_val)
        acc = accuracy_score(labels_val, preds_val)
        macro_f1 = f1_score(labels_val, preds_val, average="macro")

        print(f"Epoch {epoch}/{EPOCHS}  loss={train_loss:.4f}  acc={acc:.4f}  "
              f"macroF1={macro_f1:.4f}  AUROC={auroc:.4f}  recall@1%FPR={recall_1p:.4f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # final evaluation
    logits_test, probs_test, preds_test, labels_test, embs_test = evaluate(
        encoder, clf, test_loader
    )
    energy_scores = np.logaddexp.reduce(logits_test, axis=1)

    if USE_MAHALANOBIS:
        print("Fitting Mahalanobis...")
        encoder.eval()
        Z_train = []
        Y_train = []

        with torch.no_grad():
            for xb, yb in DataLoader(train_ds, batch_size=BATCH_SIZE):
                z = encoder(xb.to(DEVICE)).cpu().numpy().astype(np.float32)
                Z_train.append(z)
                Y_train.append(yb.numpy())

        Z_train = np.concatenate(Z_train, axis=0)
        Y_train = np.concatenate(Y_train, axis=0)

        m = MahalanobisStats()
        m.fit(Z_train, Y_train)
        maha = m.mahal(embs_test)
        final_scores = energy_scores - maha
    else:
        final_scores = energy_scores

    bin_labels = (labels_test != 0).astype(int)
    auroc_final = roc_auc_score(bin_labels, final_scores)

    fpr, tpr, thr = roc_curve(bin_labels, final_scores)
    idx = np.where(fpr <= 0.01)[0]
    recall_1p = tpr[idx[-1]] if len(idx) else tpr[0]

    acc_final = accuracy_score(labels_test, preds_test)
    macrof1_final = f1_score(labels_test, preds_test, average="macro")

    print("\n=== FINAL METRICS ===")
    print("Binary AUROC :", auroc_final)
    print("Recall@1%FPR:", recall_1p)
    print("Accuracy    :", acc_final)
    print("Macro F1    :", macrof1_final)

    # save results
    np.save(LOGITS_FILE, logits_test)
    np.save(LABELS_FILE, labels_test)
    np.save(PREDS_FILE, preds_test)
    np.save(SCORES_FILE, final_scores)

    # embeddings save
    encoder.eval()
    Z_train2, Y_train2 = [], []
    with torch.no_grad():
        for xb, yb in DataLoader(train_ds, batch_size=BATCH_SIZE):
            z = encoder(xb.to(DEVICE)).cpu().numpy().astype(np.float32)
            Z_train2.append(z)
            Y_train2.append(yb.numpy())

    Z_train2 = np.concatenate(Z_train2, axis=0)
    Y_train2 = np.concatenate(Y_train2, axis=0)

    np.save(TRAIN_EMBS_FILE, Z_train2)
    np.save(TRAIN_LABELS_FILE, Y_train2)

    # PCA plot
    pca = PCA(n_components=2)
    emb2 = pca.fit_transform(Z_train2)

    plt.figure(figsize=(8,6))
    sc = plt.scatter(emb2[:, 0], emb2[:, 1], c=Y_train2, s=6, cmap="tab20")
    plt.colorbar(sc)
    plt.title("PCA of Train Embeddings")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "pca_train_embs.png"))
    plt.close()

    # Save model
    torch.save({
        "encoder": encoder.state_dict(),
        "classifier": clf.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_var": scaler.var_
    }, MODEL_PATH)

    print("\nModel + artifacts saved.")

    return {
        "auroc": float(auroc_final),
        "recall_1%": float(recall_1p),
        "accuracy": float(acc_final),
        "macro_f1": float(macrof1_final),
    }


if __name__ == "__main__":
    results = run_pipeline()
    print("\nRESULT SUMMARY:", results)
