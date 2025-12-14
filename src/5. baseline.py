import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
%matplotlib inline

# =========================
# Dataset for NPY files
# =========================
class NPYDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# =========================
# MLP Model
# =========================
class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden=256, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# =========================
# Training Function
# =========================
def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

# =========================
# Evaluation (AUROC + basic metrics)
# =========================
def eval_ood(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()

            all_probs.append(probs)
            all_labels.append(y.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # MULTI-CLASS AUROC
    auc_value = roc_auc_score(
        all_labels,
        all_probs,
        multi_class='ovr',
        average='macro'
    )

    print("\nAUROC (macro-ovr):", auc_value)

    # ===== BINARY AUROC (Benign vs Attack) =====
    # Treat class 0 = benign, others = attack
    bin_labels = (all_labels != 0).astype(int)
    bin_scores = 1 - probs[:, 0]   # high score = attack

    try:
        bin_auc = roc_auc_score(bin_labels, bin_scores)
        print("\nBinary AUROC (Benign vs Attack):", bin_auc)
    except:
        bin_auc = None
        print("\nBinary AUROC could not be computed.")

    # Convert to predicted classes for metrics
    preds = np.argmax(all_probs, axis=1)

    print("\nClassification Report:")
    print(classification_report(all_labels, preds))

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, preds))

    return auc_value, all_probs, all_labels, preds

# =========================
# Plotting Function
# =========================
#def plot_results(all_labels, preds, probs):  CHANGE HERE
def plot_results(all_labels, preds, probs, X_train, model, device):
    num_classes = probs.shape[1]

    # -------- CONFUSION MATRIX --------
    cm = confusion_matrix(all_labels, preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()

    # ---- FIX: ADD VALUES ON SQUARES ----
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                    ha='center', va='center', color='black')

    plt.tight_layout()
    plt.show()

    # -------- ROC CURVE --------
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(all_labels, classes=np.arange(num_classes))

    fpr, tpr, roc_auc = {}, {}, {}

    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC={roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title("Multi-Class ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

# -------- PCA Train --------
def pca_train_embeddings(model, X_train, device):
    model.eval()

    # Extract features from the hidden layer
    with torch.no_grad():
        #x = torch.tensor(X_train, dtype=torch.float32).to(device)  CHANGE HERE
        x = torch.tensor(X_train, dtype=torch.float32).to(device)
        # Forward pass until second layer (before final output)
        hidden = model.net[:-1](x).cpu().numpy()

    # PCA to 2 components
    pca = PCA(n_components=2)
    proj = pca.fit_transform(hidden)

    plt.figure(figsize=(7, 6))
    plt.scatter(proj[:, 0], proj[:, 1], s=3, alpha=0.5)
    plt.title("PCA of Train Embeddings (Hidden Layer)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.show()

# -------- Plot Score Distribution --------
def plot_score_distribution(probs):
    max_scores = np.max(probs, axis=1)

    plt.figure(figsize=(7, 5))
    plt.hist(max_scores, bins=50, alpha=0.7)
    plt.title("Distribution of Softmax Scores (Confidence)")
    plt.xlabel("Max Softmax Probability")
    plt.ylabel("Frequency")
    plt.show()

# =========================
# MAIN
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_DIR = "/content/drive/MyDrive/Colab Notebooks/nids_ood_research/src/preprocess_cicids2018"

    # Load NPY datasets
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    Y_train = np.load(os.path.join(DATA_DIR, "Y_train.npy"))
    Y_test = np.load(os.path.join(DATA_DIR, "Y_test.npy"))

    print("Loaded NPY shapes:")
    print("X_train:", X_train.shape)
    print("Y_train:", Y_train.shape)
    print("X_test:", X_test.shape)
    print("Y_test:", Y_test.shape)

    train_loader = DataLoader(NPYDataset(X_train, Y_train), batch_size=128, shuffle=True)
    test_loader = DataLoader(NPYDataset(X_test, Y_test), batch_size=128, shuffle=False)

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(Y_train))

    model = BaselineMLP(input_dim=input_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # TRAINING LOOP
    for ep in range(5):
        loss = train(model, train_loader, optimizer, device)
        auc_value, probs, labels, preds = eval_ood(model, test_loader, device)
        print(f"Epoch {ep}: Loss={loss:.4f}  AUROC={auc_value:.4f}")

    # Save model
    torch.save(model.state_dict(), "baseline_model.pth")
    print("Saved model as baseline_model.pth")

    # PLOTTING (after final epoch)
    #plot_results(labels, preds, probs)  CHANGE HERE
    plot_results(labels, preds, probs, X_train, model, device)

    # PCA of Train Embeddings
    pca_train_embeddings(model, X_train, device)

    # Score Distribution
    plot_score_distribution(probs)

if __name__ == "__main__":
    main()
