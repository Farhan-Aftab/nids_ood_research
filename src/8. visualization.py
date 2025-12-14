import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_curve
)
from sklearn.decomposition import PCA

# =============================
#  PATH CONFIG
# =============================
RESULT_DIR = "/content/drive/MyDrive/Colab Notebooks/nids_ood_research/src"

TRAIN_EMBS_FILE   = os.path.join(RESULT_DIR, "train_embs.npy")
TRAIN_LABELS_FILE = os.path.join(RESULT_DIR, "train_labels.npy")
TEST_SCORES_FILE  = os.path.join(RESULT_DIR, "scores_improved.npy")
LOGITS_FILE       = os.path.join(RESULT_DIR, "logits_improved.npy")
LABELS_FILE       = os.path.join(RESULT_DIR, "labels_improved.npy")
PREDS_FILE        = os.path.join(RESULT_DIR, "preds_improved.npy")

# =============================
#  Utility
# =============================
def load_npy(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path)

def softmax_np(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

# =============================
#  Plots
# =============================
def plot_score_distribution(scores, labels):
    plt.figure(figsize=(8,5))
    plt.hist(scores[labels==0], bins=50, alpha=0.6, label="Benign")
    plt.hist(scores[labels!=0], bins=50, alpha=0.6, label="Attacks")
    plt.legend()
    plt.title("OOD Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("score_distribution.png", dpi=300)
    plt.show()

    plt.figure(figsize=(10,6))
    plt.hist(scores_benign, bins=100, alpha=0.6, label="Benign", log=True)
    plt.hist(scores_attack, bins=100, alpha=0.6, label="Attacks", log=True)
    plt.xlabel("Score")
    plt.ylabel("Count (log scale)")
    plt.legend()
    plt.title("OOD Score Distribution (Log Scale)")
    plt.show()

    sns.kdeplot(scores_benign, label="Benign", fill=True)
    sns.kdeplot(scores_attack, label="Attacks", fill=True)
    plt.xlabel("Score")
    plt.title("KDE Score Density (Benign vs Attacks)")
    plt.legend()
    plt.show()

def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()

def plot_pca(embeddings, labels):
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(emb_2d[:,0], emb_2d[:,1], c=labels, cmap="tab20", s=10, alpha=0.7)
    plt.colorbar(scatter)
    plt.title("PCA of Train Embeddings")
    plt.tight_layout()
    plt.savefig("pca_train_embeddings.png", dpi=300)
    plt.show()

def plot_pca_ood(embeddings, labels):
    """
    PCA of embeddings colored by OOD (0 = benign, 1 = attack)
    """
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)

    # Binary labels: 0 = benign, 1 = attack
    ood_labels = (labels != 0).astype(int)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(
        emb_2d[:,0], emb_2d[:,1],
        c=ood_labels, cmap="coolwarm", s=12, alpha=0.7
    )
    cbar = plt.colorbar(scatter, ticks=[0,1])
    cbar.ax.set_yticklabels(['Benign', 'Attack'])
    plt.title("PCA of Train Embeddings (OOD vs Benign)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig("pca_train_embeddings_ood.png", dpi=300)
    plt.show()

# =============================
#  OOD Metrics
# =============================
def compute_ood_metrics(scores, labels):
    y_true = (labels != 0).astype(int)
    auroc = roc_auc_score(y_true, scores)
    fpr, tpr, thr = roc_curve(y_true, scores)
    idx = np.where(fpr <= 0.01)[0]
    recall_1p = tpr[idx[-1]] if len(idx) else tpr[0]
    return auroc, recall_1p

# =============================
#  MAIN
# =============================
def main():
    # Load data
    train_embs   = load_npy(TRAIN_EMBS_FILE)
    train_labels = load_npy(TRAIN_LABELS_FILE)
    test_scores  = load_npy(TEST_SCORES_FILE)
    logits_test  = load_npy(LOGITS_FILE)
    labels_test  = load_npy(LABELS_FILE)
    preds_test   = load_npy(PREDS_FILE)

    # Multi-class classification metrics
    acc = accuracy_score(labels_test, preds_test)
    macro_f1 = f1_score(labels_test, preds_test, average="macro")
    print("\n===== MULTI-CLASS CLASSIFICATION METRICS =====")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Macro F1     : {macro_f1:.4f}")
    plot_confusion(labels_test, preds_test)

    # OOD detection metrics (using energy scores)
    energy_scores = np.logaddexp.reduce(logits_test, axis=1)
    auroc, recall_1p = compute_ood_metrics(energy_scores, labels_test)
    print("\n===== OOD DETECTION METRICS =====")
    print(f"AUROC                : {auroc:.4f}")
    print(f"Recall @ 1% FPR      : {recall_1p:.4f}")
    plot_score_distribution(energy_scores, labels_test)

    # PCA of train embeddings
    plot_pca(train_embs, train_labels)

    # PCA of OOD train embeddings
    plot_pca_ood(train_embs, train_labels)

    print("\nAll visualizations generated successfully!")

if __name__ == "__main__":
    main()
