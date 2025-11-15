import numpy as np
import torch
import torch.nn.functional as F

class MahalanobisStats:
    def __init__(self):
        self.class_means = {}
        self.precision = None

    def fit(self, embeddings, labels):
        embeddings = np.asarray(embeddings)
        labels = np.asarray(labels)
        classes = np.unique(labels)
        for c in classes:
            self.class_means[c] = embeddings[labels==c].mean(axis=0)
        cov = np.cov(embeddings.T) + 1e-6*np.eye(embeddings.shape[1])
        self.precision = np.linalg.pinv(cov)

    def mahalanobis(self, emb):
        # emb shape (B, D)
        scores = []
        for c, mu in self.class_means.items():
            delta = emb - mu[None,:]
            m = np.sum(delta.dot(self.precision) * delta, axis=1)
            scores.append(m)
        return np.min(np.stack(scores, axis=1), axis=1)

def energy_score(logits, T=1.0):
    return -T * torch.logsumexp(logits / T, dim=1).detach().cpu().numpy()
