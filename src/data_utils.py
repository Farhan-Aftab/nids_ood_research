import numpy as np
import os
from tqdm import tqdm

# Simple helpers to load/save per-flow packet sequence numpy arrays
# Expected format: each flow is a fixed-length sequence array (L, feature_dim)

def load_flow_dataset(folder):
    # folder contains .npy files: flows.npy (N, L, D), labels.npy (N,) optionally
    flows = np.load(os.path.join(folder, 'flows.npy'))
    labels_path = os.path.join(folder, 'labels.npy')
    labels = None
    if os.path.exists(labels_path):
        labels = np.load(labels_path)
    return flows, labels

def save_embeddings(path, emb):
    np.save(path, emb)
