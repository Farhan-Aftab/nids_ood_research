import numpy as np

# Correct paths (NO labels.npy exists)
y_train = np.load("/content/drive/MyDrive/Colab Notebooks/nids_ood_research/src/preprocess_cicids2018/Y_train.npy")
y_test  = np.load("/content/drive/MyDrive/Colab Notebooks/nids_ood_research/src/preprocess_cicids2018/Y_test.npy")

# Combine to check all labels
labels = np.concatenate([y_train, y_test])

print("Unique labels:", np.unique(labels))
print("Number of classes:", len(np.unique(labels)))