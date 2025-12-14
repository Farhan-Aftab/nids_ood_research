import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DATA_CSV_DIR = "CIC-IDS2018-CSV"  # change to your CSV folder
OUT_DIR = "processed_cicids2018"

class NpyFlowDataset(torch.utils.data.Dataset):
    def __init__(self, folder, mode='train'):
        if mode=='train':
            self.X = np.load(os.path.join(folder,'X_train.npy'))
            self.Y = np.load(os.path.join(folder,'Y_train.npy'))
        else:
            self.X = np.load(os.path.join(folder,'X_test.npy'))
            self.Y = np.load(os.path.join(folder,'Y_test.npy'))
        self.X = self.X.astype(np.float32)
        self.Y = self.Y.astype(np.int64)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx])

def make_dataloader(folder, split='train', batch_size=64, shuffle=True, num_workers=2):
    ds = NpyFlowDataset(folder, split=split)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
