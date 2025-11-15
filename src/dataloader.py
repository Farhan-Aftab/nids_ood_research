import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class NpyFlowDataset(Dataset):
    def __init__(self, folder, split='train', transform=None):
        self.folder = folder
        self.transform = transform
        X_path = os.path.join(folder, f'X_{split}.npy')
        y_path = os.path.join(folder, f'y_{split}.npy')
        self.X = np.load(X_path)
        self.y = np.load(y_path)
        self.X = self.X.astype('float32')
        self.y = self.y.astype('int64')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = int(self.y[idx])
        if self.transform:
            x = self.transform(x)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

def make_dataloader(folder, split='train', batch_size=64, shuffle=True, num_workers=2):
    ds = NpyFlowDataset(folder, split=split)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)