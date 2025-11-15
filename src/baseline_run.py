import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from dataloader import NpyFlowDataset

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

def train(model, loader, optimizer, device):
    model.train()
    total_loss=0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*x.size(0)
    return total_loss/len(loader.dataset)

def eval_ood(model, loader, device):
    model.eval()
    probs=[]; labels=[]
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device)
            logits = model(x)
            p = F.softmax(logits, dim=1)[:,1].cpu().numpy()
            probs.append(p)
            labels.append(y.numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    auc = roc_auc_score(labels, probs)
    np.save('scores_baseline.npy', probs)
    np.save('labels_baseline.npy', labels)
    return auc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(NpyFlowDataset('processed_cicids2017','train'), batch_size=128, shuffle=True)
    test_loader = DataLoader(NpyFlowDataset('processed_cicids2017','test'), batch_size=128, shuffle=False)
    X0 = np.load('processed_cicids2017/X_train.npy'); input_dim = X0.shape[1]
    model = BaselineMLP(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(3):
        loss = train(model, train_loader, optimizer, device)
        auc = eval_ood(model, test_loader, device)
        print(f"Epoch {ep}: Loss={loss:.4f} AUROC={auc:.4f}")
    torch.save(model.state_dict(), 'baseline_model.pth')

if __name__=='__main__':
    main()