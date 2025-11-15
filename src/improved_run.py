import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from dataloader import NpyFlowDataset

class FlowTransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=128, proj_dim=128):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1,1,d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(nn.Linear(d_model, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))

    def forward(self, x):
        if x.dim()==2: x=x.unsqueeze(1)
        z = self.transformer(self.input_proj(x)+self.pos_emb)
        z = self.pool(z.permute(0,2,1)).squeeze(-1)
        return self.proj(z)

class MahalanobisStats:
    def fit(self, embeddings, labels):
        self.means = {c: embeddings[labels==c].mean(axis=0) for c in np.unique(labels)}
        cov = np.cov(embeddings.T)+1e-6*np.eye(embeddings.shape[1])
        self.prec = np.linalg.pinv(cov)
    def mahal(self, emb):
        return np.min(np.stack([np.sum((emb-m)**2 @ self.prec, axis=1) for m in self.means.values()], axis=1), axis=1)

def energy_score(logits, T=1.0):
    return -T*torch.logsumexp(logits/T, dim=1).detach().cpu().numpy()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(NpyFlowDataset('processed_cicids2017','train'), batch_size=128, shuffle=True)
    test_loader = DataLoader(NpyFlowDataset('processed_cicids2017','test'), batch_size=128, shuffle=False)
    X0 = np.load('processed_cicids2017/X_train.npy'); input_dim = X0.shape[1]
    encoder = FlowTransformerEncoder(input_dim).to(device)
    clf = nn.Linear(128,2).to(device)
    optim = torch.optim.Adam(list(encoder.parameters())+list(clf.parameters()), lr=1e-4)
    for ep in range(3):
        encoder.train(); clf.train(); total=0.0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            z = encoder(x); logits=clf(z)
            loss = F.cross_entropy(logits, y); optim.zero_grad(); loss.backward(); optim.step()
            total+=loss.item()*x.size(0)
        print(f'Epoch {ep}: Loss={total/len(train_loader.dataset):.4f}')
    # Mahalanobis fit
    encoder.eval(); clf.eval()
    train_embs = np.concatenate([encoder(x.to(device)).cpu().detach().numpy() for x,_ in train_loader])
    train_labels = np.concatenate([y.numpy() for _,y in train_loader])
    m = MahalanobisStats(); m.fit(train_embs, train_labels)
    # Test
    all_scores, all_labels = [], []
    for x,y in test_loader:
        x = x.to(device)
        z = encoder(x).cpu().detach().numpy()
        logits = clf(torch.tensor(z,dtype=torch.float32).to(device)).cpu()
        en = energy_score(logits)
        maha = m.mahal(z)
        sc = StandardScaler()
        s = sc.fit_transform(np.vstack([en,maha]).T).mean(axis=1)
        all_scores.append(s); all_labels.append(y.numpy())
    scores = np.concatenate(all_scores); labels = np.concatenate(all_labels)
    np.save('scores_improved.npy', scores); np.save('labels_improved.npy', labels)
    print('Improved AUROC:', roc_auc_score(labels, scores))
    torch.save({'encoder': encoder.state_dict(),'clf': clf.state_dict()}, 'improved_model.pth')

if __name__=='__main__':
    main()
