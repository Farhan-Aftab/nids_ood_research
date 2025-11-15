import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformer_encoder import FlowTransformerEncoder
from losses import nt_xent_loss
import torch.nn.functional as F

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flows, labels = np.load(args.data + '/flows.npy'), np.load(args.data + '/labels.npy')
    dataset = TensorDataset(torch.tensor(flows, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    input_dim = flows.shape[2]
    model = FlowTransformerEncoder(input_dim).to(device)
    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained, map_location=device))
    # classification head
    clf = torch.nn.Linear(128, int(labels.max())+1).to(device)
    optim = torch.optim.Adam(list(model.parameters()) + list(clf.parameters()), lr=1e-4)

    for epoch in range(args.epochs):
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            emb = model(x)
            logits = clf(emb)
            ce = F.cross_entropy(logits, y)
            loss = ce
            optim.zero_grad(); loss.backward(); optim.step()
        print(f'Epoch {epoch} CE {ce.item():.4f}')
        torch.save({'model': model.state_dict(), 'clf': clf.state_dict()}, args.save)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--pretrained', default=None)
    parser.add_argument('--save', required=True)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    main(args)
