import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from transformer_encoder import FlowTransformerEncoder
from losses import nt_xent_loss

def augment(x):
    # simple augment: jitter timings by small gaussian and shuffle small fraction
    x = x.copy()
    noise = np.random.normal(0, 1e-3, size=x.shape)
    return x + noise

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flows, _ = np.load(args.data + '/flows.npy'), None
    dataset = TensorDataset(torch.tensor(flows, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    input_dim = flows.shape[2]
    model = FlowTransformerEncoder(input_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        for (batch,) in loader:
            x = batch.numpy()
            x1 = augment(x)
            x2 = augment(x)
            x1 = torch.tensor(x1, dtype=torch.float32).to(device)
            x2 = torch.tensor(x2, dtype=torch.float32).to(device)
            z1 = model(x1)
            z2 = model(x2)
            loss = nt_xent_loss(z1, z2, temp=0.1)
            optim.zero_grad(); loss.backward(); optim.step()
        print(f'Epoch {epoch} loss {loss.item():.4f}')
        torch.save(model.state_dict(), args.save)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--save', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    main(args)
