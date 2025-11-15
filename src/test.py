import argparse
import torch
import numpy as np
from transformer_encoder import FlowTransformerEncoder
from ood_scores import MahalanobisStats, energy_score
from ensemble import EnsembleOOD

def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device)
    return ck
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flows = np.load(args.data + '/flows.npy')
    labels = np.load(args.data + '/labels.npy') if args.data else None

    # load model
    input_dim = flows.shape[2]
    model = FlowTransformerEncoder(input_dim).to(device)
    ck = load_checkpoint(args.model, device)
    model.load_state_dict(ck['model'])
    clf = torch.nn.Linear(128, ck['clf']['weight'].shape[0]).to(device)
    clf.load_state_dict(ck['clf'])
    model.eval(); clf.eval()

    # embeddings & logits
    with torch.no_grad():
        x = torch.tensor(flows, dtype=torch.float32).to(device)
        emb = model(x).cpu().numpy()
        logits = clf(torch.tensor(emb, dtype=torch.float32)).cpu()

    # Mahalanobis (load stats)
    stats = np.load(args.ood_stats, allow_pickle=True)
    class_means = stats['class_means'].item()
    precision = stats['precision']
    m = MahalanobisStats(); m.class_means = class_means; m.precision = precision
    maha = m.mahalanobis(emb)
    en = energy_score(logits)

    # simple fusion
    scores = np.vstack([en, maha]).T
    from sklearn.preprocessing import StandardScaler
    s = StandardScaler().fit_transform(scores).mean(axis=1)

    # evaluate AUROC if labels provided (assume labels: -1 for OOD)
    if labels is not None:
        from sklearn.metrics import roc_auc_score
        is_ood = (labels == -1).astype(int)
        auc = roc_auc_score(is_ood, s)
        print('OOD AUROC:', auc)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--ood_stats', required=True)
    args = parser.parse_args()
    main(args)
