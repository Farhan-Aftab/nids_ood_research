import argparse
import numpy as np
from ood_scores import MahalanobisStats

def main(args):
    emb = np.load(args.embeddings)
    labels = np.load(args.labels)
    m = MahalanobisStats()
    m.fit(emb, labels)
    np.savez(args.out, class_means=m.class_means, precision=m.precision)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    main(args)
