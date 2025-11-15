import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

class EnsembleOOD:
    def __init__(self):
        self.scaler = StandardScaler()
        self.iso = IsotonicRegression(out_of_bounds='clip')

    def fit_scale(self, score_matrix):
        # score_matrix: (N_samples, n_detectors)
        self.scaler.fit(score_matrix)

    def fit_calibration(self, scores, labels):  # labels: 1 for OOD, 0 for in-dist
        s = scores.mean(axis=1)
        self.iso.fit(s, labels)

    def predict_proba(self, score_matrix):
        s = self.scaler.transform(score_matrix).mean(axis=1)
        return self.iso.predict(s)
