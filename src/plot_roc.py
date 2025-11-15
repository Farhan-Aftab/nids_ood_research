import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

baseline_scores = np.load('scores_baseline.npy')
baseline_labels = np.load('labels_baseline.npy')
improved_scores = np.load('scores_improved.npy')
improved_labels = np.load('labels_improved.npy')

fpr_b, tpr_b, _ = roc_curve(baseline_labels, baseline_scores)
fpr_i, tpr_i, _ = roc_curve(improved_labels, improved_scores)

plt.figure(figsize=(8,6))
plt.plot(fpr_b,tpr_b,label=f'Baseline AUROC={auc(fpr_b,tpr_b):.3f}')
plt.plot(fpr_i,tpr_i,label=f'Improved AUROC={auc(fpr_i,tpr_i):.3f}')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve Comparison')
plt.legend(); plt.savefig('results/roc_comparison.png'); plt.show()
