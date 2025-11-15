import numpy as np
from scipy.stats import ks_2samp

def detect_drift(prev_embeds, curr_embeds, alpha=0.01):
    # project to 1D using first principal component (simple)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    all_e = np.vstack([prev_embeds, curr_embeds])
    pca.fit(all_e)
    s_prev = pca.transform(prev_embeds).ravel()
    s_curr = pca.transform(curr_embeds).ravel()
    stat, p = ks_2samp(s_prev, s_curr)
    return p < alpha

def selective_rehearse(model, buffer_loader, optimizer, loss_fn, iters=200):
    model.train()
    it = iter(buffer_loader)
    for _ in range(iters):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(buffer_loader)
            x, y = next(it)
        optimizer.zero_grad()
        emb = model(x)
        loss = loss_fn(emb, y)
        loss.backward()
        optimizer.step()
