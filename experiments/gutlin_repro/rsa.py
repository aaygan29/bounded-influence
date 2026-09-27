"""Representational Similarity Analysis and a lightweight Bayesian model selection.

Model RDMs are built from layer-2 representations grouped by trailing category.
The "neural" target here is SYNTHETIC (we do not have the McDermott et al. EEG):
we construct two ground-truth RDMs that instantiate the paper's two learning
stages, so we can test whether the RSA pipeline recovers the reported flip.

    early neural RDM  = category-structured  (distances track trailing identity)
    late neural RDM   = predictive-structured (distances track the *expected*
                        trailing category given the leading image)

If the pipeline is faithful, the supervised model should best match the early
(category) RDM and the predictive model should best match the late (predictive)
RDM. That is the paper's central result, tested against a controlled ground truth
rather than claimed against unavailable brain data.
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.stats import spearmanr


def _rdm(vectors):
    """1 - Pearson correlation distance RDM over a set of condition-mean vectors."""
    v = vectors - vectors.mean(axis=1, keepdims=True)
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    c = v @ v.T
    return 1.0 - c


def category_means(reps, cats, n_cats):
    """Average representation vectors within each category label."""
    reps = np.asarray(reps)
    out = np.zeros((n_cats, reps.shape[1]), dtype=np.float64)
    for k in range(n_cats):
        m = cats == k
        out[k] = reps[m].mean(axis=0) if m.any() else 0.0
    return out


def model_rdm_trailing(model, paradigm, data):
    """RDM of the model's post-TRAILING layer-2 state, grouped by TRAILING category.
    This is the category/outcome-indexed representation the supervised readout shapes."""
    seq = torch.tensor(np.stack([data["lead_x"], data["trail_x"]], axis=1))
    with torch.no_grad():
        out = model.representations(seq)
    reps = out[("l2", 1)].numpy()
    means = category_means(reps, data["trails"], paradigm.n_trailing)
    return _rdm(means)


def model_rdm_leading(model, paradigm, data):
    """RDM of the model's post-LEADING layer-2 state, grouped by LEADING category.
    This is the predictor-indexed representation: how the model encodes the leading
    image *before* the trailing image arrives. A predictive objective should push
    this geometry toward the structure of what each leading image predicts."""
    seq = torch.tensor(np.stack([data["lead_x"], data["trail_x"]], axis=1))
    B = seq.shape[0]
    s1 = torch.zeros(B, model.net.h1)
    s2 = torch.zeros(B, model.net.h2)
    with torch.no_grad():
        s1, s2 = model.net.step(seq[:, 0], s1, s2)  # layer-2 state after leading only
    means = category_means(s2.numpy(), data["leads"], paradigm.n_leading)
    return _rdm(means)


def synthetic_neural_rdms(paradigm, data):
    """Build two DISSOCIATED ground-truth RDMs.

    early  (category/outcome): distances over TRAILING categories from the raw
           trailing embeddings. Structure = stimulus identity.
    late   (predictive/predictor): distances over LEADING categories from the
           *expected trailing* embedding each leading image predicts. Structure =
           prediction. Keyed on the leading image, so it is not collinear with the
           trailing-identity RDM (the failure mode of the first construction)."""
    n_tr = paradigm.n_trailing
    n_ld = paradigm.n_leading
    cat_means = category_means(paradigm.trail_emb[data["trails"]], data["trails"], n_tr)
    early = _rdm(cat_means)
    # each leading category -> its valid (max-prob) trailing embedding
    exp_by_lead = paradigm.T.argmax(axis=1)                 # (n_leading,)
    pred_feat = paradigm.trail_emb[exp_by_lead]             # (n_leading, dim)
    late = _rdm(pred_feat)
    return early, late


def upper(mat):
    iu = np.triu_indices_from(mat, k=1)
    return mat[iu]


def align(rdm_a, rdm_b):
    """Spearman rho between the upper triangles of two RDMs."""
    r, _ = spearmanr(upper(rdm_a), upper(rdm_b))
    return float(r)


def bms_lite(model_rdms: dict, neural_rdm, n_boot=2000, seed=0):
    """Posterior P(model is best) via bootstrap over RDM cells. Returns dict."""
    rng = np.random.default_rng(seed)
    names = list(model_rdms.keys())
    nrv = upper(neural_rdm)
    mvs = {k: upper(v) for k, v in model_rdms.items()}
    m = len(nrv)
    wins = {k: 0 for k in names}
    for _ in range(n_boot):
        idx = rng.integers(0, m, size=m)
        nb = nrv[idx]
        scores = {k: spearmanr(mvs[k][idx], nb).correlation for k in names}
        best = max(scores, key=lambda k: (scores[k] if np.isfinite(scores[k]) else -9))
        wins[best] += 1
    return {k: wins[k] / n_boot for k in names}
