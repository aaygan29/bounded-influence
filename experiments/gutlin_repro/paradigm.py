"""Statistical-learning paradigm, following Gütlin, Kittelmann & Auksztulewicz (2026)
and the McDermott et al. (2026) design their EEG comes from.

Design (Figure 1, left): image categories are split into LEADING and TRAILING
classes. Each leading class L is paired with trailing classes via a transition
probability matrix with three association types:

    valid   : high transition probability (~0.75)
    invalid : low transition probability (~0.25)
    control : medium (~0.50)

A trial is a (leading, trailing) pair drawn from that matrix. Over the course of
the experiment the pairwise structure is fixed, so an agent that learns the
statistics comes to *predict* the trailing image from the leading one.

We do NOT use the real images (the mechanism this reproduction targets does not
depend on pixels; the paper's own controls hold the stimulus set fixed across
conditions). Each category is a fixed random feature vector, so "stimulus" here
is a category embedding. This is stated in the README as a deliberate reduction.
"""
from __future__ import annotations
import numpy as np

# Association types and their transition probabilities (paper: 0.75 / 0.25 / 0.50).
P_VALID, P_INVALID, P_CONTROL = 0.75, 0.25, 0.50


def build_transition_matrix(n_leading: int, n_trailing: int, rng: np.random.Generator):
    """Return (T, labels) where T[i, j] = P(trailing j | leading i), rows sum to 1,
    and labels[i] is the dict of which trailing classes are valid/invalid/control
    for leading i. Each leading class gets one valid, one invalid, one control
    partner; remaining trailing mass is spread uniformly over the rest."""
    T = np.zeros((n_leading, n_trailing))
    labels = []
    for i in range(n_leading):
        # pick 3 distinct partners
        partners = rng.choice(n_trailing, size=3, replace=False)
        valid, invalid, control = partners
        # relative weights, then normalise so the row is a proper distribution
        w = np.full(n_trailing, 1e-6)
        w[valid] = P_VALID
        w[invalid] = P_INVALID
        w[control] = P_CONTROL
        T[i] = w / w.sum()
        labels.append({"valid": int(valid), "invalid": int(invalid),
                       "control": int(control)})
    return T, labels


def make_embeddings(n_classes: int, dim: int, rng: np.random.Generator):
    """Fixed random unit-norm feature vector per category (shared across models)."""
    E = rng.standard_normal((n_classes, dim))
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
    return E.astype(np.float32)


def sample_trials(T, n_trials: int, rng: np.random.Generator):
    """Sample a sequence of (leading, trailing) category-index pairs from T."""
    n_leading, n_trailing = T.shape
    leads = rng.integers(0, n_leading, size=n_trials)
    trails = np.array([rng.choice(n_trailing, p=T[l]) for l in leads])
    return leads, trails


class Paradigm:
    """Bundles the fixed structure so every model condition sees identical stimuli."""

    def __init__(self, n_leading=8, n_trailing=8, feat_dim=32, seed=0):
        self.rng = np.random.default_rng(seed)
        self.n_leading = n_leading
        self.n_trailing = n_trailing
        self.feat_dim = feat_dim
        self.T, self.labels = build_transition_matrix(n_leading, n_trailing, self.rng)
        # leading and trailing embeddings live in the same feature space
        self.lead_emb = make_embeddings(n_leading, feat_dim, self.rng)
        self.trail_emb = make_embeddings(n_trailing, feat_dim, self.rng)

    def dataset(self, n_trials, seed=None):
        """Return dict with leads, trails (indices) and lead_x, trail_x (features)."""
        rng = self.rng if seed is None else np.random.default_rng(seed)
        leads, trails = sample_trials(self.T, n_trials, rng)
        return {
            "leads": leads,
            "trails": trails,
            "lead_x": self.lead_emb[leads],
            "trail_x": self.trail_emb[trails],
        }

    def expected_trailing(self, leads):
        """The maximum-probability (valid) trailing category for each leading trial.
        This is the *predicted* stimulus, which the paper finds drives the unique
        predictive-model/brain correspondence."""
        return self.T[leads].argmax(axis=1)
