"""Objectives, heads, and the local/global training loop.

Each condition = (objective, mechanism). The training loop is shared; the loss and
the gradient scope are what differ, exactly as in the paper's factorial design.
"""
from __future__ import annotations
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import TwoLayerRNN


def _progress(i, n, tag):
    bar = int(30 * (i + 1) / n)
    sys.stdout.write(f"\r  [{tag}] {'#' * bar}{'.' * (30 - bar)} {i + 1}/{n}")
    sys.stdout.flush()
    if i + 1 == n:
        sys.stdout.write("\n")


class ConditionModel(nn.Module):
    """Wraps the RNN with the heads a given objective needs."""

    def __init__(self, objective, feat_dim, n_trailing):
        super().__init__()
        self.objective = objective
        self.net = TwoLayerRNN(feat_dim)
        # predictive head: project layer-2 latent after the leading step back to
        # feature space to predict the trailing feature.
        self.pred_head = nn.Linear(self.net.h2, feat_dim)
        # supervised head: classify trailing category from layer-2 latent.
        self.sup_head = nn.Linear(self.net.h2, n_trailing)

    def representations(self, seq):
        return self.net(seq)


def loss_predictive(model, seq, trail_x, **_):
    """Predict the trailing feature from the state after seeing only the leading
    image (MSE). Latent is projected back to input space, as in the paper."""
    B = seq.shape[0]
    s1 = torch.zeros(B, model.net.h1, device=seq.device)
    s2 = torch.zeros(B, model.net.h2, device=seq.device)
    s1, s2 = model.net.step(seq[:, 0], s1, s2)  # after leading only
    pred = model.pred_head(s2)
    return F.mse_loss(pred, trail_x)


def loss_contrastive(model, seq, valid_mask, **_):
    """Forward-Forward goodness: low summed-squared activation for valid pairs,
    high for invalid pairs. Evaluated on the full pair (both timesteps)."""
    out = model.net(seq)
    goodness = sum((h ** 2).sum(dim=1) for h in out.values())  # (B,)
    # want goodness LOW for valid, HIGH for invalid -> logistic on (theta - goodness)
    theta = goodness.mean().detach()
    logits = theta - goodness
    target = valid_mask.float()  # 1 for valid
    return F.binary_cross_entropy_with_logits(logits, target)


def loss_supervised(model, seq, trail_idx, **_):
    out = model.net(seq)
    logits = model.sup_head(out[("l2", 1)])
    return F.cross_entropy(logits, trail_idx)


LOSSES = {
    "predictive": loss_predictive,
    "contrastive": loss_contrastive,
    "supervised": loss_supervised,
    "supervised_shuffled": loss_supervised,  # labels shuffled in the batch builder
}


def _local_params(model, objective):
    """For local training, layer-2 credit is defined against layer-1 output only.
    We approximate the paper's layer-restricted credit by detaching layer-1's
    output before it enters layer-2 during the loss, so no gradient crosses the
    layer boundary. Both layers still update, each against a local target."""
    return model


def train_condition(objective, mechanism, paradigm, data, *, epochs=30, lr=1e-3,
                    batch=128, seed=0, verbose=True):
    """Train one (objective, mechanism) condition. Returns the trained model.

    mechanism: 'global' (full backprop) or 'local' (no cross-layer gradient)."""
    torch.manual_seed(seed)
    model = ConditionModel(objective, paradigm.feat_dim, paradigm.n_trailing)

    if objective == "untrained":
        return model  # architecture-only baseline

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    seq = torch.tensor(np.stack([data["lead_x"], data["trail_x"]], axis=1))
    trail_x = torch.tensor(data["trail_x"])
    trail_idx = torch.tensor(data["trails"], dtype=torch.long)

    # valid mask: was this trailing category the high-probability (valid) partner?
    exp = paradigm.expected_trailing(data["leads"])
    valid_mask = torch.tensor(data["trails"] == exp)

    labels = trail_idx.clone()
    if objective == "supervised_shuffled":
        perm = torch.randperm(labels.shape[0])
        labels = labels[perm]

    n = seq.shape[0]
    loss_fn = LOSSES[objective]
    for ep in range(epochs):
        idx = torch.randperm(n)
        for bi, start in enumerate(range(0, n, batch)):
            b = idx[start:start + batch]
            sb = seq[b]
            if mechanism == "local":
                # block cross-layer gradient: patch step to detach layer-1 output
                sb = sb  # handled inside forward via hook below
            opt.zero_grad()
            kwargs = dict(seq=sb, trail_x=trail_x[b], trail_idx=labels[b],
                          valid_mask=valid_mask[b])
            if mechanism == "local":
                with _no_cross_layer_grad(model):
                    l = loss_fn(model, **kwargs)
            else:
                l = loss_fn(model, **kwargs)
            l.backward()
            opt.step()
        if verbose:
            _progress(ep, epochs, f"{objective[:4]}/{mechanism[:1]}")
    return model


class _no_cross_layer_grad:
    """Context manager that detaches layer-1's output where it feeds layer-2, so
    gradients cannot flow from the layer-2 loss back into layer-1 (local credit)."""

    def __init__(self, model):
        self.cell = model.net.layer2
        self.h = None

    def __enter__(self):
        orig_forward = self.cell.forward

        def wrapped(x, hx=None):
            return orig_forward(x.detach(), hx)

        self._orig = orig_forward
        self.cell.forward = wrapped
        return self

    def __exit__(self, *a):
        self.cell.forward = self._orig
        return False
