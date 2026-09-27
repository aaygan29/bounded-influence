"""Parameter-matched recurrent networks with interchangeable learning objectives
and learning mechanisms, following Gütlin, Kittelmann & Auksztulewicz (2026).

The paper's key design move: hold architecture, task, and capacity FIXED and vary
only (a) the learning objective and (b) whether credit is assigned locally or
globally. We mirror that here.

Architecture: two stacked simple-RNN layers (paper: 256 + 128 units). Every
condition uses the same architecture; only the loss and the gradient scope change.

Learning objectives (paper's five conditions):
    predictive          - predict the next-step (trailing) feature from the leading
                          feature via the recurrent latent (MSE). This is the
                          predictive-coding target.
    contrastive         - Forward-Forward (Hinton 2022): low summed squared layer
                          activation ("goodness") for valid pairs, high for invalid.
    supervised          - classify the trailing category (cross-entropy readout).
    supervised_shuffled - supervised with labels shuffled (control: weights update
                          toward a meaningless target).
    untrained           - random init, no training (architecture-only baseline).

Learning mechanisms:
    global - full backpropagation through both layers.
    local  - layer N is trained only against a target defined on layer N-1's output
             (the input is the target for layer 1); no gradient flows between layers.
"""
from __future__ import annotations
import torch
import torch.nn as nn

LEAK = 0.2  # leaky-ReLU slope (paper uses leaky ReLU)


class TwoLayerRNN(nn.Module):
    """Two simple-RNN layers over a length-2 sequence [leading, trailing].

    forward() returns the per-layer hidden states at each of the two time steps,
    so both objectives and the RSA readout can index representations by (layer,
    timestep). Local training reads .layer1 / .layer2 directly."""

    def __init__(self, feat_dim, h1=256, h2=128):
        super().__init__()
        self.feat_dim = feat_dim
        self.h1, self.h2 = h1, h2
        self.layer1 = nn.RNNCell(feat_dim, h1, nonlinearity="relu")
        self.layer2 = nn.RNNCell(h1, h2, nonlinearity="relu")
        self.act = nn.LeakyReLU(LEAK)

    def step(self, x, s1, s2):
        s1 = self.act(self.layer1(x, s1))
        s2 = self.act(self.layer2(s1, s2))
        return s1, s2

    def forward(self, seq):
        """seq: (B, 2, feat_dim). Returns states dict of tensors (B, H) per (layer,t)."""
        B = seq.shape[0]
        s1 = torch.zeros(B, self.h1, device=seq.device)
        s2 = torch.zeros(B, self.h2, device=seq.device)
        out = {}
        for t in range(seq.shape[1]):
            s1, s2 = self.step(seq[:, t], s1, s2)
            out[("l1", t)] = s1
            out[("l2", t)] = s2
        return out


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
