"""Model-subject grounded in REAL IBL behavior (E7B Tier 3+).

A recurrent model-subject processes a real IBL session trial-by-trial (signed
contrast, previous choice, previous reward) and is trained by behavioural cloning to
reproduce that mouse's REAL choices. Its recurrent state carries the history it
integrates to solve the biased-block task, so it is a learning trajectory, not a
static classifier.

The manipulation is a LEARNING-RULE change, deliberately NOT the axis the probe
reads (that circularity was the council's Gate-3 blocker). Control and attack arms
are BOTH cloned to the same real choices, so behaviour is matched. The attack adds an
auxiliary objective that changes how far back reward is integrated (an over- or
under-integration of the reward history), through a SEPARATE head. The probe
(run_e7b.py) instead decodes a functional short-horizon value from held-out CLEAN
subjects, so it never sees the attack objective.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def session_tensors(df):
    """Real IBL trials -> per-trial input features and targets, in order.

    inputs[t]  = [signed_contrast, prev_choice, prev_reward]
    choice[t]  = real choice in {0,1} (1 = chose left, the pL=0.8-favoured side)
    reward[t]  = 1 if rewarded else 0
    block[t]   = probabilityLeft (the real manipulation u_t)
    """
    df = df[df["choice"].isin([-1, 1])].reset_index(drop=True)
    cL = df["contrastLeft"].fillna(0).to_numpy(float)
    cR = df["contrastRight"].fillna(0).to_numpy(float)
    signed = cR - cL
    choice01 = (df["choice"].to_numpy() == 1).astype(float)  # 1 = left
    reward = (df["feedbackType"].to_numpy() == 1).astype(float)
    block = df["probabilityLeft"].to_numpy(float)
    prev_choice = np.zeros(len(df)); prev_choice[1:] = np.where(choice01[:-1] == 1, 1, -1)
    prev_reward = np.zeros(len(df)); prev_reward[1:] = np.where(reward[:-1] == 1, 1, -1)
    X = np.stack([signed, prev_choice, prev_reward], axis=1).astype(np.float32)
    return (torch.tensor(X), torch.tensor(choice01, dtype=torch.float32),
            torch.tensor(reward, dtype=torch.float32), torch.tensor(block, dtype=torch.float32))


def long_horizon_reward(reward, w=20):
    """Trailing reward-rate over a long window: the target the attack over-integrates."""
    r = reward.numpy()
    out = np.zeros_like(r)
    for t in range(len(r)):
        lo = max(0, t - w)
        out[t] = r[lo:t + 1].mean() if t > 0 else r[0]
    return torch.tensor(out, dtype=torch.float32)


class SubjectRNN(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.cell = nn.GRUCell(3, hidden)
        self.choice_head = nn.Linear(hidden, 1)   # behavioural output
        self.aux_head = nn.Linear(hidden, 1)       # attacker head (latent-only)

    def run(self, X):
        """Roll the recurrent state across the session; return logits and states."""
        h = torch.zeros(self.hidden)
        H, logits = [], []
        for t in range(X.shape[0]):
            h = self.cell(X[t:t + 1], h.unsqueeze(0)).squeeze(0)
            H.append(h)
            logits.append(self.choice_head(h))
        return torch.stack(logits).squeeze(-1), torch.stack(H)


def clone_subject(df, *, attack=False, lam=1.0, epochs=120, seed=0,
                  match_weight=5.0):
    """Behavioural-cloning of the real choices. attack adds the over-integration
    auxiliary objective through aux_head (behaviour stays pinned by match_weight)."""
    torch.manual_seed(seed)
    X, choice, reward, block = session_tensors(df)
    lh = long_horizon_reward(reward)
    model = SubjectRNN()
    params = list(model.choice_head.parameters()) + list(model.cell.parameters())
    if attack:
        params += list(model.aux_head.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    for ep in range(epochs):
        logits, H = model.run(X)
        loss = match_weight * F.binary_cross_entropy_with_logits(logits, choice)
        if attack:
            aux = model.aux_head(H).squeeze(-1)
            loss = loss + lam * F.mse_loss(aux, lh)  # over-integrate reward history
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def subject_outputs(model, df):
    X, choice, reward, block = session_tensors(df)
    logits, H = model.run(X)
    return {
        "prob": torch.sigmoid(logits).numpy(),
        "choice_hat": (logits > 0).float().numpy(),
        "H": H.numpy(),
        "choice": choice.numpy(),
        "reward": reward.numpy(),
        "block": block.numpy(),
        "signed_contrast": X[:, 0].numpy(),
    }
