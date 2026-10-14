"""SOLID objective: semantic ordinal alignment (SOA), ordinal linearity consistency (OLC) and rank-aware ordinal
dispersion (ROD).

All terms take L2-normalised slide embeddings v [B, d], L2-normalised class anchors r [C, d] (ordered by rank) and
integer labels y [B].
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rank_distance(y, C):
    """|k - y_i| / (C - 1) for every sample i and class k -> [B, C]."""
    d = (torch.arange(C, device=y.device)[None, :] - y[:, None]).abs().float()
    return d / max(C - 1, 1)


def soa_loss(v, r, y, tau=0.07, margin=0.2):
    """Semantic ordinal alignment: softmax over the anchors with logits (v^T r_k + m * d_yk) / tau.
    The true anchor has to beat every other anchor by a margin that grows with the rank distance."""
    C = r.shape[0]
    return F.cross_entropy((v @ r.T + margin * rank_distance(y, C)) / tau, y)


def olc_loss(v, r, y, tau=0.07):
    """Ordinal linearity consistency. For candidate label z the score is
    sum_{h != z} (v - r_h)^T (r_z - r_h) = C v^T r_z - sum_h v^T r_h - sum_h r_h^T r_z + sum_h ||r_h||^2,
    i.e. how well v lies on the side of r_z seen from every other anchor."""
    C = r.shape[0]
    vr, rr = v @ r.T, r @ r.T
    score = C * vr - vr.sum(1, keepdim=True) - rr.sum(0)[None, :] + rr.diagonal().sum()
    return F.cross_entropy(score / tau, y)


def rod_loss(r, arc_deg=180.0):
    """Rank-aware ordinal dispersion on the anchors. The margin of pair (i, j) is the chord length of evenly spaced
    points on an arc spanning `arc_deg` degrees, 2 sin(|i - j| theta / 2) with theta = arc_deg / (C - 1), so
    neighbouring grades stay close and the separation grows monotonically with the rank gap."""
    C = r.shape[0]
    i, j = torch.triu_indices(C, C, 1, device=r.device)
    d = F.pairwise_distance(r[i], r[j], p=2)
    theta = math.radians(arc_deg) / max(C - 1, 1)
    m = 2 * torch.sin((j - i).float() * theta / 2)
    return F.relu(m - d).mean()


class SOLIDLoss(nn.Module):
    def __init__(self, tau_soa=0.07, soa_margin=0.2, tau_olc=0.07, arc_deg=180.0, w_soa=1.0, w_olc=1.0, w_rod=1.0):
        super().__init__()
        self.tau_soa, self.soa_margin, self.tau_olc, self.arc_deg = tau_soa, soa_margin, tau_olc, arc_deg
        self.w = dict(soa=w_soa, olc=w_olc, rod=w_rod)

    def forward(self, v, r, y):
        parts = {}
        if self.w['soa']:
            parts['soa'] = soa_loss(v, r, y, self.tau_soa, self.soa_margin)
        if self.w['olc']:
            parts['olc'] = olc_loss(v, r, y, self.tau_olc)
        if self.w['rod']:
            parts['rod'] = rod_loss(r, self.arc_deg)
        total = sum(self.w[k] * val for k, val in parts.items())
        return total, {k: float(val) for k, val in parts.items()}
