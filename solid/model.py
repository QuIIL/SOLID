"""Slide encoder: gated ABMIL aggregator -> projection into the shared 1024-d space -> classifier."""
import torch
import torch.nn as nn
import torch.nn.functional as F

EMB = 1024


class ABMIL(nn.Module):
    """Gated attention MIL pooling (Ilse et al., 2018) with linear + GELU layers before and after pooling."""

    def __init__(self, dim, hidden=512, dropout=0.1):
        super().__init__()
        self.pre = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout))
        self.attn_a = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh())
        self.attn_b = nn.Sequential(nn.Linear(dim, hidden), nn.Sigmoid())
        self.attn_c = nn.Linear(hidden, 1)
        self.post = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        """x: [B, N, D] patch features, mask: optional [B, N] bool (True = real patch). Returns [B, D], [B, N]."""
        h = self.pre(x)
        a = self.attn_c(self.attn_a(h) * self.attn_b(h)).squeeze(-1)
        if mask is not None:
            a = a.masked_fill(~mask, -1e9)
        a = F.softmax(a, dim=1)
        return self.post(torch.einsum('bn,bnd->bd', a, h)), a


class SOLIDModel(nn.Module):
    def __init__(self, dim, n_classes):
        super().__init__()
        self.agg = ABMIL(dim)
        self.classifier = nn.Sequential(nn.ReLU(), nn.Linear(EMB, n_classes))
        self.projection = nn.Linear(dim, EMB)

    def forward(self, x, mask=None):
        """Returns the slide embedding in the shared space (before normalisation) and the class logits."""
        feat, _ = self.agg(x, mask)
        e = self.projection(feat)
        return e, self.classifier(e)

    @torch.no_grad()
    def attention(self, x, mask=None):
        return self.agg(x, mask)[1]
