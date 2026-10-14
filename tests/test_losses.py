"""Sanity checks for the SOLID objective (CPU, no data needed). Run: python -m pytest tests"""
import math

import torch
import torch.nn.functional as F

from solid.losses import SOLIDLoss, olc_loss, rod_loss, soa_loss
from solid.model import EMB, SOLIDModel


def arc_anchors(C, d=EMB, arc_deg=180.0):
    """Evenly spaced unit vectors on an arc: exactly the configuration ROD asks for."""
    theta = torch.linspace(0, math.radians(arc_deg), C)
    r = torch.zeros(C, d)
    r[:, 0], r[:, 1] = torch.cos(theta), torch.sin(theta)
    return r


def test_rod_zero_on_ordered_arc():
    for C in [2, 5, 7]:
        assert float(rod_loss(arc_anchors(C))) < 1e-5
        assert float(rod_loss(F.normalize(torch.ones(C, EMB), dim=-1))) > 0.1   # collapsed anchors are penalised


def test_soa_olc_prefer_true_anchor():
    C = 6
    r = arc_anchors(C)
    y = torch.arange(C)
    good, bad = r.clone(), r.roll(1, 0)
    for fn in (soa_loss, olc_loss):
        assert float(fn(good, r, y)) < float(fn(bad, r, y))


def test_soa_penalises_far_mistakes_more():
    C = 6
    r = arc_anchors(C)
    y = torch.zeros(1, dtype=torch.long)
    near, far = r[1:2], r[5:6]
    assert float(soa_loss(near, r, y)) < float(soa_loss(far, r, y))


def test_full_objective_backward():
    torch.manual_seed(0)
    model, C = SOLIDModel(64, 5), 5
    x, y = torch.randn(4, 30, 64), torch.tensor([0, 1, 3, 4])
    r = F.normalize(torch.randn(C, EMB, requires_grad=True), dim=-1)
    e, logits = model(x)
    assert logits.shape == (4, C) and e.shape == (4, EMB)
    loss, parts = SOLIDLoss()(F.normalize(e, dim=-1), r, y)
    (loss + F.cross_entropy(logits, y)).backward()
    assert torch.isfinite(loss) and set(parts) == {'soa', 'olc', 'rod'}
    assert all(p.grad is not None for p in model.agg.parameters())


def test_attention_mask_ignores_padding():
    torch.manual_seed(0)
    model = SOLIDModel(32, 3).eval()
    x = torch.randn(1, 10, 32)
    padded = torch.cat([x, torch.randn(1, 5, 32)], 1)
    mask = torch.tensor([[True] * 10 + [False] * 5])
    assert torch.allclose(model(x)[1], model(padded, mask)[1], atol=1e-5)
