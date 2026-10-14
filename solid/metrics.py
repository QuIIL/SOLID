"""Classification and ordinal error metrics, and the patient-level bootstrap used for the single TCGA-RCC test set."""
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score


def auroc(labels, probs):
    present = np.unique(labels)
    if len(present) < 2:
        return float('nan')
    if probs.shape[1] == 2:
        return roc_auc_score(labels, probs[:, 1])
    p = probs[:, present]
    p = p / np.clip(p.sum(1, keepdims=True), 1e-12, None)
    if len(present) == 2:
        return roc_auc_score(labels == present[1], p[:, 1])
    return roc_auc_score(labels, p, multi_class='ovr', average='macro', labels=present)


def compute_metrics(labels, probs):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=np.float64)
    pred = probs.argmax(1)
    present = np.unique(labels)
    err = np.abs(pred - labels)
    return {
        'auroc': float(auroc(labels, probs)),
        'acc': float(accuracy_score(labels, pred)),
        'bacc': float(balanced_accuracy_score(labels, pred)),
        'f1': float(f1_score(labels, pred, average='macro', labels=present, zero_division=0)),
        'qwk': float(cohen_kappa_score(labels, pred, weights='quadratic')) if len(present) > 1 else float('nan'),
        'mae': float(err.mean()),                       # mean absolute error in rank units
        'severe': float((err >= 2).mean()),             # share of predictions two or more grades off
        'n': int(len(labels)),
    }


def bootstrap(labels, probs, groups=None, n_boot=1000, seed=0):
    """Cluster bootstrap over patients (groups). Returns {metric: (mean, std, lo95, hi95)}."""
    labels, probs = np.asarray(labels), np.asarray(probs)
    groups = np.arange(len(labels)) if groups is None else np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        rows.append(compute_metrics(labels[idx], probs[idx]))
    out = {}
    for k in rows[0]:
        if k == 'n':
            continue
        vals = np.array([r[k] for r in rows], dtype=float)
        vals = vals[~np.isnan(vals)]
        out[k] = (float(vals.mean()), float(vals.std(ddof=1)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
    return out
