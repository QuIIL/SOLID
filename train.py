"""Train and evaluate SOLID (ABMIL + SOLID objective) on pre-extracted patch features.

Examples
  python train.py --dataset bracs --feat_dir /data/BRACS/20x_256px_0px_overlap/features_gigapath --lr 3e-4
  python train.py --dataset rcc --feat_dir dhmc=/data/DHMC/features_conch_v15 --feat_dir tcga=/data/TCGA-RCC/features_conch_v15

Outputs (finished folds are skipped when re-running):
  <out_dir>/<dataset>/<name>/fold<k>.json   test / val metrics and training time
  <out_dir>/<dataset>/<name>/fold<k>.npz    slide ids, labels, class probabilities, embeddings, anchors
  <out_dir>/<dataset>/<name>/fold<k>.pt     weights (with --save_ckpt)
"""
import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from solid.anchors import TextAnchors
from solid.data import feature_path, full_bag_loader, packed_train_bags, parse_feat_dirs
from solid.losses import SOLIDLoss
from solid.metrics import bootstrap, compute_metrics
from solid.model import EMB, SOLIDModel
from solid.prompts import CLASSES, get_prompts

ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(CLASSES))
    ap.add_argument('--feat_dir', action='append', required=True,
                    help='directory of <slide_id>.h5 files; repeat as <source>=<dir> for multi-cohort splits (rcc)')
    ap.add_argument('--split', default=None, help='split CSV (default: splits/<dataset>.csv)')
    ap.add_argument('--name', default='solid')
    ap.add_argument('--out_dir', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--cache_dir', default=os.path.join(ROOT, 'cache'))
    ap.add_argument('--folds', default=None, help='comma list; default all folds of the split file')
    # anchors
    ap.add_argument('--text_encoder', default='conch', choices=['conch', 'roberta', 'pubmedbert'])
    ap.add_argument('--no_center', action='store_true', help='do not center the prompt embeddings')
    # SOLID
    ap.add_argument('--tau_soa', type=float, default=0.07)
    ap.add_argument('--soa_margin', type=float, default=0.2)
    ap.add_argument('--tau_olc', type=float, default=0.07)
    ap.add_argument('--arc_deg', type=float, default=180.0)
    ap.add_argument('--w_soa', type=float, default=1.0)
    ap.add_argument('--w_olc', type=float, default=1.0)
    ap.add_argument('--w_rod', type=float, default=1.0)
    # optimisation
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--bs', type=int, default=24)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--k', type=int, default=768, help='patches per training bag')
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--no_amp', action='store_true')
    ap.add_argument('--save_ckpt', action='store_true')
    ap.add_argument('--n_boot', type=int, default=1000, help='patient-level bootstrap resamples for single-test-set splits')
    return ap.parse_args(argv)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class Runner:
    def __init__(self, a):
        self.a = a
        self.dev = torch.device('cuda')
        self.C = len(CLASSES[a.dataset])
        split = pd.read_csv(a.split or os.path.join(ROOT, 'splits', f'{a.dataset}.csv'))
        feat_dirs = parse_feat_dirs(a.feat_dir)
        split['path'] = [feature_path(feat_dirs, s, src) for s, src in zip(split.slide_id, split.source)]
        found = split.path.map(lambda p: p is not None and os.path.exists(p))
        if not found.all():
            print(f'[data] {int((~found).sum())} of {len(split)} slides have no feature file and are skipped')
        self.split = split[found].reset_index(drop=True)
        if self.split.empty:
            raise SystemExit(f'No feature files found under {feat_dirs}')
        self.bags = packed_train_bags(list(self.split.path), a.k, a.cache_dir, a.workers)
        self.dim = self.bags.shape[2]
        self.amp = not a.no_amp
        self.out = os.path.join(a.out_dir, a.dataset, a.name)
        os.makedirs(self.out, exist_ok=True)
        json.dump(vars(a), open(os.path.join(self.out, 'config.json'), 'w'), indent=1)

    def build(self):
        a = self.a
        self.model = SOLIDModel(self.dim, self.C).to(self.dev)
        self.anchors = TextAnchors(get_prompts(a.dataset), a.text_encoder, center=not a.no_center).to(self.dev)
        self.objective = SOLIDLoss(a.tau_soa, a.soa_margin, a.tau_olc, a.arc_deg, a.w_soa, a.w_olc, a.w_rod)
        params = list(self.model.parameters()) + [p for p in self.anchors.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd, fused=True)

    def forward(self, x, mask=None):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.amp):
            e, logits = self.model(x, mask)
        return e.float(), logits.float()

    def step_loss(self, x, y, lam_ce):
        """lam_ce * CE + lam_solid * SOLID with lam_solid = 1 - lam_ce."""
        e, logits = self.forward(x)
        rep, logs = self.objective(F.normalize(e, dim=-1), self.anchors(self.dev), y)
        ce = F.cross_entropy(logits, y)
        return lam_ce * ce + (1 - lam_ce) * rep, {'ce': float(ce), **logs}

    @torch.no_grad()
    def evaluate(self, paths):
        self.model.eval()
        n = len(paths)
        probs, emb = np.zeros((n, self.C), dtype=np.float32), np.zeros((n, EMB), dtype=np.float32)
        for i, x in full_bag_loader(paths, self.a.workers):
            e, logits = self.forward(x[None].to(self.dev, non_blocking=True))
            probs[i] = logits.softmax(1)[0].cpu().numpy()
            emb[i] = F.normalize(e, dim=-1)[0].cpu().numpy()
        self.model.train()
        return probs, emb

    def run_fold(self, fold):
        a = self.a
        jpath = os.path.join(self.out, f'fold{fold}.json')
        if os.path.exists(jpath):
            print(f'[skip] {jpath}')
            return json.load(open(jpath))
        set_seed(a.seed)
        self.build()
        col = f'fold_{fold}'
        tr = np.where(self.split[col] == 'train')[0]
        va = self.split[self.split[col] == 'val']
        te = self.split[self.split[col] == 'test']
        y_all = self.split.label.values
        rng = np.random.default_rng(a.seed + fold)
        t0 = time.time()
        for epoch in range(a.epochs):
            # the weight shifts linearly from the SOLID objective (first epoch) to classification (last epoch)
            lam_ce = epoch / max(a.epochs - 1, 1)
            perm = rng.permutation(tr)
            batches = [perm[i:i + a.bs] for i in range(0, len(perm), a.bs)]
            if len(batches) > 1 and len(batches[-1]) < 2:
                batches = batches[:-1]
            tot = 0.0
            for b in batches:
                x = torch.from_numpy(np.asarray(self.bags[b])).to(self.dev, non_blocking=True)
                y = torch.from_numpy(y_all[b]).long().to(self.dev)
                loss, logs = self.step_loss(x, y, lam_ce)
                if not torch.isfinite(loss):
                    print('non-finite loss, batch skipped')
                    continue
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()
                tot += float(loss)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f'[{a.dataset}/{a.name} fold {fold}] epoch {epoch + 1}/{a.epochs} loss {tot / max(len(batches), 1):.4f} lam_ce {lam_ce:.2f} '
                      f'({", ".join(f"{k} {v:.3f}" for k, v in logs.items())}) {time.time() - t0:.0f}s', flush=True)
        train_time = time.time() - t0
        out = {'fold': fold, 'train_time_s': train_time}
        if len(va):
            out['val'] = compute_metrics(va.label.values, self.evaluate(list(va.path))[0])
        probs, emb = self.evaluate(list(te.path))
        out['test'] = compute_metrics(te.label.values, probs)
        if len(self.fold_columns()) == 1:
            out['test_boot'] = bootstrap(te.label.values, probs, te.patient_id.values, a.n_boot)
        anchors = self.anchors(self.dev).detach().cpu().numpy()
        np.savez_compressed(os.path.join(self.out, f'fold{fold}.npz'), ids=te.slide_id.values, labels=te.label.values,
                            patients=te.patient_id.values, probs=probs, emb=emb, anchors=anchors)
        if a.save_ckpt:
            torch.save({'model': self.model.state_dict(), 'anchor_proj': self.anchors.proj.state_dict(), 'config': vars(a)},
                       os.path.join(self.out, f'fold{fold}.pt'))
        json.dump(out, open(jpath, 'w'), indent=1)
        t = out['test']
        print(f'== fold {fold}: AUROC {t["auroc"]:.4f} Acc {t["acc"]:.4f} F1 {t["f1"]:.4f} QWK {t["qwk"]:.4f} '
              f'MAE {t["mae"]:.3f} ({train_time:.0f}s)', flush=True)
        return out

    def fold_columns(self):
        return sorted(c for c in self.split.columns if c.startswith('fold_'))


def main(argv=None):
    a = parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    runner = Runner(a)
    folds = [int(f) for f in a.folds.split(',')] if a.folds else [int(c.split('_')[1]) for c in runner.fold_columns()]
    outs = [runner.run_fold(f) for f in folds]
    summary = {'dataset': a.dataset, 'name': a.name, 'folds': folds,
               'test': {m: (float(np.mean([o['test'][m] for o in outs])),
                            float(np.std([o['test'][m] for o in outs], ddof=1)) if len(outs) > 1 else 0.0)
                        for m in outs[0]['test'] if m != 'n'}}
    if 'test_boot' in outs[0]:
        summary['test_boot'] = outs[0]['test_boot']
    json.dump(summary, open(os.path.join(runner.out, 'summary.json'), 'w'), indent=1)
    print(json.dumps(summary.get('test_boot', summary['test']), indent=1))


if __name__ == '__main__':
    main()
