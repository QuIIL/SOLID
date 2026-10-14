"""Bag loading from TRIDENT feature files (<feat_dir>/<slide_id>.h5 with a `features` dataset).

Training uses one fixed subset of K patches per slide (same subset every epoch). The subsets of all slides in a split
file are packed once into a float16 memmap so that training does not re-read the full .h5 files every epoch.
"""
import hashlib
import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SEED = 1234


def parse_feat_dirs(entries):
    """['/path'] -> {None: '/path'}; ['dhmc=/a', 'tcga=/b'] -> {'dhmc': '/a', 'tcga': '/b'} (keys = split `source`)."""
    out = {}
    for e in entries:
        src, sep, path = e.partition('=')
        out[src if sep else None] = path if sep else src
    return out


def feature_path(feat_dirs, slide_id, source):
    d = feat_dirs.get(source, feat_dirs.get(None))
    return None if d is None else os.path.join(d, f'{slide_id}.h5')


def read_features(path):
    with h5py.File(path, 'r') as f:
        x = f['features'][:]
    return np.asarray(x, dtype=np.float32).reshape(-1, x.shape[-1])


def fixed_indices(n, k, seed=SEED):
    """Deterministic per-slide subset of k patches (sampled with replacement when the slide has fewer than k)."""
    g = torch.Generator().manual_seed(seed)
    if n >= k:
        return torch.randperm(n, generator=g)[:k].numpy()
    return torch.randint(n, (k,), generator=g).numpy()


class SlideBags(Dataset):
    """Full bags (k=None) or fixed k-patch bags of the given slides."""

    def __init__(self, paths, k=None):
        self.paths, self.k = list(paths), k

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = read_features(self.paths[i])
        if self.k is not None:
            x = x[fixed_indices(len(x), self.k)].astype(np.float16)
        return i, torch.from_numpy(x)


def full_bag_loader(paths, workers=4):
    return DataLoader(SlideBags(paths), batch_size=1, num_workers=workers, collate_fn=lambda b: b[0])


def packed_train_bags(paths, k, cache_dir, workers=8):
    """[len(paths), k, D] float16 memmap of fixed training bags, built once and reused across folds and runs."""
    key = hashlib.md5(('\n'.join(paths) + f'|{k}').encode()).hexdigest()[:12]
    out = os.path.join(cache_dir, f'bags_k{k}_{key}.npy')
    if os.path.exists(out):
        return np.load(out, mmap_mode='r')
    os.makedirs(cache_dir, exist_ok=True)
    loader = DataLoader(SlideBags(paths, k), batch_size=1, num_workers=workers, collate_fn=lambda b: b[0])
    mm = None
    for n, (i, x) in enumerate(loader):
        if mm is None:
            mm = np.lib.format.open_memmap(out + '.tmp', mode='w+', dtype=np.float16, shape=(len(paths), k, x.shape[1]))
        mm[i] = x.numpy()
        if n % 500 == 0:
            print(f'[cache] packing training bags {n}/{len(paths)}', flush=True)
    mm.flush()
    del mm
    os.replace(out + '.tmp', out)
    print(f'[cache] wrote {out}')
    return np.load(out, mmap_mode='r')
