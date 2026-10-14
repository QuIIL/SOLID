# SOLID

<!-- TODO: replace the placeholders below with the final paper title, authors, venue and links. -->
**SOLID: &lt;full paper title&gt;**

This is the official PyTorch implementation of SOLID, which learns ordinal slide representations for computational
pathology. SOLID aligns slide embeddings from a multiple-instance learning (MIL) aggregator with **ordered, text-derived
class anchors**. Its objective has three parts:

| Term | What it does |
|---|---|
| **SOA**: Semantic Ordinal Alignment | Softmax alignment of a slide embedding to the class anchors with a margin that grows with the rank distance, so mistakes far from the true grade cost more. |
| **OLC**: Ordinal Linearity Consistency | Seen from every other anchor, the slide embedding has to lie in the direction of its true anchor, which keeps embeddings consistent with a linear ordering of the grades. |
| **ROD**: Rank-aware Ordinal Dispersion | Keeps the anchors ordered and spread out. The margin between two anchors is the chord length of evenly spaced points on an arc, so it grows with the rank gap. |

<!-- TODO: add the method overview figure, e.g. ![SOLID](assets/overview.png) -->

## Repository layout
```
solid/
  losses.py    SOA, OLC, ROD and SOLIDLoss
  model.py     gated ABMIL aggregator, projection into the shared 1024-d space, classifier
  anchors.py   class anchors: frozen CONCH (or RoBERTa / PubMedBERT) text encoder + trainable projection
  prompts.py   class names and rank prompts ("A WSI showing a diagnosis of ...")
  data.py      TRIDENT .h5 feature loading, fixed training bags packed into a float16 cache
  metrics.py   AUROC, accuracy, balanced accuracy, macro-F1, QWK, MAE, patient-level bootstrap
train.py       training and evaluation entry point
splits/        patient-level splits used in the paper (PANDA, BRACS, TCGA-NSCLC, DHMC -> TCGA-RCC)
scripts/       commands for every dataset
tests/         sanity checks for the loss and the model
```

## Installation
```bash
git clone https://github.com/QuIIL/SOLID.git
cd SOLID
conda create -n solid python=3.10 -y
conda activate solid
pip install -r requirements.txt
```
The default anchors use the CONCH text encoder. CONCH weights are gated on Hugging Face: request access at
[MahmoodLab/conch](https://huggingface.co/MahmoodLab/conch) and run `huggingface-cli login`. Without access, pass
`--text_encoder roberta` to use RoBERTa instead.

## Data preparation
SOLID trains on pre-extracted patch features. We extract them with [TRIDENT](https://github.com/mahmoodlab/TRIDENT)
at 20x magnification without overlap:

| Features | Patch encoder | Patch size | Feature dim |
|---|---|---|---|
| TITAN / CONCH v1.5 | `conch_v15` | 512 px | 768 |
| Prov-GigaPath | `gigapath` | 256 px | 1536 |

```bash
# inside a TRIDENT checkout
python run_batch_of_slides.py --task all --wsi_dir /data/BRACS/wsi --job_dir /data/BRACS \
    --patch_encoder gigapath --mag 20 --patch_size 256 --gpu 0
# -> /data/BRACS/20x_256px_0px_overlap/features_gigapath/<slide_id>.h5
```

Datasets:
[PANDA](https://www.kaggle.com/c/prostate-cancer-grade-assessment) (ISUP grade, 6 classes),
[BRACS](https://www.bracs.icar.cnr.it/) (7 classes),
[TCGA-NSCLC](https://portal.gdc.cancer.gov/) (LUAD / LUSC),
[DHMC-RCC](https://bmirds.github.io/KidneyCancer/) for training and TCGA-RCC (KIRC / KIRP / KICH) for external testing (5 classes).

### Splits
`splits/<dataset>.csv` has one row per slide:

| column | meaning |
|---|---|
| `slide_id` | file name of the feature file without `.h5` |
| `patient_id` | patient identifier. Splits are made at patient level. |
| `label` | class index in rank order (see `solid/prompts.py`) |
| `source` | cohort the slide comes from (`main`, or `dhmc` / `tcga` for RCC) |
| `fold_k` | `train`, `val` or `test` for fold k |

* **PANDA, BRACS, TCGA-NSCLC**: 5-fold patient-level cross-validation, stratified by label. For each fold, one part is
  the test set and 1/8 of the remaining patients is the validation set.
* **RCC**: trained on DHMC (80 % train, 20 % validation) and tested once on TCGA-RCC. TCGA slides with sample type
  code 11 (solid tissue normal) are labelled *Normal*. Confidence intervals come from 1000 patient-level bootstrap
  resamples.

Slides without a feature file are skipped. To use your own data, write a CSV in the same format and pass `--split`.
Then add the class names and prompts to `solid/prompts.py`.

## Training and evaluation
```bash
python train.py --dataset bracs --lr 3e-4 \
    --feat_dir /data/BRACS/20x_256px_0px_overlap/features_gigapath

# cohorts stored in different folders: map each split `source` to its directory
python train.py --dataset rcc --lr 3e-4 \
    --feat_dir dhmc=/data/DHMC/20x_512px_0px_overlap/features_conch_v15 \
    --feat_dir tcga=/data/TCGA-RCC/20x_512px_0px_overlap/features_conch_v15
```
`scripts/run_all.sh` lists the command for every dataset.

On the first run, the fixed 768-patch training bag of every slide is packed into `cache/` as a float16 memmap. Later
folds and runs reuse it, so training never re-reads the full `.h5` files. Each fold is evaluated on full bags with the
last-epoch model. Results are written to `results/<dataset>/<name>/`:

* `fold<k>.json`: test and validation metrics (AUROC, Acc, balanced Acc, macro-F1, QWK, MAE)
* `fold<k>.npz`: slide ids, labels, predicted probabilities, slide embeddings and class anchors
* `summary.json`: mean ± std over folds, or bootstrap mean / std / 95 % CI for RCC

Finished folds are skipped when a command is run again.

### Default hyper-parameters
| | value |
|---|---|
| Aggregator | gated ABMIL, 768 patches per training bag |
| Anchors | frozen CONCH text encoder, centred prompt embeddings, trainable linear projection to 1024-d |
| SOA | τ = 0.07, rank margin m = 0.2 |
| OLC | τ = 0.07 |
| ROD | ordered arc of 180° |
| Loss weights | λ_CE = e / E rises linearly from 0 (first epoch) to 1 (last epoch), λ_SOLID = 1 − λ_CE |
| Optimiser | AdamW, weight decay 0.05, batch size 24, 50 epochs, bf16 autocast, seed 1234 |
| Learning rate | 1e-3 for PANDA, 3e-4 for BRACS, NSCLC and RCC (selected on validation) |

Every value can be changed from the command line (`python train.py -h`).

## Using the SOLID loss in your own pipeline
```python
import torch.nn.functional as F
from solid.losses import SOLIDLoss
from solid.anchors import TextAnchors

anchors = TextAnchors(["A WSI showing a diagnosis of benign prostate tissue", ...])  # one prompt per class, in rank order
solid = SOLIDLoss()

lam_ce = epoch / (epochs - 1)                 # 0 -> 1 over training; the SOLID weight is 1 - lam_ce
e, logits = model(bags)                       # slide embedding (1024-d) and classifier logits
r = anchors(device)                           # [C, 1024], L2-normalised
rep, parts = solid(F.normalize(e, dim=-1), r, y)
loss = lam_ce * F.cross_entropy(logits, y) + (1 - lam_ce) * rep
```

## Tests
```bash
python -m pytest tests
```

## Acknowledgements
Patch features are extracted with [TRIDENT](https://github.com/mahmoodlab/TRIDENT). The anchors use the text encoder
of [CONCH](https://github.com/mahmoodlab/CONCH). The aggregator follows gated attention MIL
([Ilse et al., 2018](https://arxiv.org/abs/1802.04712)).


