#!/bin/bash
# SOLID on every dataset of the paper. Point the feature directories to your TRIDENT outputs.
# PANDA, BRACS, NSCLC: 5-fold patient-level CV. RCC: DHMC train/val -> TCGA-RCC test.
set -e
cd "$(dirname "$0")/.."
GIGAPATH=20x_256px_0px_overlap/features_gigapath
CONCH15=20x_512px_0px_overlap/features_conch_v15

python train.py --dataset panda --lr 1e-3 --feat_dir /data/PANDA/$GIGAPATH --name solid_gigapath
python train.py --dataset bracs --lr 3e-4 --feat_dir /data/BRACS/$GIGAPATH --name solid_gigapath
python train.py --dataset nsclc --lr 3e-4 --feat_dir /data/TCGA-NSCLC/$GIGAPATH --name solid_gigapath
python train.py --dataset rcc --lr 3e-4 --name solid_titan \
    --feat_dir dhmc=/data/DHMC-RCC/$CONCH15 --feat_dir tcga=/data/TCGA-RCC/$CONCH15

# The same commands with $CONCH15 features (CONCH v1.5, the TITAN patch encoder) give the TITAN-feature results.
