#!/usr/bin/env bash
source deactivate
source /soft/miniconda3/activate

# Create env with packages
conda create -n deep_neuro python=3.5 numpy scipy scikit-learn pandas matplotlib tensorflow h5py imageio

# Generate tree
cd ..
mkdir -p data/raw
mkdir -p results/training/pvals
mkdir -p results/training/summary
mkdir -p results/training/plots
mkdir -p scripts/_params
mv deep_neuro/ scripts/