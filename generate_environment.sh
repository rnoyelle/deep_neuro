#!/usr/bin/env bash
source deactivate
source /soft/miniconda3/activate

# Create env with packages
conda create -n deep_neuro python=3.5 numpy scipy scikit-learn pandas matplotlib tensorflow h5py imageio

# Generate tree
cd ..
mkdir -p data/raw
mkdir -p results/training
mkdir -p results/pvals
mkdir -p results/summary
mkdir -p results/plots
mkdir -p scripts/_params
mv deep_neuro/ scripts/
