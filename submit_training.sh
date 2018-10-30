#!/bin/bash
source deactivate
source /soft/miniconda3/activate
source activate tf



n_runs="$(python param_gen.py)"

sbatch --array=1-$n_runs -o ~/results/out/%A-%a.out ./training.sh
