#!/bin/sh
source /data/$USER/conda/etc/profile.d/conda.sh && source /data/$USER/conda/etc/profile.d/mamba.sh
source myconda
conda activate focalloss

python create_model.py

### how to run
# sbatch --cpus-per-task=4 --mem=100g --mail-type=BEGIN,END --time=5:00 kerastuner.sh
