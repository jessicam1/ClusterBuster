#!/bin/sh
source /data/$USER/conda/etc/profile.d/conda.sh && source /data/$USER/conda/etc/profile.d/mamba.sh
source myconda
conda activate focalloss

SNPS=$1
GTS=$2
CALLS=$3
ANNOT=$4
OUT=$5

python src/plot_snps.py --snps {SNPS} --genotypes {GTS} --gencalls {CALLS} --annotations {ANNOT} --output_directory {OUT}

### how to run
