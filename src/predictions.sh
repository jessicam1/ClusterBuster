#!/bin/sh
source /data/$USER/conda/etc/profile.d/conda.sh && source /data/$USER/conda/etc/profile.d/mamba.sh
source myconda
conda activate focalloss

MODEL=$1
MAP=$2
INPUT_FILE=$3
OUTPUT_FILE=$4

python src/predictions.py --model ${MODEL} --snp_map ${MAP} --from_csv ${INPUT_FILE} --to_csv ${OUTPUT_FILE}

