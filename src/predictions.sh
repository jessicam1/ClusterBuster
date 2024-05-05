#!/bin/sh
source /data/$USER/conda/etc/profile.d/conda.sh && source /data/$USER/conda/etc/profile.d/mamba.sh
source myconda
conda activate focalloss

MODEL=$1
INPUT_FILE=$2
OUTPUT_FILE=$3

python src/predictions.py --model ${MODEL} --input ${INPUT_FILE} --output ${OUTPUT_FILE}

### how to run
