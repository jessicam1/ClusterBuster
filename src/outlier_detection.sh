#!/bin/sh
source /data/$USER/conda/etc/profile.d/conda.sh && source /data/$USER/conda/etc/profile.d/mamba.sh
source myconda
conda activate focalloss

CALLS=$1
NO_CALLS=$2
MISS=$3
THRESH=$4
FRQ=$5
REF=$6
OUTPUT=$7

python outlier_detection.py --calls ${CALLS} --no_calls ${NO_CALLS} --missingness_threshold ${MISS} --outlier_threshold ${THRESH} --plink_freq ${FRQ} --reference_maf ${REF} --output ${OUTPUT}