#!/bin/sh
source /data/$USER/conda/etc/profile.d/conda.sh && source /data/$USER/conda/etc/profile.d/mamba.sh
source myconda
conda activate focalloss

NDD_GENES=$1
SNPS=$2
RELEASE_FILE=$3
METRICS_DIR=$4
OUTPUT_DIR=$5

python create_data.py --ndd_genes ${NDD_GENES} --snps ${SNPS} --release_information ${RELEASE_FILE} --metrics_directory ${METRICS_DIR} --output_directory ${OUTPUT_DIR}

### how to run
