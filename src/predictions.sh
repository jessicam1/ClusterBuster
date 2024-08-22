#!/bin/sh

CONDA_PATH=$1
MAMBA_PATH=$2
ENV_NAME=$3

source ${CONDA_PATH} 
source ${MAMBA_PATH}
conda activate ${ENV_NAME}

MODEL=$4
MAP=$5
INPUT_FILE=$6
OUTPUT_FILE=$7

python src/predictions.py --model ${MODEL} --snp_map ${MAP} --from_csv ${INPUT_FILE} --to_csv ${OUTPUT_FILE}



