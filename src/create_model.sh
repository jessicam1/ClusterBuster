#!/bin/sh

CONDA_PATH=$1
MAMBA_PATH=$2
ENV_NAME=$3

source ${CONDA_PATH} 
source ${MAMBA_PATH}
conda activate ${ENV_NAME}

SAVEDIR=$4
MODEL=$5
PNAME=$6
TRAIN=$7
VAL=$8

python src/create_model.py --save_directory ${SAVEDIR} --model_name ${MODEL} --project_name ${PNAME} --training_data ${TRAIN} --validation_data ${VAL} --save_information

