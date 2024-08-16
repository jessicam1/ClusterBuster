#!/bin/sh
source /data/$USER/conda/etc/profile.d/conda.sh && source /data/$USER/conda/etc/profile.d/mamba.sh
source myconda
conda activate focalloss

SAVEDIR=$1
MODEL=$2
PNAME=$3
TRAIN=$4
VAL=$5

python src/create_model.py --save_directory ${SAVEDIR} --model_name ${MODEL} --project_name ${PNAME} --training_data ${TRAIN} --validation_data ${VAL} --save_information

