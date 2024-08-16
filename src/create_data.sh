#!/bin/sh
source /data/$USER/conda/etc/profile.d/conda.sh && source /data/$USER/conda/etc/profile.d/mamba.sh
source myconda
conda activate focalloss

SNPS=$1
PQS=$2
TPROP=$3
VPROP=$4
TSAVE=$5
VSAVE=$6
HSAVE=$7
NCSAVE=$8


python src/create_data.py --snp_list ${SNPS} --from_parquet ${PQS} --training_proportion ${TPROP} --validation_proportion ${VPROP}\
    --save_train ${TSAVE} --save_validation ${VSAVE} --save_holdout ${HSAVE} --save_nocalls ${NCSAVE} --create_snp_categorical_map
