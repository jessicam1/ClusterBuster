#!/bin/sh

CONDA_PATH=$1
MAMBA_PATH=$2
ENV_NAME=$3

source ${CONDA_PATH} 
source ${MAMBA_PATH}
conda activate ${ENV_NAME}


SNPS=$4
PQS=$5
TPROP=$6
VPROP=$7
TSAVE=$8
VSAVE=$9
HSAVE=${10}
NCSAVE=${11}
SNPMAP=${12}


python src/create_data.py --snp_list ${SNPS} --from_parquet ${PQS} --training_proportion ${TPROP} --validation_proportion ${VPROP}\
    --save_train ${TSAVE} --save_validation ${VSAVE} --save_holdout ${HSAVE} --save_no_calls ${NCSAVE} --output_snp_map ${SNPMAP}
