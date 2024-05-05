#!/usr/local/bin/python
import sys
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from tensorflow import keras

print(tf.__version__, file=sys.stderr)

parser= argparse.ArgumentParser(description="from a dataframe containing snp metrics, get predictions from a keras model")
parser.add_argument("-m", "--model", help="path to pickled trained ML model")
parser.add_argument("-i", "--input", help="input file to render predictions with (must contain R, Theta, snpID)")
parser.add_argument("-o", "--output", help="predictions output file")
args = parser.parse_args()

def determine_genotype(row):
    max_prob = max(row['AA_prob'], row['AB_prob'], row['BB_prob'])
    if row['AA_prob'] == max_prob:
        return 'AA'
    elif row['AB_prob'] == max_prob:
        return 'AB'
    else:
        return 'BB'

def get_predictions(model, data):
    # gather features
    features_numerical = ["R", "Theta"]
    features_snpid = ["snpID_cat"]
    X_numerical = df[features_numerical].astype("float").to_numpy()
    X_snpid = df[features_snpid].astype("int").to_numpy()

    # get predictions
    preds = model.predict([X_snpid, X_numerical])

    # put predictions into dataframe
    preds_df = pd.DataFrame(data=preds, columns=["AA_prob","AB_prob","BB_prob"])

    # get predicted GT from probability predictions
    preds_df['GT_predicted'] = preds_df.apply(determine_genotype, axis=1)
    
    return preds_df
    
# model_path = '/data/CARD_AA/projects/2023_05_JM_gt_clusters/models/snp_specific_evenancestry_focalloss_short/gt_snpid_r_theta_tuner_evenancestry_focalloss.keras'

# load keras model
model = tf.keras.models.load_model(args.model)

# load data
df = pd.read_csv(args.input, sep=",", header=0)

# gather features and get predictions
preds_df = get_predictions(model, df)

# merge with original df to get key info like snpID
df_final = df.join(preds_df, how="outer")

# create output file
columns = ["snpID", "snpID_cat", "Sample_ID", "chromosome", "position", "gene", "Ref", "Alt", "a1", "a2", "GenTrain_Score", "phenotype", "label", "sex", "R", "Theta", "GT", "GT_AA", "GT_AB", "GT_BB", "GT_predicted", "AA_prob", "AB_prob", "BB_prob"]
output_df = df_final[columns]

# save output file
output_df.to_csv(args.output, index=False)