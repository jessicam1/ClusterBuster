#!/usr/local/bin/python
import sys
import os
import random
import numpy as np
import pandas as pd
import argparse

def parse_arguments():
    parser= argparse.ArgumentParser(description=(
        "From a list of parquet files containing SNP metrics, create training, validation, holdout, "
        "and no-call sets for Cluster Buster and save to CSV or parquet.")
    )
    parser.add_argument("-s", "--snp_list", help="txt file containing snpIDs of interest [in APOE, GBA, SNCA, LRRK2]")
    parser.add_argument("-p", "--from_parquet", help=(
        "txt file containing list of parquet file names to render predictions with.")
    )
    parser.add_argument("--training_proportion", type=float, default=0.80, help=(
        "proportion of data to make training dataset (float from 0.00 to 1.00)."
        "default: 0.80.")
    )
    parser.add_argument("--validation_proportion", type=float, default=0.10, help=(
        "proportion of data to make validation dataset (float from 0.00 to 1.00)."
        "default: 0.10.")
    )
    parser.add_argument("-u", "--use_snp_categorical_map", help=(
        "txt file containing snpID and corresponding categorical variable (snpID_cat).")
    )
    parser.add_argument("--save_train", help="path to save training dataset, .csv or .parquet")
    parser.add_argument("--save_validation", help="path to save validation dataset, .csv or .parquet")
    parser.add_argument("--save_holdout", help="path to save holdout dataset, .csv or .parquet")
    parser.add_argument("--save_no_calls", help="path to save no calls dataset, .csv or .parquet")
    parser.add_argument("--output_snp_map", help="CSV to save mapping of snpIDs to snp categorical variable")
    
    return parser.parse_args()

def get_dataframe_from_parquets(parquet_list, snp_list):
    """
    Inputs:
    parquet_list - list of filepaths to parquet files containing snp metrics
    snp_list - list of snpIDs on NeuroBooster array to filter data
    Function:
    Load parquet files specified in parquet_list, filtered down snpIDs contained
    in snp_list, remove NaN genotypes, append data to larger dataframe combined_df
    Outputs:
    combined_df - dataframe containing snp metrics from all parquet files
    """
    combined_df = pd.DataFrame()
    for parquet_path in parquet_list:
        try:
            pq_full = pd.read_parquet(parquet_path)
            sampleid = parquet_path.split("Sample_ID=")[-1]
            pq_full['Sample_ID'] = sampleid
            pq_full = pq_full.loc[pq_full["snpID"].isin(snp_list)]
            pq = pq_full.loc[pq_full["GT"].isin(["AA","AB","BB", "NC"])]
            combined_df = pd.concat([combined_df, pq], ignore_index=True)
        except FileNotFoundError:
            print(f"Error: The parquet file {parquet_path} does not exist.", file=sys.stderr)
            continue
        except Exception as e:
            print(f"Error reading parquet file {parquet_path}: {e}", file=sys.stderr)
            continue

    return combined_df

def clean_data(df, snp_map=None):
    """
    Inputs:
    df - dataframe containing snp metrics, output from get_dataframe_from_parquets()
    snp_map - dictionary where keys are NeuroBooster Array snpIDS, 
    values are corresponding numerical categorical variable,
    default None (if snpID_cat is already in df)
    Function:
    Clean up snp_metrics, one-hot encode GT column, create snpID_cat, split df into calls (df) and no calls (nc)
    Outputs:
    df - snp metrics with valid gencall genotypes (AA, AB, BB)
    nc - snp metrics with no gencall genotypes (NC)
    """
    cols = ["GenTrain_Score", "Theta", "R"]
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce', axis=0)
    df = df.dropna(subset=cols)

    df["GT_original"] = df["GT"]
    df = pd.get_dummies(df, prefix="GT", columns=["GT"], dtype="float")
    desired_columns = ["GT_AA", "GT_AB", "GT_BB", "GT_NC"]    
    df[desired_columns] = df.reindex(columns=desired_columns, fill_value=0)
    df = df.drop(columns="GT_NC", axis=1)
    df.rename(columns={"GT_original":"GT"}, inplace=True)

    if snp_map is not None:
        df["snpID_cat"] = df["snpID"].map(snp_map)
        df = df.dropna(subset=["snpID_cat"])
    else: 
        df['snpID_cat'] = pd.factorize(df['snpID'])[0]

    nc = df.loc[df["GT"] == "NC"].copy()
    df = df.loc[df["GT"].isin(["AA","AB","BB"])].copy()
    
    return df, nc
    
def main():
    
    args = parse_arguments()
    
    with open(args.snp_list, 'r') as snps:
        snp_list = [line.strip() for line in snps]


    snp_map = None
    if args.use_snp_categorical_map:
        with open(args.snp_map, 'r') as ids:
               snp_map = dict(line.strip().split(',') for line in ids)
    else:
        snp_map = {snp: idx for idx, snp in enumerate(snp_list)}
        
    tr_num = args.training_proportion
    va_num = args.training_proportion + args.validation_proportion
    if va_num >= 1.0:
        print("Error: Sum of training proportion and validation proportion cannot be greater than 1.00", file=sys.stderr)
        sys.exit()
        
    with open(args.from_parquet, 'r') as pqs:
        training_list = []
        validation_list = []
        holdout_list = []
        for line in pqs:
            pq = line.strip()
            r = random.random()
            if (0.0 <=  r <= tr_num): 
                training_list.append(pq) 
            elif (tr_num < r <= va_num): 
                validation_list.append(pq) 
            else:
                holdout_list.append(pq) 
        pqs.close()

    if training_list:
        training = get_dataframe_from_parquets(training_list, snp_list)
        training, train_nc = clean_data(training, snp_map=snp_map)
    else:
        training = pd.DataFrame()  
        train_nc = pd.DataFrame()
    
    if validation_list:
        validation = get_dataframe_from_parquets(validation_list, snp_list)
        validation, val_nc = clean_data(validation, snp_map=snp_map)
    else:
        validation = pd.DataFrame()
        val_nc = pd.DataFrame()
    
    if holdout_list:
        holdout = get_dataframe_from_parquets(holdout_list, snp_list)
        holdout, holdout_nc = clean_data(holdout, snp_map=snp_map)
    else:
        holdout = pd.DataFrame()
        holdout_nc = pd.DataFrame()
    
    nc = pd.concat([train_nc, val_nc, holdout_nc], axis=0) if not train_nc.empty or not val_nc.empty or not holdout_nc.empty else pd.DataFrame()
    
    save_dict = {
        args.save_train: training,
        args.save_validation: validation,
        args.save_holdout: holdout,
        args.save_no_calls: nc
    }
    
    for save_arg, dataframe in save_dict.items():
        if not dataframe.empty:
            if save_arg.endswith('.parquet'):
                dataframe.to_parquet(save_arg)
            else:
                dataframe.to_csv(save_arg, index=False)
    

    if args.output_snp_map:
        snp_map_df = training[['snpID', 'snpID_cat']].drop_duplicates()
        try:
            snp_map_df.to_csv(args.output_snp_map, index=False, header=False)
        except Exception as e:
            print(f"Error saving SNP map CSV file {args.output_snp_map}: {e}", file=sys.stderr)
            sys.exit(1)
if __name__ == "__main__":
    main()
