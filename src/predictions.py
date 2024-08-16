#!/usr/local/bin/python
import sys
import csv
import os
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from tensorflow import keras

def parse_arguments():
    parser= argparse.ArgumentParser(description=(
        "From a CSV or list of parquets containing SNP metrics, ",
        "get genotype predictions from Cluster Buster keras model")
    )
    parser.add_argument("-m", "--model", help="path to pickled trained ML model")
    parser.add_argument("-s", "--snp_map", help="path txt file with mapping of snpIDs to categorical variables (snpID, snpID_cat)")
    parser.add_argument("-c", "--from_csv", help="input file to render predictions with (must contain R, Theta, snpID)")
    parser.add_argument("-p", "--from_parquet", help="txt file containing list of parquet file names to render predictions with")
    parser.add_argument("--to_csv", help="name of CSV to save predictions")
    parser.add_argument("--to_parquet", help="name of parquet to save predictions")
    parser.add_argument("--parquet_to_parquet", help=(
        "Name of output directory."
        "Save predictions from each parquet file to a new parquet file."
        "New parquet file is named {csv_or_parquet_name}_predictions.parquet.")
    )  
    return parser.parse_args()

    
def determine_genotype(row):
    """
    Method used in get_predictions(model, df) to
    determine predicted genotype from probabilities
    """
    
    max_prob = max(row['AA_prob'], row['AB_prob'], row['BB_prob'])
    if row['AA_prob'] == max_prob:
        return 'AA'
    elif row['AB_prob'] == max_prob:
        return 'AB'
    else:
        return 'BB'

def get_predictions(model, df):
    """
    Inputs:
    model - loaded keras model
    df - dataframe containing snpID_cat, R, Theta
    Function: use keras model to render genotype predictions
        for every row of df
    Outputs:
    df - dataframe containing original data plus GT_predicted,
        AA_prob, AB_prob, BB_prob
    """
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
    
    df = df.reset_index(drop=True)
    preds_df = preds_df.reset_index(drop=True)

    # Concatenate the original dataframe and predictions dataframe
    df = pd.concat([df, preds_df], axis=1)
    
    return df

def main():
    args = parse_arguments()

    model = tf.keras.models.load_model(args.model)

    if args.from_csv and args.from_parquet:
        print("must specify only one input", file=sys.stderr)
    if args.from_csv and args.parquet_to_parquet:
        print("--parquet_to_parquet can only be used when --from_parquet is used", file=sys.stderr)
        
    if args.from_parquet:
        parquet_list = []
        with open(args.from_parquet, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                parquet_list.append(row[0])
        
    if args.snp_map:
        with open(args.snp_map, 'r') as ids:
            snp_map = {}
            for line in ids:
                key, value = line.strip().split(',')
                snp_map[str(key)] = value
            ids.close()
    
    columns = ["snpID", "snpID_cat", "Sample_ID", "chromosome", "position", "Ref", "Alt", "ALLELE_A", 
               "ALLELE_B", "a1", "a2", "GenTrain_Score", "R", "Theta", "GT", "GT_predicted", "AA_prob", "AB_prob", "BB_prob"]

    if args.from_csv:
        try:
            df = pd.read_csv(args.from_csv, sep=",", header=0)
            if args.snp_map:
                df["snpID_cat"] = df["snpID"].map(snp_map)
            df = df.dropna(subset=["snpID_cat"])
            preds_df = get_predictions(model, df)
            existing_columns = [col for col in columns if col in preds_df.columns]
            output_df = preds_df[existing_columns]
            if args.to_csv:
                try:
                    output_df.to_csv(args.to_csv, index=False)
                except Exception as e:
                    print(f"Error saving CSV file {args.to_csv}: {e}", file=sys.stderr)
                    sys.exit(1)
            if args.to_parquet:
                try:
                    output_df.to_parquet(args.to_parquet, index=False)
                except Exception as e:
                    print(f"Error saving parquet file {output_filename}: {e}", file=sys.stderr)
                    sys.exit(1)
        
        except FileNotFoundError:
            print(f"Error: The file {args.from_csv} does not exist.", file=sys.stderr)
            sys.exit(1)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error processing CSV file: {e}", file=sys.stderr)
            sys.exit(1)

    if args.from_parquet:
        if args.parquet_to_parquet:
            for parquet in parquet_list:
                try:
                    pq = pd.read_parquet(parquet)
                    if args.snp_map:
                        pq["snpID_cat"] = pq["snpID"].map(snp_map)
                        pq = pq.dropna(subset=["snpID_cat"])

                except FileNotFoundError:
                    print(f"Error: The parquet file {parquet} does not exist.", file=sys.stderr)
                    continue
                except Exception as e:
                    print(f"Error reading parquet file {parquet}: {e}", file=sys.stderr)
                    continue
                
                preds_df = get_predictions(model, pq)
                existing_columns = [col for col in columns if col in preds_df.columns]
                output_df = preds_df[existing_columns]
                filename = os.path.splitext(os.path.basename(parquet))[0]
                output_filename = f"{args.parquet_to_parquet}/{filename}_predictions.parquet"
                print(output_filename)
                try:
                    output_df.to_parquet(output_filename, index=False)
                except Exception as e:
                    print(f"Error saving parquet file {output_filename}: {e}", file=sys.stderr)
                    continue

        elif args.to_csv or args.to_parquet:
            combined_df = pd.DataFrame()
            for parquet in parquet_list:
                try:
                    pq = pd.read_parquet(parquet)
                    combined_df = pd.concat([combined_df, pq], ignore_index=True)
                except FileNotFoundError:
                    print(f"Error: The parquet file {parquet} does not exist.", file=sys.stderr)
                    continue
                except Exception as e:
                    print(f"Error reading parquet file {parquet}: {e}", file=sys.stderr)
                    continue
            
            if combined_df.empty:
                print("Error: No data found in parquet files.", file=sys.stderr)
                sys.exit(1)
            
            if args.snp_map:
                combined_df["snpID_cat"] = combined_df["snpID"].map(snp_map)
                combined_df = combined_df.dropna(subset=["snpID_cat"])
            
            preds_df = get_predictions(model, combined_df)
            existing_columns = [col for col in columns if col in preds_df.columns]
            output_df = preds_df[existing_columns]
            if args.to_csv:
                try:
                    output_df.to_csv(args.to_csv, index=False)
                except Exception as e:
                    print(f"Error saving CSV file {args.to_csv}: {e}", file=sys.stderr)
                    sys.exit(1)
            if args.to_parquet:
                try:
                    print("trying to save")
                    output_df.to_parquet(args.to_parquet, index=False)
                except Exception as e:
                    print(f"Error saving Parquet file {args.to_parquet}: {e}", file=sys.stderr)
                    sys.exit(1)


if __name__ == "__main__":
    main()