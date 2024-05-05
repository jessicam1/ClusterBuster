#!/usr/local/bin/python
import sys
import os
import pickle
import glob
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

parser= argparse.ArgumentParser(description="fro")
parser.add_argument("-n", "--ndd_genes", help="csv with ndd-related genes and their chromosome and positions")
parser.add_argument("-s", "--snps", help="parquet containing snpIDs and their ref, alt, chromosome, position")
parser.add_argument("-r", "--release_information", help="containing sample information including ancestry, phenotype")
parser.add_argument("-m", "--metrics_directory", help="directory to stream snp metrics files from when gathering data")
parser.add_argument("-o", "--output_directory", help="directory save uncleaned data and train, test, val, nc cleaned data to")
args = parser.parse_args()

# example parquet for --snps "/data/CARD_AA/projects/2023_05_JM_gt_clusters/snp_metrics/snp_metrics_204842400049/Sample_ID=204842400049_R01C01"
# release 6 "/data/CARD_AA/projects/2023_05_JM_gt_clusters/snp_metrics/gp2_sampleids_release6_with_ancestries.tab"
# metrics dir = "/data/GP2/raw_genotypes/snp_metrics"
# savedir = "/data/CARD_AA/projects/2023_05_JM_gt_clusters/capstone/data" or test in /data/testing_create_data


def filter_ndd_snps(ndd_genes, specific_genes, df):
    # Change values in chromosome column from "chrN" to "N"
    ndd_genes['chromosome'] = ndd_genes['chromosome'].str.replace('chr', '')
    # Convert 'chromosome', 'start_pos', and 'end_pos' to numeric
    ndd_genes['start_pos'] = pd.to_numeric(ndd_genes['start_pos'])
    ndd_genes['end_pos'] = pd.to_numeric(ndd_genes['end_pos'])

    # Filter gene_df for specific genes and select the first instance
    filtered_gene_df = ndd_genes[ndd_genes['gene'].isin(specific_genes)].groupby('gene').first().reset_index()
    
    # filter the snp metrics file down to genes of interest using snpid from df and chrom, pos from ndd_genes
    filtered_snps = pd.DataFrame(columns=df.columns)

    # Iterate over each row in the gene DataFrame
    for _, gene_row in filtered_gene_df.iterrows():
        # Filter SNPs based on chromosome and position ranges
        condition = (df['chromosome'] == gene_row['chromosome']) & \
                    (df['position'] >= gene_row['start_pos']) & \
                    (df['position'] <= gene_row['end_pos'])
        # Append the filtered SNPs to the result DataFrame, add "gene" column
        filtered_snps = pd.concat([filtered_snps, df[condition].assign(gene=gene_row["gene"])], ignore_index=True)

    # Reset the index of the result DataFrame
    filtered_snps.reset_index(drop=True, inplace=True)
    return filtered_snps

def filter_metrics_by_snps(metrics, snps, m_col, s_col): #, appended_data):
    """ 
    filter parquet files with a list of snps 
    metrics - dataframe of parquet file
    snps - dataframe of snps file to filter with
    m_col - snp id column name in parquet metrics df
    s_col - snp id column name in snp df
    """
    metrics = metrics.merge(snps, how="inner", left_on=m_col, right_on=s_col)
    return metrics

def collect_snp_metrics(release, metrics_dir, testing=False):
    """ 
    this function uses release information to sample evenly from ancestries and using the constructed paths from the 'filename' column, 
    appends the snp metrics of samples of interest to appended_data dataframe
    testing - while testing this function, script, set testing=True so that only 3 files are tested to avoid long looping
    """
    # get samples evenly distributed across ancestry
    if testing==True:
        sampled = release.groupby('label').\
            apply(lambda x: x.sample(n=1))
    else:
        sampled = release.groupby('label').\
            apply(lambda x: x.sample(n=65))
    sampled = sampled.reset_index(drop=True)
    sampled['path']= sampled['SentrixBarcode_A'].astype(str) + '/' + 'snp_metrics_' + sampled['SentrixBarcode_A'].astype(str) + '/Sample_ID=' + sampled['filename'].astype(str)
    # add the filenames SentrixBarcode_A_SentrixPosition_A to a list
    snp_metrics = sampled['path'].tolist()
    
    # load each parquet file in the directory, filter it for specific SNPs in snpsfile using filter_metrics_by_snps(), and append filtered data to dataframe
    # this takes a really long time
    appended_data = pd.DataFrame()
    snp_col = "snpID"
    metrics_snp_col = "snpID"
    i = 0 
    for filename in snp_metrics:
        filepath = os.path.join(metrics_dir, filename)
        i += 1
        if testing==True:
            if i==25:
                break
        try:
            metrics = pd.read_parquet(filepath)
            sampleid = filepath.split("Sample_ID=")[-1]
            metrics['Sample_ID'] = sampleid
            metrics_data = filter_metrics_by_snps(
                            metrics, filtered_snps, metrics_snp_col, snp_col)
            appended_data = pd.concat([appended_data, metrics_data], axis=0, ignore_index=True)
        except:
            print(f"File '{filename}' not found. Skipping...", file=sys.stderr)
            continue
    return appended_data

def clean_data(appended_data, release, testing=False):
    drop_cols = {"Ref_y", "Alt_y", "chromosome_y", "position_y"}
    appended_data = appended_data.drop(labels=drop_cols, axis=1)
    # add demographic info
    appended_data = appended_data.merge(release[["filename", "label", "phenotype", "sex"]], how="left", left_on="Sample_ID", right_on="filename")
    # some of the files/Sample_IDs we picked didn't exist in snp_metrics resulting in uneven representation of each ancestry
    # resample the SampleIDs to ensure even representation among ancestries
    # first get unique sampleIDs and labels into dataframe for sampling
    samples = appended_data["Sample_ID"].unique()
    samples_df = pd.DataFrame({'Sample_ID': samples})
    samples_df = samples_df.merge(release[["filename","label"]], how="left", left_on="Sample_ID", right_on="filename")
    samples_df = samples_df.drop(columns="filename")
    
    # get samples evenly distributed across ancestry
    if testing==True:
        resampled = samples_df.groupby('label').\
            apply(lambda x: x.sample(n=1))
    else:
        resampled = samples_df.groupby('label').\
            apply(lambda x: x.sample(n=44))
    resampled = resampled.reset_index(drop=True)
    
    # now use resampled SampleIDs to filter appended_data
    resampled_sampleids = resampled["Sample_ID"].tolist()
    appended_data = appended_data[appended_data["Sample_ID"].isin(resampled_sampleids)]
    
    if testing==True:
        print(appended_data.columns, file=sys.stderr)
    # column cleanup
    # drop_cols = {"Ref_y", "Alt_y", "chromosome_y", "position_y"}
    # appended_data = appended_data.drop(labels=drop_cols, axis=1)
    cols = {"position_x":"position", "chromosome_x":"chromosome", "Ref_x":"Ref", "Alt_x":"Alt"}
    appended_data.rename(columns=cols, inplace=True)
    
    # clean data
    appended_data = appended_data.loc[appended_data['GenTrain_Score'].notna()]
    appended_data = appended_data.loc[appended_data['Theta'].notna()]
    appended_data = appended_data.loc[appended_data['R'].notna()]
    # drop mysterious NaN that don't get picked up by notna(), np.NaN, "NaN", or "nan"
    appended_data = appended_data[appended_data['GT'].isin(['AA', 'AB', 'BB', 'NC'])]    
    
    # one hot encode GT column
    appended_data["GT_original"] = appended_data["GT"]
    appended_data = pd.get_dummies(appended_data, prefix="GT", columns=["GT"], dtype="float")
    appended_data = appended_data.drop(labels="GT_NC", axis=1)
    # rename snpID_original col back to snpID for better bookkeeping
    cols = {"GT_original":"GT"}
    appended_data.rename(columns=cols, inplace=True)
    
    # make column snpID_cat that has snpID has numerical categorical feature
    appended_data['snpID_cat'] = pd.factorize(appended_data['snpID'])[0]
    
    if testing==True:
        print(appended_data.columns)
        print(appended_data.shape)
        print(appended_data["GT"].unique())
    return appended_data


# read in all ndd genes
cols = ["gene", "chromosome", "start_pos", "end_pos"]
ndd_genes = pd.read_csv(args.ndd_genes, sep=",", names=cols)

# Define the list of specific genes
specific_genes = ['SNCA', 'APOE', 'GBA', 'LRRK2']
                    
# read in parquet containing snps
snps = pd.read_parquet(args.snps)
snps = snps[["snpID","Ref","Alt","chromosome","position"]]

# read in release information
release = pd.read_csv(args.release_information, sep="\t", header=0)

# establish snps of interest within genes of interest
filtered_snps = filter_ndd_snps(ndd_genes, specific_genes, snps)
# collect snp metrics of snps of interest, sampled evenly by ancestry into large dataframe
appended_data = collect_snp_metrics(release, args.metrics_directory, testing=True)
# save large uncleaned appended data
# appended_data.to_csv(f"{args.output_directory}/appended_data_uncleaned.csv", sep=",")
# clean the data
appended_data = clean_data(appended_data, release, testing=True)

# split the data into train, val, test, nc
nc = appended_data[appended_data["GT"] == "NC"]
df = appended_data[appended_data["GT"] != "NC"]
df_train, df_val = train_test_split(df, test_size=0.10, random_state=0)
df_val, df_test = train_test_split(df_val, test_size=0.50, random_state=0)

# save the data
df_train.to_csv(f"{args.output_directory}/train.csv", index=False)
df_val.to_csv(f"{args.output_directory}/val.csv", index=False)
df_test.to_csv(f"{args.output_directory}/test.csv", index=False)
nc.to_csv(f"{args.output_directory}/nc_test.csv", index=False)
df.to_csv(f"{args.output_directory}/no_nc.csv", index=False)