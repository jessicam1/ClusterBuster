#!/usr/local/bin/python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
import time
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from tensorflow import keras
from sklearn.cluster import KMeans

parser= argparse.ArgumentParser(description="from dataframes containing SNP metrics and predictions, mark outlier SNPs")
parser.add_argument("-c", "--calls", help="csv with snp metrics and predictions on previously called SNPs (typically train+val+test)")
parser.add_argument("-n", "--no_calls", help="csv with snp metrics and predictions on previously no-call SNPs")
parser.add_argument("-m", "--missingness_threshold", type=float, default=0.5, help="minimum fraction of missingness to call as outlier(default 0.5)")
parser.add_argument("-t", "--outlier_threshold", type=float, default=2, help="multiplier to standard deviation to call as outlier (default 2)")
parser.add_argument("-p", "--plink_freq", help="path to csv/frq output from plink --freq")
parser.add_argument("-r", "--reference_maf", help="path to parquet file containing 1000GenomesProject SNPs and reference minor allele frequencies")
parser.add_argument("-o", "--output", help="outliers output file")
args = parser.parse_args()

# plink_freq fpath = "/data/CARD_AA/projects/2023_05_JM_gt_clusters/plink/GP2_release6_maf.frq"
# maf_reference fpath = "/data/CARD_AA/projects/2023_05_JM_gt_clusters/snp_metrics/hg38_maf"

def mark_outliers(df, multiplier, missingness_threshold):
    # Calculate the mean and standard deviation of the numerical differences
    diff_r_original_predicted = abs(df['R_tightness_original'] - df['R_tightness_final'])
    diff_theta_original_predicted = abs(df['Theta_tightness_original'] - df['Theta_tightness_final'])
    diff_maf_original_predicted = df['MAF_original'] - df['MAF_predicted']
    diff_maf_reference_predicted = abs(df['MAF_reference'] - df['MAF_predicted'])
    
    mean_diff_r = diff_r_original_predicted.mean()
    std_diff_r = diff_r_original_predicted.std()

    mean_diff_theta = diff_theta_original_predicted.mean()
    std_diff_theta = diff_theta_original_predicted.std()

    mean_maf_diff = diff_maf_original_predicted.mean()
    std_maf_diff = diff_maf_original_predicted.std()

    mean_maf_ref_diff = diff_maf_reference_predicted.mean()
    std_maf_ref_diff = diff_maf_reference_predicted.std()

    # Calculate the threshold for marking outliers
    threshold_diff_r = mean_diff_r + multiplier*std_diff_r
    threshold_diff_theta = mean_diff_theta + multiplier*std_diff_theta
    threshold_maf_diff = mean_maf_diff + multiplier*std_maf_diff
    threshold_maf_ref_diff = mean_maf_ref_diff + multiplier*std_maf_ref_diff
    
    ### NEW - OUTLIER IF MAF_ORIGINAL AND MAF_PREDICTED DIFFER BINS ###
    # Define bins
    bins = [-1, 0.000, 0.001, 0.01, 0.05, 0.10, 1.0] #-1 bin is for handling NaN values, see below comment
    # Replace NaN values with -1 in MAF_reference column because pd.cut can't take NaN
    df['MAF_reference'] = df['MAF_reference'].fillna(-1)

    # Assign bin labels
    bin_labels = [f'Bin_{i}' for i in range(len(bins)-1)]
    
    # Assign bins to MAF_original and MAF_predicted
    df['MAF_original_bin'] = pd.cut(df['MAF_original'], bins=bins, labels=bin_labels)
    df['MAF_predicted_bin'] = pd.cut(df['MAF_predicted'], bins=bins, labels=bin_labels)
    df['MAF_reference_bin'] = pd.cut(df['MAF_reference'], bins=bins, labels=bin_labels, include_lowest=True)


    # Create boolean columns indicating outlier status for each condition
    # maf outliers determined by change in bin
    df['outlier_maf_original_predicted'] = df['MAF_original_bin'] != df['MAF_predicted_bin']
    df['outlier_maf_reference_predicted'] = df['MAF_reference_bin'] != df['MAF_predicted_bin']
    # change outlier_maf_reference_predicted to false if it was true because the MAF bin is -1 (previously NaN) and we don't care about that 
    df.loc[df['MAF_reference_bin'] == 'Bin_0', 'outlier_maf_reference_predicted'] = False
    # tightness outliers determined by threshold
    df['outlier_R_tightness'] = abs(diff_r_original_predicted) > threshold_diff_r
    df['outlier_Theta_tightness'] = abs(diff_theta_original_predicted) > threshold_diff_theta
    # missingness determined by fraction
    df['outlier_high_missingness'] = abs(df['miss_frac']) > missingness_threshold

    # Create 'outlier' column with combined outlier conditions
    df['outlier'] = (df['outlier_R_tightness'] | df['outlier_Theta_tightness'] | df['outlier_maf_reference_predicted'] | df['outlier_high_missingness'] | df['outlier_maf_original_predicted'])

    # Create 'outlier_reason' column indicating the reason for being an outlier
    df.loc[df['outlier'], 'outlier_reason'] = df[['outlier_R_tightness', 'outlier_Theta_tightness', 'outlier_maf_reference_predicted', 'outlier_high_missingness', 'outlier_maf_original_predicted']].idxmax(axis=1).apply(lambda x: x.replace('outlier_', ''))

    # Fill non-outlier rows in 'outlier_reason' column with NaN
    df['outlier_reason'].fillna('', inplace=True)

    # Drop intermediate outlier check columns
    # df.drop(['outlier_R_tightness', 'outlier_Theta_tightness', 'outlier_maf_reference_predicted', 'outlier_high_missingness', 'outlier_maf_original_predicted'], axis=1, inplace=True)

    return df

def cluster_tightness(data, n_clusters):
    """
    use K means on each snpID to calculate cluster tightness along R and Theta
    internal method used in def get_cluster_tightness()
    """
    # Initialize KMeans
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    # Fit the data
    kmeans.fit(data)
    # Get the cluster assignments and centroids
    assignments = kmeans.labels_
    centroids = kmeans.cluster_centers_
    tightness = 0
    # For each cluster
    for i in range(n_clusters):
        # Get the points in this cluster
        cluster_points = data[assignments == i]
        # Calculate the sum of squared distances from the centroid
        distances = np.sum((cluster_points - centroids[i]) ** 2)
        # Add to the total tightness
        tightness += distances
    # new idea: make tightness an average by dividing by number of clusters
    tightness = tightness/n_clusters
    return tightness


def get_cluster_tightness(data, gt_col_name, new_col_suffix):
    """
    get number of clusters and cluster tightness calculations for R and Theta for each snpID
    uses above method cluster_tightness()
    gt_col_name - specify which column to base clusters on (either original GT column or predicted GT column
    new_col_suffix - specify the suffix of the R_tightness, theta_tightness new columns (recommend "original" or "predicted"
    """
    # get number of different GTs predicted for each SNP for clustering later and add them to series
    gb = data.groupby("snpID")
    snp_clusters_dict = {}
    for key, item in gb:
        snp_clusters_dict[key] = item["GT"].nunique()        
    n_series = pd.Series(snp_clusters_dict, name=f"clusters _{new_col_suffix}")

    # merge list of clusters per SNP with original dataframe
    data = data.merge(n_series.rename(f"clusters_{new_col_suffix}"), how="left", left_on="snpID", right_index=True)
    # create df where there are no snps with original_clusters = 0
    # can't apply kmeans to 0 clusters
    no_nan = data[data[f"clusters_{new_col_suffix}"] > 0]
    # calculate cluster tightness along R and Theta
    g = no_nan.groupby("snpID")
    
    # df2 = g[['R','Theta']].apply(lambda x: tightness(x, n_clusters=1))
    df2 = g[['R','Theta', f"clusters_{new_col_suffix}"]].apply(lambda x: cluster_tightness(x[["R","Theta"]], n_clusters=x[f"clusters_{new_col_suffix}"].iloc[0]))
    # zscore the calculated R, Theta tightness
    # df2[["R", "Theta"]] = df2[["R", "Theta"]].apply(zscore)
    # rename to reflect tightness operation
    df2.rename(columns={"R":f"R_tightness_{new_col_suffix}", "Theta":f"Theta_tightness_{new_col_suffix}"}, inplace=True)

    return df2

def gt_cat(row, gt_col_name):
    # mapping function make a GT categorical variable - one hot encode GT column into GT_cat column
    # internal method used in convert_GT_to_cat()
    if row[f"{gt_col_name}"] == "AA":
        return 0
    if row[f"{gt_col_name}"] == "AB":
        return 1
    if row[f"{gt_col_name}"] == "BA":
        return 1
    if row[f"{gt_col_name}"] == "BB":
        return 2
    if row[f"{gt_col_name}"] == "NC":
        return np.NaN

def convert_GT_to_cat(data, gt_col_name, new_col_name):
    # apply gt_cat() mapping function to add new column with 0/1/2/NaN instead of AA/AB/BB/NC
    # data[f"{gt_col_name}_{new_col_suffix}"] = data.apply(lambda row: gt_cat(row, gt_col_name), axis=1)
    data[f"{new_col_name}"] = data[f"{gt_col_name}"].map({"AA":0, "AB":1, "BA":1, "BB":2, "NC":np.NaN})
    return data

def count_nc(df):
    """
    internal method used by calculate_predicted_maf to count no-calls appearing in set
    """
    # Group by 'SNPID' and count occurrences of 'NC' in 'GT' column
    nc_counts = df.groupby('snpID')['GT'].apply(lambda x: x[x.str.contains('NC')].count())
    # Convert nc_counts to a DataFrame with columns 'snpID' and 'num_NC'
    nc_counts = pd.DataFrame(nc_counts)
    nc_counts.rename(columns={"GT":"num_NC"}, inplace=True)
    nc_counts = nc_counts.reset_index()
    
    return nc_counts

def count_minor_allele(row):
    # Determine if A1 or A2 from plink .frq is the minor allele
    # Count minor alleles in new GT of each SNP only if previous GT was NC
    
    # Determine the minor allele
    minor_allele = np.where(row["MAF_original"] < 0.5, row["A1"], row["A2"])
    
    # Handle complements by converting minor allele to a holder value
    holder_A1 = np.where(np.isin(minor_allele, ["A", "T"]), "X", "Y")
    holder_a1 = np.where(np.isin(row["a1"], ["A", "T"]), "X", "Y")
    a1_frq = row["A1"]
    a2_frq = row["A2"]
    a1_snps = row["a1"]
    a2_snps = row["a2"]
    # Determine which allele is the minor allele
    if a1_frq == a1_snps:
        # straightforward match
        MA = "A"
    elif a1_frq == a2_snps:
        # straightforward match
        MA = "B"
    elif holder_A1 == holder_a1:
        # handles opposite strand (complements) being measured
        MA = "A"
    else:
        # handles opposite strand (complements) being measured
        MA = "B"
    row["minor_allele"] = MA
    
    # Count the occurrences of the minor allele in GT_predicted depending on which NN value is minor allele
    if row["GT"] == "NC":
        MA_count = np.where(MA == "A", row['GT_predicted'].count('A'), row['GT_predicted'].count('B')).sum()
        # MA_count = np.where(MA == "A", row['GT_predicted'].str.count('A'), row['GT_predicted'].str.count('B')).sum()
        # ^ attribute error str object has no attribute str
        row["num_MA"] = MA_count
    else:
        # Don't count minor alleles if original GT is AA, AB, or BB
        row["num_MA"] = 0    
    
    return row

def calculate_predicted_maf(df):
    # NEW METHOD
    # maf_old = NCHROBS/totalsamples
    # so totalsamples = NCHROBS/maf_old
    num_original_snps = df["NCHROBS"] / df["MAF_original"]
    num_new_snps = df["num_NC"]
    total_snps = num_original_snps + num_new_snps
    total_minor_alleles = df["NCHROBS"] + df["num_new_MA"]
    # maf_pred = NCHROBS + newMAcount / (totalsamples + newsamples)
    MAF_predicted = total_minor_alleles / total_snps
    df["MAF_predicted"] = MAF_predicted
    # return df
    
    # OLD METHOD - INCORRECT
    # THIS METHOD ASSUMED THAT NCHROBS WAS THE TOTAL NUNBER OF SAMPLES THAT MAF WAS CALCULATED WITH, NOT OBSERVATIONS OF THE MINOR ALLELE
    # Calculate total number of observations and minor alleles
    # df['total_observations'] = df['NCHROBS'] + df['num_NC']
    # df['total_minor_alleles'] = (df['MAF_original'] * df['NCHROBS']) + df['num_MA']
    # Update MAF based on new observations
    # df['MAF_predicted'] = df['total_minor_alleles'] / df['total_observations']
    return df

def calculate_missingness(data):
    """ 
    internal method used in flag_missingess to get fraction missingness per snp
    """
    missing_value = "NC"
    grouped = data.groupby("snpID")["GT"].apply(lambda x: (x == missing_value).sum(axis=0) / len(x)).reset_index()
    grouped.columns = ['snpID', 'miss_frac']
    return grouped

def get_missingness(data):
    grouped = calculate_missingness(data)
    # missingness_dict = dict(zip(grouped['snpID'], grouped['miss_frac']))
    # data['miss_frac'] = data['snpID'].map(missingness_dict)
    return grouped

def complement_base(base):
    """ internal method used within merge_snp_dataframes """ 
    complement_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return complement_dict.get(base, base)

def inverse_maf(maf):
    """ internal method used within merge_snp_dataframes """ 
    if pd.notna(maf):
        return 1 - maf
    return maf

def expand_repeat_sequence(sequence):
    """ internal method used within merge_snp_dataframes """ 
    """ handles alleles like "4C" and changes to "CCCC" """
    result = ""
    count = ""
    for char in sequence:
        if char.isdigit():
            count += char
        else:
            if count:
                result += int(count) * char
                count = ""
            else:
                result += char
    
    return result

def switch_multi_allelic(row):
    """ internal method used within merge_snp_dataframes """ 
    # Check if ref_x, alt_x, ref_y, or alt_y are multi-allelic and apply expand_sequence if needed
    multi_allelic_columns = ['ref_x', 'alt_x', 'ref_y', 'alt_y']
    # Convert values in specified columns to strings
    row[multi_allelic_columns] = row[multi_allelic_columns].astype(str)
    for col in multi_allelic_columns:
        if len(row[col]) > 1:
            row[col] = expand_repeat_sequence(row[col])
    # Check if ref_y is multi-allelic, switch value to alt_y and invert maf if so
    if len(row['ref_y']) > 1:
        # Flip values of ref_y and alt_y
        row['ref_y'], row['alt_y'] = row['alt_y'], row['ref_y']  
        # Invert the MAF if not NaN
        if pd.notna(row['maf']):
            row['maf'] = 1 - row['maf']
    # Check if ref_x and ref_y are of different lengths
    if len(row['ref_x']) != len(row['ref_y']):
        row['maf'] = pd.NA
    
    return row

def merge_snp_dataframes(df1, df2):
    """ THIS IS THE METHOD TO GET REFERENCE MAF INTO SNP_METRICS DATAFRAME """
    # get dataframes organized
    df2[["chromosome", "position"]] = df2[["chromosome", "position"]].astype("str")
    df1[["chromosome", "position"]] = df1[["chromosome", "position"]].astype("str")
    df1.rename(columns={"Ref":"ref", "Alt":"alt"}, inplace=True)
    df2[["ref", "alt"]] = df2[["ref", "alt"]].astype("str")
    df1[["ref", "alt"]] = df1[["ref", "alt"]].astype("str")

    # Merge dataframes on chromosome, position, and reference
    merged_df = pd.merge(df1, df2, how='left', on=['chromosome', 'position'])
    
    # Switch values for multi-allelic ref_y
    merged_df = merged_df.apply(switch_multi_allelic, axis=1)
    
    # Handle cases where ref_x and ref_y are different
    different_refs = (merged_df['ref_x'] != merged_df['ref_y'])

    # Set maf to NaN if ref_y is NaN
    merged_df['maf'] = np.where(different_refs & merged_df['ref_y'].isna(), pd.NA, merged_df['maf'])

    # Inverse maf for cases where ref_x and ref_y are different and not complements
    non_complement_refs = different_refs & ~(merged_df['ref_x'].apply(complement_base) == merged_df['ref_y'])
    merged_df.loc[non_complement_refs, 'maf'] = merged_df.loc[non_complement_refs, 'maf'].apply(inverse_maf)

    # Add a column indicating if the maf had to be made inverse
    merged_df['maf_inverse_flag'] = np.where(non_complement_refs, 1, 0)
    
    # Set maf_inverse_flag to NaN when maf is NaN
    merged_df['maf_inverse_flag'] = np.where(merged_df['maf'].isna(), pd.NA, merged_df['maf_inverse_flag'])
    
    # Replace "nan" with "NaN" in the entire dataframe
    merged_df.replace("nan", "NaN", inplace=True, regex=True)
    
    # Drop unnecessary columns
    merged_df = merged_df.drop(["ref_y", "alt_y"], axis=1)
    
    # rename maf to MAF_ref
    merged_df.rename(columns={"maf":"MAF_reference", "snpID_y":"snpID_ref"}, inplace=True)
    
    return merged_df

### RUN CODE ### 

# read input calls predictions and no calls predictions and concatenate into one dataframe
calls = pd.read_csv(args.calls, sep=",", header=0)
no_calls = pd.read_csv(args.no_calls, sep=",", header=0)
df = pd.concat([calls, no_calls])
 
# read maf reference and gp2 plink freq into dataframes
maf_gp2 = pd.read_csv(args.plink_freq, sep='\s+', header=0)
maf_gp2.rename(columns={"MAF":"MAF_original"}, inplace=True)

refmaf = pd.read_parquet(args.reference_maf, engine="pyarrow")
refmaf = refmaf.drop_duplicates(subset="snpID", keep="first")

# convert GT variables to categorical
df = convert_GT_to_cat(df, "GT_predicted", "GT_predicted_cat")

print("calculating cluster tightness", file=sys.stderr)
# get GT cluster tightness for original GTs and predicted GTs from snpid and clusters model
ct_original = get_cluster_tightness(df, "GT_cat", "original")
ct_snpid = get_cluster_tightness(df, "GT_predicted_cat", "final")

# merge cluster tightness metrics for original and snpid model into a snpid model specific dataframe
metrics_snpid = ct_original.merge(ct_snpid, how="left", on="snpID")
metrics_snpid.reset_index(inplace=True)

# merge GP2 maf into large df to get A1, A2, and MAF_original for every snp for determining minor allele
df = df.merge(maf_gp2, how="left", left_on="snpID", right_on="SNP")
# also get GP2 maf into metrics for calculating MAF_predicted from MAF_original
metrics_snpid = metrics_snpid.merge(maf_gp2, how="left", left_on="snpID", right_on="SNP")

print("calculating MAF_original", file=sys.stderr)
# get number of NC per snp and number of minor alleles per SNP for calculating new MAF
nc_counts = count_nc(df)
metrics_snpid = metrics_snpid.merge(nc_counts, how="left", left_on="snpID", right_on="snpID")
metrics_snpid.rename(columns={"num_NC_y":"num_NC"}, inplace=True)
# determine if A or B from the neural network is the minor allele and count new instances of the minor allele
df = df.apply(count_minor_allele, axis=1)
# sum MA counts for each snpID and make it a dataframe
ma_counts = df.groupby("snpID").apply(lambda x: x["num_MA"].sum()).reset_index(name="num_new_MA")
metrics_snpid = metrics_snpid.merge(ma_counts, how="left", left_on="snpID", right_on="snpID")#, suffixes=("","_x"))
# calculate predicted/new MAF using NC counts and MA counts
metrics_snpid = calculate_predicted_maf(metrics_snpid)

print("getting missingness", file=sys.stderr)
# add missingness to dataframes
missing_snpid = get_missingness(df)
metrics_snpid = metrics_snpid.merge(missing_snpid, how="left", left_on="snpID", right_on="snpID")

# get chromosome, position, Ref, and Alt into dataframe to use for merging with reference maf dataframe [and gene for later reference]
metrics_snpid = metrics_snpid.merge(df[["snpID", "chromosome", "position", "Ref", "Alt", 'gene']], how="left", left_on="snpID", right_on="snpID")

print("adding reference MAF", file=sys.stderr)
# add reference maf to dataframe with snps from genes of interest
metrics_snpid = merge_snp_dataframes(metrics_snpid, refmaf)
# clean up columns
metrics_snpid.rename(columns={"snpID_x":"snpID", "ref_x":"ref", "alt_x":"alt"}, inplace=True)
metrics_snpid.drop(columns=["SNP", "CHR", "num_NC", "num_new_MA", "A1", "A2", "NCHROBS"], inplace=True)
# drop duplicates
metrics_snpid = metrics_snpid.drop_duplicates(subset="snpID", keep="first")

print("marking outliers", file=sys.stderr)
# mark outliers
df_snpid_outliers = mark_outliers(metrics_snpid, args.outlier_threshold, args.missingness_threshold)

# save outliers df
print("saving dataframe", file=sys.stderr)
df_snpid_outliers.to_csv(args.output, index=False)