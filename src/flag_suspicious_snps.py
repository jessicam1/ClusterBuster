#!/usr/local/bin/python
import sys
import os
import time
import argparse
import numpy as np
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser(description="Mark SNPs as suspicious based on genotype clustering behavior")
    parser.add_argument("-c", "--calls", nargs='+', required=True, help="CSV with SNPs, genotypes, and snp metrics (R, Theta, ALLELE_A")
    parser.add_argument("-g", "--genotype_col", help="Name of the genotype column (e.g., GT)")
    parser.add_argument("-a", "--a1counts_col", help="Name of the A1counts column (e.g., A1counts)")
    parser.add_argument("-d", "--centroid_distance_threshold", type=float, default=0.15, help="Threshold at which distance between centroids is too close")
    parser.add_argument("-w", "--cluster_width_threshold", type=float, default=0.4, help="Threshold at which width of genotype cluster is too wide (Theta)")
    parser.add_argument("-p", "--dead_probe_threshold", type=float, default=0.15, help="Threshold at which centroids are too low (dead probe)")
    parser.add_argument("-r", "--round_counts", action="store_true", default=False, help="Round A1counts column")
    parser.add_argument("-o", "--output", required=True, help="SNP-level output file")
    return parser.parse_args()

def cluster_tightness_per_snp(data, genotype_col):
    def cluster_tightness(group):
        tightness_dict = {"snpID": group.name}
        for gt in [0.0, 1.0, 2.0]:
            data = group.loc[group[genotype_col] == gt].copy()
            if len(data) > 0:
                centroid_r = data["R"].sum() / len(data)
                centroid_theta = data["Theta"].sum() / len(data)
                distances = data[["R", "Theta"]].to_numpy() - np.array([centroid_r, centroid_theta])
                distances_r = np.abs(distances[:, 0])
                distances_theta = np.abs(distances[:, 1])
                tightness_dict.update({
                    f'R_centroid_{gt}': centroid_r,
                    f'Theta_centroid_{gt}': centroid_theta,
                    f'R_avg_distance_{gt}': np.mean(distances_r),
                    f'Theta_avg_distance_{gt}': np.mean(distances_theta),
                    f'R_stddev_distance_{gt}': np.std(distances_r),
                    f'Theta_stddev_distance_{gt}': np.std(distances_theta),
                    f'R_max_difference_{gt}': data['R'].max() - data['R'].min(),
                    f'Theta_max_difference_{gt}': data['Theta'].max() - data['Theta'].min(),
                })
            else:
                for metric in ["R_centroid", "Theta_centroid", "R_avg_distance", "Theta_avg_distance", "R_stddev_distance", "Theta_stddev_distance", "R_max_difference", "Theta_max_difference"]:
                    tightness_dict[f'{metric}_{gt}'] = np.NaN
        return pd.Series(tightness_dict)

    return data.groupby("snpID").apply(cluster_tightness).reset_index(drop=True)

def convert_GT_to_cat(data, genotype_col, new_col_name):
    data[new_col_name] = data[genotype_col].map({"AA": 0, "AB": 1, "BA": 1, "BB": 2, "NC": np.NaN})
    return data

def calculate_GT(row, a1counts_col):
    if row["ALLELE_A"] == 0:
        if row[a1counts_col] == 2.0:
            return "AA"
        elif row[a1counts_col] == 1.0:
            return "AB"
        elif row[a1counts_col] == 0.0:
            return "BB"
    elif row["ALLELE_A"] == 1:
        if row[a1counts_col] == 0.0:
            return "AA"
        elif row[a1counts_col] == 1.0:
            return "AB"
        elif row[a1counts_col] == 2.0:
            return "BB"
    return None

def flag_close_centroids(data, threshold=0.15):
    data['close_centroids'] = data.apply(
        lambda row: any(
            np.abs(row[f'Theta_centroid_{gt1}'] - row[f'Theta_centroid_{gt2}']) < threshold
            for gt1, gt2 in [(0.0, 1.0), (0.0, 2.0), (1.0, 2.0)]
        ), axis=1
    )
    return data

def flag_wide_clusters(data, threshold=0.4):
    data['wide_cluster'] = data[
        ['Theta_max_difference_0.0', 'Theta_max_difference_1.0', 'Theta_max_difference_2.0']
    ].gt(threshold).any(axis=1)
    return data

def flag_dead_probe(data, threshold=0.15):
    below_threshold_count = data[
        ['R_centroid_0.0', 'R_centroid_1.0', 'R_centroid_2.0']
    ].lt(threshold).sum(axis=1)
    data['dead_probe'] = below_threshold_count >= 1
    return data

def flag_suspicious_imputation(data):
    data['suspicious'] = data[['wide_cluster', 'dead_probe', 'close_centroids']].any(axis=1)
    return data

def main():
    args = parse_arguments()

    if args.a1counts_col and args.genotype_col:
        print("Must specify either A1counts column or genotype column.", file=sys.stderr)
        sys.exit(1)
        
    calls = pd.concat([pd.read_csv(call) for call in args.calls], ignore_index=True)

    if args.genotype_col:
        gt_col = args.genotype_col
        
    if args.a1counts_col:
        a1_col = args.a1counts_col
        if args.round_counts:
            calls[f'{args.a1counts_col}_rounded'] = calls[args.a1counts_col].round()
            a1col = f'{args.a1counts_col}_rounded'
            gt_col = "GT"
            calls[gt_col] = calls.apply(calculate_GT, axis=1, a1counts_col=a1col)

    calls = convert_GT_to_cat(calls, gt_col, f"{gt_col}_cat")

    ct = cluster_tightness_per_snp(calls, f"{gt_col}_cat")
    ct = flag_wide_clusters(ct, threshold=args.cluster_width_threshold)
    ct = flag_dead_probe(ct, threshold=args.dead_probe_threshold)
    ct = flag_close_centroids(ct, threshold=args.centroid_distance_threshold)
    ct = flag_suspicious_imputation(ct)

    ct.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()
