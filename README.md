# ClusterBuster

## A Machine Learning Algorithm for Genotyping SNPs from Raw Data

### Purpose
Cluster Buster is a system for recovering the genotypes of no-call SNPs on the Neurobooster array after processing with Illumina Gencall. It is a neural network, outlier detection algorithm, and web app for visualizing SNP genotypes.

### How to Run
A pipeline is conveniently provided in cluster_buster_pipeline.ipynb 

The pipeline is set up to run on NIH's HPC Systems. Running each cell in order will take you through the steps:
1. Gathering and cleaning the data
   Outputs:
       train.csv : training dataset
       validation.csv : validation dataset
       test.csv : hold-out test dataset
       nc_test.csv : dataset containing SNPs of previous no-call genotypes
2. Training and tuning the neural network using the training and validation data.
   Outputs:
       gt_snpid_r_theta_tuner/ : directory for keras-tuner
       gt_model.keras: final keras model
       model_summary.txt : contains model architecture and some hyperparameters
       model_config.json : contains in-depth detail about model layers
       optimizer_config.json: contains  in-depth detail about hyperparameters
3. Rendering predictions from neural network on dataset containing all SNPs without no-calls, the test dataset, and the no-call dataset
   Outputs:
       no_nc_predictions.csv : predictions on the full dataset of all SNPs without no-call genotypes
       test_predictions.csv : predictions on the hold-out test dataset to quanity neural network performance
       nc_predictions.csv : predictions on the no-call dataset
4. Marking outliers. This uses the predictions from the previous set along with metrics from PLINK software.
   Outputs:
       outliers.csv : csv containing each snpID along with outlier metrics and outlier status
5. Visualizing snps with the web app
    The web app allows users to visualize SNP prediction along with metrics about the SNP. 