# ClusterBuster

## A Machine Learning Algorithm for Genotyping SNPs from Raw Data

### Purpose
Cluster Buster is a system for recovering the genotypes of no-call SNPs on the Neurobooster array after genotyping with Illumina Genome Studio. It is a neural network, genotype concordance analysis, and SNP genotype plotting system.

### How to Run
A pipeline is conveniently provided in cluster_buster_pipeline.ipynb 

The pipeline is set up to run on NIH's HPC Systems. Running each cell in order will take you through the steps:
1. Gathering and cleaning the data\
   Outputs:\
       &nbsp;&nbsp;&nbsp;&nbsp;train.csv : training dataset\
       &nbsp;&nbsp;&nbsp;&nbsp;validation.csv : validation dataset\
       &nbsp;&nbsp;&nbsp;&nbsp;test.csv : hold-out test dataset\
       &nbsp;&nbsp;&nbsp;&nbsp;nc_test.csv : dataset containing SNPs of previous no-call genotypes\
2. Training and tuning the neural network using the training and validation data.\
   Outputs:\
       &nbsp;&nbsp;&nbsp;&nbsp;gt_snpid_r_theta_tuner/ : directory for keras-tuner\
       &nbsp;&nbsp;&nbsp;&nbsp;gt_model.keras: final keras model\
       &nbsp;&nbsp;&nbsp;&nbsp;model_summary.txt : contains model architecture and some hyperparameters\
       &nbsp;&nbsp;&nbsp;&nbsp;model_config.json : contains in-depth detail about model layers\
       &nbsp;&nbsp;&nbsp;&nbsp;optimizer_config.json: contains  in-depth detail about hyperparameters\
3. Rendering predictions from neural network on dataset containing all SNPs without no-calls, the test dataset, and the no-call dataset\
   Outputs:\
       &nbsp;&nbsp;&nbsp;&nbsp;no_nc_predictions.csv : predictions on the full dataset of all SNPs without no-call genotypes\
       &nbsp;&nbsp;&nbsp;&nbsp;test_predictions.csv : predictions on the hold-out test dataset to quanity neural network performance\
       &nbsp;&nbsp;&nbsp;&nbsp;nc_predictions.csv : predictions on the no-call dataset\
4. Marking outliers. This uses the predictions from the previous set along with metrics from PLINK software.\
   Outputs:\
       &nbsp;&nbsp;&nbsp;&nbsp;outliers.csv : csv containing each snpID along with outlier metrics and outlier status\
5. Visualizing snps with the web app\
    The web app allows users to visualize SNP prediction along with metrics about the SNP. 
