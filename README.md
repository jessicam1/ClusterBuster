# ClusterBuster

## A Machine Learning Algorithm for Genotyping SNPs from Raw Data

### Purpose
Cluster Buster is a system for recovering the genotypes of no-call SNPs on the Neurobooster array after genotyping with Illumina Genome Studio. It is a genotype-predicting neural network and SNP genotype plotting system.<br><br>

### Getting Started
Create a Conda environment for the project using requirements.txt.
```
conda create --name clusterbuster --file requirements.txt
```

### Usage 

#### Run Basic Cluster Buster Pipeline
A pipeline is conveniently provided in cluster_buster_pipeline.ipynb. The pipeline can be run directly within the jupyter notebook or on a HPC system with sbatch commands. <br><br>
```
create_data.sh
```
Purpose: Gathering, cleaning, and outputting SNP metrics data for Cluster Buster genotype prediction\
Outputs:<br>
       &nbsp;&nbsp;&nbsp;&nbsp;training dataset CSV<br>
       &nbsp;&nbsp;&nbsp;&nbsp;validation dataset CSV<br>
       &nbsp;&nbsp;&nbsp;&nbsp;hold-out test dataset CSV<br>
       &nbsp;&nbsp;&nbsp;&nbsp;dataset CSV containing SNPs of previous no-call genotypes<br><br>
```
create_model.sh
```
Purpose: Training and tuning the Cluster Buster neural network using the training and validation data to predict SNP genotypes<br>
Outputs:<br>
       &nbsp;&nbsp;&nbsp;&nbsp;keras neural network tuner directory<br>
       &nbsp;&nbsp;&nbsp;&nbsp;keras model<br>
       &nbsp;&nbsp;&nbsp;&nbsp;model_summary.txt : contains model architecture and some hyperparameters<br>
       &nbsp;&nbsp;&nbsp;&nbsp;model_config.json : contains in-depth detail about model layers<br>
       &nbsp;&nbsp;&nbsp;&nbsp;optimizer_config.json: contains  in-depth detail about hyperparameters<br><br>
```
predictions.sh
```
Purpose: Rendering predictions from neural network on four datasets: training, validation, testing, and no-calls<br>
Outputs:<br>
       &nbsp;&nbsp;&nbsp;&nbsp;predictions on the training dataset<br>
       &nbsp;&nbsp;&nbsp;&nbsp;predictions on the validation dataset<br>
       &nbsp;&nbsp;&nbsp;&nbsp;predictions on the hold-out test dataset to quantify neural network performance<br>
       &nbsp;&nbsp;&nbsp;&nbsp;predictions on the no-call dataset<br>

```
plot_snp_figures.sh
```
Purpose: Create figures visualizing SNP genotypes.\
Outputs:<br>
       &nbsp;&nbsp;&nbsp;&nbsp;a directory for every snpID given, containing a PNG of predicted genotypes<br><br>
#### Get Predictions Directly from Parquet Files 
* Specify the Keras model with --model flag where {model} is the name of the Keras model.<br>
* The Cluster Buster neural network requires a categorical (numerical) version of snpID called "snpID_cat" that is mapped the same way as the snpIDs during training. Specify the SNP mapping scheme with your own CSV (snpID, snpID_cat) or the CSV output from create_data.py.<br>
* If you want to get genotype predictions using Cluster Buster directly from parquet files, feed the --from_parquet flag a text file containing a list of parquet files {pq_list}.<br>
* If you want to save these predictions to an aggregated CSV or parquet file, use --to_csv or --to_parquet.<br>
* If you want to save the predictions for each parquet in their own, new parquet file, use --parquet_to_parquet. Fies are named "{original_pq_filename}_predictions.parquet" and saved to specified directory {predictions_directory}.<br>
```
python predictions.py \
    --model {model} \
    --snp_map {snp_map} \ 
    --from_parquet {pq_list} \
    --parquet_to_parquet {predictions_directory}
```
Or run the bash file that runs predictions.py in a specific environment.
```
bash src/predictions.sh \
    {conda_source} \
    {mamba_source} \
    {conda_environment_name} \   
    {model} \
    {snp_map} \
    {validation_data} \
    {validation_predictions} 
```
#### Get Figure for Each SNP Comparing Predicted, Imputed, and WGS Genotypes. 
You must have a dataframe containing SNPs of interest along with their predicted genotypes (GT_predicted), imputed genotypes (GT_imputed), and WGS genotypes (GT_wgs). 
This file can be created by the user independently or follow the steps in render_analyze_genotype_predictions.ipynb.
* To specify snpIDs to plot, feed --snps a text file containing snpIDS {snp_list}. <br>
* To specify the genotypes to plot, feed --genotypes either CSV or parquet files containing SNP metrics and genotypes; multiple files accepted {dataset1} {dataset2}. <br>
* To specify where to save the figure files (PDF for combined figure, PNG for individual genotype plots; named with cleaned snpID string), give --output_directory a directory {output_directory}.<br>
* If you want to create a PDF with imputed, WGS, and predicted genotypes side by side, use --plot_full_figure flag. <br>
* If you want to save individual PNGs with single genotypes (only imputed genotypes, etc), specify with flags --plot_predicted_individual, --plot_imputed_individual, --plot_wgs_individual.<br>
```
python plot_snp_figures.py \
   --snps {snp_list} \
   --genotypes {dataset1} {dataset2} \
   --annotations {annotations_csv} \
   --output_directory {output_directory} \
   --plot_predicted_individual \
   --plot_imputed_individual \
   --plot_wgs_individual \
   --plot_full_figure \
```
Or run the bash file that runs plot_snp_figures.py in a specific environment.
```
bash plot_snp_figures.sh \
    {conda_source} \
    {mamba_source} \
    {conda_environment_name} \   
    {snp_list} \
    {dataset1} {dataset2} \
    {output_directory} \
    --plot_predicted_individual \
    --plot_imputed_individual \
    --plot_wgs_individual \
    --plot_full_figure
```
### Notebooks
In the analysis folder are two provided notebooks.
```
render_analyze_predicted_genotypes.ipynb
```
This notebook starts from rendering predictions directly from parquet files. Then, using PLINK, imputed genotypes are extracted, transformed and merged with the predictions dataset. Then, using PLINK, WGS genotypes are extracted, transformed, and merged with the predictions dataset. This dataset is saved to CSV. <br>
The analysis section analyzes rates of concordance between the different genotyping methods. A dataset summarizing the concordance and counts per SNP is output to CSV. It also selects high-performing SNPs (according to a concordance threshold) and calculates a new call rate for those SNPs.
```
compare_gencall_concordance.ipynb
```
This notebook uses the output dataset from render_analyze_predicted_genotypes.ipyb to analyze concordance between the valid Gencall genotypes (AA, AB, BB) with imputed and WGS genotypes. A dataset summarizing per-SNP concordance rates is output to CSV.<br>
The analysis section compares Gencall concordance with GenTrain Score, compares Gencall concordance rates with predicted genotype concordance rates, and defines SNPs to exclude from Gencall genotyping based on concordance. The difference in Cluster Buster performance between included and excluded SNPs is analyzed and visualized with a box plot.
