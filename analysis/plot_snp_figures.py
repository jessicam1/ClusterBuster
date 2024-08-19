#!/usr/local/bin/python
import sys
import os
import csv
import string
import argparse
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.image as mpimg
import plotly.subplots as sp
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot SNP genotypes from prediction, imputation, and WGS side-by-side and save to PDF"
    )
    parser.add_argument(
        "-s", "--snps", required=True, 
        help="CSV containing snpIDs to plot, one on each line"
    )
    parser.add_argument(
        "-g", "--genotypes",
        help=(
            "CSV containing snpIDs, Theta, R, predicted genotypes ('GT_predicted'), "
            "imputed genotypes ('GT_imputed'), and WGS genotypes ('GT_wgs')"
        )
    )
    parser.add_argument(
        "-c", "--gencalls",
        help=(
            "CSV of gencall genotypes containing snpIDs, genotypes ('GT'), Theta, R, "
            "ALLELE_A, ALLELE_B, a1, a2 from snp metrics pipeline"
        )
    )
    parser.add_argument(
        "-a", "--annotations",
        help=(
            "CSV to make plot annotations with. Must contain columns 'snpID', "
            "'GenTrain_Score', 'gene', rate of concordance with imputation "
            "('concordance_imputed'), rate of concordance with WGS ('concordance_wgs')"
        )
    )
    parser.add_argument(
        "-i", "--save_individual", action='store_true',
        help="Save individual PNGs of each plot (predicted, imputed, WGS genotypes)"
    )
    parser.add_argument(
        "-o", "--output_directory", required=True,
        help=(
            "Path to output directory for PDFs of plotted SNPs. "
            "Saves structure {output_directory}/{gene}/{snp}/{file}"
        )
    )
    args = parser.parse_args()
    return args


def plot_snp_alleles(data, snp_id, calls, allele_column, result_frame=None):
    """
    Generates a scatter plot for a given SNP showing allele distributions.

    Parameters:
    ----------
    data : pandas.DataFrame
        DataFrame containing data for a particular snpID, including columns for Theta, R, ALLELE_A, a1, a2, and genotype calls.
    snp_id : str
        Identifier for the SNP being plotted.
    calls : pandas.DataFrame
        DataFrame containing gencall genotype calls for  particular snpID, including a column for 'Original Alleles'.
    allele_column : str
        The type of genotype to plot (e.g., "Predicted Alleles", "Imputed Alleles", "WGS Alleles").
    result_frame : pandas.DataFrame, optional
        DataFrame containing SNP analysis results. Default is None.
        
    Returns:
    -------
    plotly.graph_objs._figure.Figure
        Plotly Figure object representing the SNP scatter plot.

    """
    
    gt_col_dict = {
        "Original Alleles": "GT",
        "Predicted Alleles": "GT_predicted",
        "Imputed Alleles": "GT_imputed",
        "WGS Alleles": "GT_wgs"
    }

    # Determine the alleles based on ALLELE_A
    if data['ALLELE_A'].iloc[0] == 0:
        allele_A = data['a1'].iloc[0]
        allele_B = data['a2'].iloc[0]
    else:
        allele_A = data['a2'].iloc[0]
        allele_B = data['a1'].iloc[0]
    
    # Create a new column for genotype labels
    gt_col = gt_col_dict[allele_column]
    data[f'{allele_column}'] = data[f'{gt_col}'].map({
        "AA": f'{allele_A}{allele_A}', 
        "AB": f'{allele_A}{allele_B}', 
        "BB": f'{allele_B}{allele_B}'
    })
    calls["Original Alleles"] = calls["GT"].map({
        "AA": f'{allele_A}{allele_A}', 
        "AB": f'{allele_A}{allele_B}', 
        "BB": f'{allele_B}{allele_B}'
    })   
    
    fig1 = px.scatter(data, x='Theta', y='R', color=allele_column, 
                     # color_discrete_map={'AA': 'blue', 'AB': 'orange', 'BB': 'green', 'NC': 'red'},
                     color_discrete_map={
                         f'{allele_A}{allele_A}': 'blue', 
                         f'{allele_A}{allele_B}': 'orange', 
                         f'{allele_B}{allele_B}': 'green', 
                         np.NaN: 'red'},
                     )
    
    color_map_trace = {
        f'{allele_A}{allele_A}': 'cornflowerblue', 
        f'{allele_A}{allele_B}': 'gold', 
        f'{allele_B}{allele_B}': 'lightgreen',
        'nan' : 'red'
    }


    for genotype, color in color_map_trace.items():
        filtered_calls = calls[calls["Original Alleles"] == genotype]
        fig1.add_scatter(
            x=filtered_calls['Theta'],
            y=filtered_calls['R'],
            mode='markers',
            marker=dict(color=color, size=6, opacity=0.4),
            # name=f'gencall {genotype}',
            showlegend=False
        )
    
    fig1.update_xaxes(range=[-0.05, 1.05])
    fig1.update_yaxes(range=[0.00, 3.50])
    fig1.update_layout(showlegend=False)
        
    return fig1

def plot_snp_alleles_save_individual(data, snp_id, calls, allele_column, result_frame=None, output_file=None):
    """
    Generates a scatter plot for a given SNP showing allele distributions.

    Parameters:
    ----------
    data : pandas.DataFrame
        DataFrame containing data for a particular snpID, including columns for Theta, R, ALLELE_A, a1, a2, and genotype calls.
    snp_id : str
        Identifier for the SNP being plotted.
    calls : pandas.DataFrame
        DataFrame containing gencall genotype calls for  particular snpID, including a column for 'Original Alleles'.
    allele_column : str
        The type of genotype to plot (e.g., "Predicted Alleles", "Imputed Alleles", "WGS Alleles").
    result_frame : pandas.DataFrame, optional
        DataFrame containing SNP analysis results. Default is None.
    output_file : str, optional
        File path to save the individual plot image if `save_individual` is True. Default is None.
    """
    
    gt_col_dict = {
        "Original Alleles": "GT",
        "Predicted Alleles": "GT_predicted",
        "Imputed Alleles": "GT_imputed",
        "WGS Alleles": "GT_wgs"
    }

    # Determine the alleles based on ALLELE_A
    if data['ALLELE_A'].iloc[0] == 0:
        allele_A = data['a1'].iloc[0]
        allele_B = data['a2'].iloc[0]
    else:
        allele_A = data['a2'].iloc[0]
        allele_B = data['a1'].iloc[0]
    
    # Create a new column for genotype labels
    gt_col = gt_col_dict[allele_column]
    data[f'{allele_column}'] = data[f'{gt_col}'].map({
        "AA": f'{allele_A}{allele_A}', 
        "AB": f'{allele_A}{allele_B}', 
        "BB": f'{allele_B}{allele_B}'
    })
    calls["Original Alleles"] = calls["GT"].map({
        "AA": f'{allele_A}{allele_A}', 
        "AB": f'{allele_A}{allele_B}', 
        "BB": f'{allele_B}{allele_B}'
    })   
    
    fig1 = px.scatter(data, x='Theta', y='R', color=allele_column,
                        title=f'{snp_id} {allele_column}',
                        color_discrete_map={
                            f'{allele_A}{allele_A}': 'blue',
                            f'{allele_A}{allele_B}': 'orange',
                            f'{allele_B}{allele_B}': 'green', 
                            np.NaN: 'red'},
                     )
    
    color_map_trace = {
        f'{allele_A}{allele_A}': 'cornflowerblue', 
        f'{allele_A}{allele_B}': 'gold', 
        f'{allele_B}{allele_B}': 'lightgreen',
        'nan' : 'red'
    }


    for genotype, color in color_map_trace.items():
        filtered_calls = calls[calls["Original Alleles"] == genotype]
        fig1.add_scatter(
            x=filtered_calls['Theta'],
            y=filtered_calls['R'],
            mode='markers',
            marker=dict(color=color, size=6, opacity=0.4),
            showlegend=True,
            name=f'Calls ({genotype})'
        )
    
    fig1.update_xaxes(range=[-0.05, 1.05])
    fig1.update_yaxes(range=[0.00, 3.50])
    fig1.update_layout(showlegend=True)
        
    fig1.write_image(output_file)

def plot_combined_snp_plots(data, result_frame, snp_id, gene, calls, output_file=None, show_fig=True):

    """
    Generates three scatter plots for a particular snpID, comparing multiple genotype types (Imputed, WGS, Predicted).
    Semi-transparent layer of gencall genotypes in the back of each scatter plot.

    Parameters:
    ----------
    data : pandas.DataFrame
        DataFrame containing raw data for a particular snpID, including columns for Theta, R, ALLELE_A, a1, a2, and genotype calls.
    result_frame : pandas.DataFrame
        DataFrame containing SNP analysis results, including concordance rates for WGS and Imputation.
    snp_id : str
        Identifier for the SNP being plotted.
    calls : pandas.DataFrame
        DataFrame containing gencall genotype calls for the particular snpID, including a column for 'Original Alleles'.
    output_file : str, optional
        File path to save the combined plot image. Default is None.
    show_fig : bool, optional
        If True, displays the combined figure. Default is True.
    
    Returns:
    -------
    plotly.graph_objs._figure.Figure
        Plotly Figure object representing the combined SNP scatter plots.

    """
    
    fig1 = plot_snp_alleles(data, snp_id, calls, "Imputed Alleles", result_frame)
    fig2 = plot_snp_alleles(data, snp_id, calls, "WGS Alleles", result_frame)
    fig3 = plot_snp_alleles(data, snp_id, calls, "Predicted Alleles", result_frame)


    combined_fig = sp.make_subplots(rows=1, cols=3, subplot_titles=("Imputed Genotypes", "WGS Genotypes", "Predicted Genotypes"))

    for trace in fig1['data']:
        trace.update(showlegend=False)
        combined_fig.add_trace(trace, row=1, col=1)
    for trace in fig2['data']:
        trace.update(showlegend=False)
        combined_fig.add_trace(trace, row=1, col=2)
    for trace in fig3['data']:
        trace.update(showlegend=False)
        combined_fig.add_trace(trace, row=1, col=3)
        
    combined_fig.update_xaxes(range=[-0.05, 1.05], row=1, col=1)
    combined_fig.update_yaxes(range=[0.00, 3.50], row=1, col=1)
    combined_fig.update_xaxes(range=[-0.05, 1.05], row=1, col=2)
    combined_fig.update_yaxes(range=[0.00, 3.50], row=1, col=2)
    combined_fig.update_xaxes(range=[-0.05, 1.05], row=1, col=3)
    combined_fig.update_yaxes(range=[0.00, 3.50], row=1, col=3)
    combined_fig.update_layout(height=500, width=1500, title_text=f"{snp_id} in {gene}")

    if data['ALLELE_A'].iloc[0] == 0:
        allele_A = data['a1'].iloc[0]
        allele_B = data['a2'].iloc[0]
    else:
        allele_A = data['a2'].iloc[0]
        allele_B = data['a1'].iloc[0]
        
    gt_map_legend = {
        f'{allele_A}{allele_A}': 'blue', 
        f'{allele_A}{allele_B}': 'orange', 
        f'{allele_B}{allele_B}': 'green'
    }
    call_map_legend = {
        f'gencall {allele_A}{allele_A}': 'cornflowerblue', 
        f'gencall {allele_A}{allele_B}': 'gold', 
        f'gencall {allele_B}{allele_B}': 'lightgreen'
    }
    
    gts = pd.concat([data[['Predicted Alleles', 'Imputed Alleles', 'WGS Alleles']]], axis=1).values.flatten()
    unique_gts = set(gts)
    call_gts = pd.concat([calls['Original Alleles']], axis=1).values.flatten()
    unique_call_gts = set(call_gts)

    gt_color_map_legend = {k: v for k, v in gt_map_legend.items() if k.split(' ')[-1] in unique_gts}
    call_gt_color_map_legend = {k: v for k, v in call_map_legend.items() if k.split(' ')[-1] in unique_call_gts}
    color_map_legend_specific = {**gt_color_map_legend, **call_gt_color_map_legend}

    for genotype, color in color_map_legend_specific.items():
        combined_fig.add_scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(color=color, opacity=1.0),
            name=f'{genotype}',
            showlegend=True,
            # visible='legendonly'
        )

    wgs_concordance = float(result_frame['concordance_wgs'].iloc[0])
    imp_concordance = float(result_frame['concordance_imputed'].iloc[0])
    annotation_text = ""
    if not np.isnan(wgs_concordance):
        annotation_text += f"Concordance with WGS: {wgs_concordance:.2f}%<br>"
    if not np.isnan(imp_concordance):
        annotation_text += f"Concordance with Imputation: {imp_concordance:.2f}%<br>"

    combined_fig.add_annotation(
        xref='x3', yref='y3',
        x=0.70, y=3.25, 
        # x=0.70, y=0.5, 
        text=annotation_text,
        showarrow=False,
        font=dict(size=11),
        row=1, col=3
        )

    if show_fig == True:
        combined_fig.show()

    combined_fig.write_image(output_file)
    
    return combined_fig
    
def main():

    args = parse_arguments()

    snp_list = []
    with open(args.snps, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            snp_list.append(row[0])

    genotypes = pd.read_csv(args.genotypes, header=0, sep=",", low_memory=False)
    gencalls = pd.read_csv(args.gencalls, header=0, sep=",", low_memory=False)
    calls = gencalls.loc[gencalls["GT"].isin(["AA","AB","BB"])].copy()
    annotations = pd.read_csv(args.annotations, header=0, sep=",")

    for snp in snp_list:
        try:
            snp_annotations = annotations.loc[annotations["snpID"] == snp].copy()
            gene = snp_annotations["gene"].iloc[0]
            snp_data = genotypes.loc[genotypes['snpID'] == snp].copy()
            snp_calls = calls.loc[calls['snpID'] == snp].copy()
    
            translator = str.maketrans('', '', string.punctuation)
            snp_string = snp.translate(translator) 
            
            snp_folder_path = f"{args.output_directory}/{gene}/{snp_string}"
            os.makedirs(snp_folder_path, exist_ok=True)
    
            combine_output_file = f"{snp_folder_path}/{snp_string}_combined.pdf"
            combined_fig = plot_combined_snp_plots(
                snp_data, snp_annotations, snp, gene, snp_calls, output_file=combine_output_file, show_fig=False
            )
    
            if args.save_individual:
                plot_snp_alleles_save_individual(snp_data, snp, snp_calls, "Imputed Alleles", snp_annotations, output_file =f"{snp_folder_path}/{snp_string}_imputed_genotypes.png")
                plot_snp_alleles_save_individual(snp_data, snp, snp_calls, "WGS Alleles", snp_annotations, output_file=f"{snp_folder_path}/{snp_string}_wgs_genotypes.png")
                plot_snp_alleles_save_individual(snp_data, snp, snp_calls, "Predicted Alleles", snp_annotations, output_file=f"{snp_folder_path}/{snp_string}_predicted_genotypes.png")
        except Exception as e:
            print(f"An exception with {snp} occurred.", file=sys.stderr)
            print(e, file=sys.stderr)
    
if __name__ == "__main__":
    main()