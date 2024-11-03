from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import pandas as pd

def plot_km_curves(data, genes, time_col, event_col):
    """
    Plots Kaplan-Meier curves for a list of genes.
    
    Parameters:
    - data: pandas DataFrame containing the survival information and gene expression levels.
    - genes: list of strings, each representing a gene name (column in the data).
    - time_col: string, name of the column representing survival time.
    - event_col: string, name of the column representing event occurrence (1 = event, 0 = censored).
    """
    kmf = KaplanMeierFitter()
    
    # Loop through each gene and plot the Kaplan-Meier curve
    for gene in genes:
        # Split the data based on the median expression of the gene
        median_expr = data[gene].median()
        high_expr = data[data[gene] > median_expr]
        low_expr = data[data[gene] <= median_expr]
        
        # Fit and plot Kaplan-Meier curve for high expression
        kmf.fit(high_expr[time_col], event_observed=high_expr[event_col], label=f'{gene} High Expression')
        ax = kmf.plot_survival_function()
        
        # Fit and plot Kaplan-Meier curve for low expression
        kmf.fit(low_expr[time_col], event_observed=low_expr[event_col], label=f'{gene} Low Expression')
        kmf.plot_survival_function(ax=ax)
        
        # Add plot titles and labels
        plt.title(f"Kaplan-Meier Curves for {gene}")
        plt.xlabel("Time (days)")
        plt.ylabel("Survival Probability")
        plt.legend()
        
        # save to output folder
        plt.savefig(f'output/KMC_{gene.replace("|", "_")}.png')
        plt.close()

        # reset figure
        plt.clf()
        plt.close()

df = pd.read_csv("data/processed/KIRC_merged_clin_RSEM.csv")

possible_genes = ['ANAPC7', 'STRADA', 'MARS', 'SEC61A2', 'OTX1', 'C12orf32', 'COPS7B', 'DONSON', 'MBOAT7', 'SLC5A6', 'EIF4EBP2', 'SORBS2', 'AMOT']

plot_km_curves(df, possible_genes, 'time_days', 'patient_dead')