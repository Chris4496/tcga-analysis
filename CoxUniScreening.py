import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from pickle import dump

def main():
    # read the data
    merged = pd.read_csv("data/processed/KIRC_merged_clin_RSEM.csv")

    # remove the 'patient_barcode' column
    del merged['patient_barcode']

    # remove columns that the values are all constant
    constant_columns = [col for col in merged.columns if merged[col].nunique() <= 1]
    print(f"Constant columns: {constant_columns}")
    merged = merged.drop(columns=constant_columns)

    # save to cache
    with open('cache/constant_columns.pkl', 'wb') as f:
        dump(constant_columns, f)

    # seperate predictor and outcome
    predictors = merged.drop(columns=['patient_dead', 'time_days'])
    outcome = merged[['patient_dead', 'time_days']]

    # a list to store the results
    univariate_results = []

    # progess counter
    gene_counter = 0
    total = len(predictors.columns)

    # gene that failed to fit
    failed_genes = []

    # loop through each gene
    for gene in predictors.columns:
        # add counter
        gene_counter += 1
        
        # create new df that have the nessecary columns
        temp_df = pd.DataFrame({
            'T': outcome['time_days'],
            'E': outcome['patient_dead'],
            gene: predictors[gene]
        })
        
        # fit univariate model
        cph = CoxPHFitter()
        try:
            cph.fit(temp_df, duration_col="T", event_col="E")
        except ConvergenceError:
            print(f"Failed to fit {gene}")
            failed_genes.append(gene)
            continue

        # store results
        univariate_results.append(cph.summary)

        # indicate progress
        print(f"Progress: {gene_counter} / {total}, -> Fitting {gene}")

    # store failed genes
    print('Failed genes: ', failed_genes)
    with open('cache/failed_genes.pkl', 'wb') as f:
        dump(failed_genes, f)
        
    # drop failed genes
    print(f"Removed {len(failed_genes)} genes")

    # store results
    print('Saving results')
    with open('cache/univariate_cph_results.pkl', 'wb') as f:
        dump(univariate_results, f)

    
if __name__ == "__main__":
    main()