import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
import pickle
import os
import time

def cox_univariate_screening(df, clin, retrieve_from_cache=True):
    if retrieve_from_cache:
        # search for cache file
        for file in os.listdir("cache"):
            if f"KIRC_RSEM_screening_result" in file:
                print(f"RSEM screening result cache found: cache/{file}")
                # load from cache
                with open(f'cache/{file}', 'rb') as f:
                    return pickle.load(f)
                    
    print(f"RSEM screening result cache not found")
    print("start screening")

    # merge the clinical data with the expression data
    merged = pd.merge(df, clin, on='patient_barcode', how='inner')

    # remove the 'patient_barcode' column
    del merged['patient_barcode']

    # remove columns that the values are all constant
    constant_columns = [col for col in merged.columns if merged[col].nunique() <= 1]
    merged = merged.drop(columns=constant_columns)

    # seperate predictor and outcome
    predictors = merged.drop(columns=['dead', 'time(day)'])
    outcome = merged[['dead', 'time(day)']]

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
            'T': outcome['time(day)'],
            'E': outcome['dead'],
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
        print(f"Progress: {gene_counter} / {total}, -> Fitting {gene}", end="\r", flush=True)
        
    # drop failed genes
    print(f"Removed {len(failed_genes)} genes")

    # save to cache
    with open(f'cache/KIRC_RSEM_screening_result_{time.strftime("%Y%m%d_%H%M%S")}.pkl', 'wb') as f:
        pickle.dump(univariate_results, f)

    return univariate_results

    
if __name__ == "__main__":
    main()