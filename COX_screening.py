import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from pickle import dump

def main():
    # read the data
    merged = pd.read_csv("processed_data/KIRC_merged_clin_RSEM.csv")

    # extract and remove the 'patient_barcode' column
    patientBarcode = merged['patient_barcode']
    del merged['patient_barcode']

    # remove "gene_id" column
    del merged['gene_id']

    # remove columns that the values are all constant
    constant_columns = [col for col in merged.columns if merged[col].nunique() <= 1]
    print(f"Constant columns: {constant_columns}")
    merged = merged.drop(columns=constant_columns)

    # remove columns that have all NA values
    merged = merged.dropna()

    # remove columns with low variance
    # Set a threshold for low variance
    variance_threshold = 1e-4

    variances = merged.var()
    low_variance_cols = variances[variances < variance_threshold].index

    merged = merged.drop(columns=low_variance_cols)


    # seperate predictor and outcome
    predictors = merged.drop(columns=['patient_dead', 'time_days'])
    outcome = merged[['patient_dead', 'time_days']]

    # a list to store the results
    univariate_results = []

    # progess counter
    gene_counter = 0
    total = len(predictors.columns)

    # loop through each gene
    for gene in predictors.columns:
        # indicate progress
        print(f"Progress: {gene_counter} / {total}, -> Fitting {gene}")

        # create new df that have the nessecary columns
        temp_df = pd.DataFrame({
            'T': outcome['time_days'],
            'E': outcome['patient_dead'],
            gene: predictors[gene]
        })
        
        # fit univariate model
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(temp_df, duration_col="T", event_col="E")

        # store results
        univariate_results.append(cph.summary)

        # add counter
        gene_counter += 1
        
    # store results
    print('Saving results')
    with open('KIRC_cph.pickle', 'wb') as f:
        dump(univariate_results, f)
    
if __name__ == "__main__":
    main()