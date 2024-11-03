import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import k_fold_cross_validation
from pprint import pprint

from pickle import load, dump

top = 500
    
def CoxRegression(df, gene_names, penalty=0.1, l1_ratio=0.1):
    # create new df that have the nessecary columns
    temp = df[['time_days', 'patient_dead', *gene_names]]

    # fit cox regression
    model = CoxPHFitter(penalizer=penalty, l1_ratio=l1_ratio)
    model.fit(temp, duration_col="time_days", event_col="patient_dead")

    model.print_summary()

    return model


def main():
    # read the data
    merged = pd.read_csv("data/processed/KIRC_merged_clin_RSEM.csv")
    
    # read univariate results
    with open('cache/univariate_cph_results_filtered.pkl', 'rb') as f:
        univariate_results = load(f)
        
    univariate_results_sorted = sorted(univariate_results, key=lambda df: df['-log2(p)'].iloc[0], reverse=True)

    model = CoxRegression(merged, [x.index[0] for x in univariate_results_sorted[:50]], penalty=1, l1_ratio=0.1)

    summary = model.summary

    # sort by coef
    summary_sorted = summary.sort_values(by='coef', ascending=False)

    print(summary_sorted.head(10))



if __name__ == "__main__":
    main()