import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import k_fold_cross_validation
from pprint import pprint

from pickle import load, dump

top = 500
    
# def CoxRegression(df, gene_names, penalty=0.1, l1_ratio=0.1):
#     # create new df that have the nessecary columns
#     temp = df[['time_days', 'patient_dead', *gene_names]]

#     # fit cox regression
#     model = CoxPHFitter(penalizer=penalty, l1_ratio=l1_ratio)
#     model.fit(temp, duration_col="time_days", event_col="patient_dead")

#     # model.print_summary()

#     return model


def main():
    # read the data
    merged = pd.read_csv("data/processed/KIRC_merged_clin_RSEM.csv")
    
    # read univariate results
    with open('cache/univariate_cph_results_filtered.pkl', 'rb') as f:
        univariate_results = load(f)
        
    univariate_results_sorted = sorted(univariate_results, key=lambda df: df['-log2(p)'].iloc[0], reverse=True)

    # Hyperparameters tuning
    top_x = [10, 20, 30, 50, 100, 200, 300, 400, 500, 1000]
    penalties = [10, 5, 1, 0.5, 0.1]
    l1_ratios = [1.0, 0.8, 0.5, 0.3, 0.1, 0]

    # top 300 genes
    top = [x.index[0] for x in univariate_results_sorted[:30]]

    # get top 300 genes
    temp = merged[['time_days', 'patient_dead', *top]]

    cph_list = list()

    for penatly in penalties:
        for l1 in l1_ratios:
            cph = CoxPHFitter(penalizer=penatly, l1_ratio=l1)
            cph_list.append(cph)

    scores = k_fold_cross_validation(cph_list, temp, duration_col='time_days', event_col='patient_dead', scoring_method='concordance_index')

    # take the mean
    scores = [sum(x)/len(x) for x in scores]

    scores = np.array(scores).reshape(len(penalties), len(l1_ratios))

    print(scores)

    # save results
    with open('cache/c_index_top30_penalty_vs_l1_ratio.pkl', 'wb') as f:
        dump(scores, f)


if __name__ == "__main__":
    main()