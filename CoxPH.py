import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter

from pickle import load, dump

top = 500
    
def CoxRegression(df, gene_names, penalty=0.1, l1_ratio=0.1):
    # create new df that have the nessecary columns
    temp = df[['time_days', 'patient_dead', *gene_names]]

    # fit cox regression
    model = CoxPHFitter(penalizer=penalty, l1_ratio=l1_ratio)
    model.fit(temp, duration_col="time_days", event_col="patient_dead")

    # model.print_summary()

    return model


def main():
    # read the data
    merged = pd.read_csv("data/processed/KIRC_merged_clin_RSEM.csv")
    
    # read univariate results
    with open('cache/univariate_cph_results_filtered.pkl', 'rb') as f:
        univariate_results = load(f)
        
    univariate_results_sorted = sorted(univariate_results, key=lambda df: df['-log2(p)'].iloc[0], reverse=True)

    # Hyperparameters tuning
    top_x = [10, 20, 30, 50, 100, 200, 300, 400, 500, 1000]
    penalties = [10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.001]
    l1_ratios = [1.0, 0.5, 0.1, 0.05, 0.01]

    for tx in top_x:
        for penalty in penalties:
            for l1_ratio in l1_ratios:
                print(f"Fitting CoxPH with penalty={penalty}, l1_ratio={l1_ratio}, top={tx}")
                try:
                    model = CoxRegression(merged, [x.index[0] for x in univariate_results_sorted[:tx]], penalty=penalty, l1_ratio=l1_ratio)
                except:
                    print(f"Failed to fit CoxPH with penalty={penalty}, l1_ratio={l1_ratio}, top={tx}")
                    continue
                
                # save results
                with open(f'models/cox_ph_top{tx}_penalty{penalty}_l1_ratio{l1_ratio}.pkl', 'wb') as f:
                    dump(model, f)






if __name__ == "__main__":
    main()