import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from pickle import load

def plot_KMC(df, gene_name, cph):
    # create new df that have the nessecary columns
    new = df[['time_days', 'patient_dead', gene_name]]

    kmf = KaplanMeierFitter()

    median_exp = df[gene_name].median()
    new['high_exp'] = df[gene_name] > median_exp

    # fit for low expression
    kmf.fit(new[new['high_exp'] == False]['time_days'], new[new['high_exp'] == False]['patient_dead'], label="low_exp")
    ax = kmf.plot(ci_show=False)

    # fit for high expression
    kmf.fit(new[new['high_exp'] == True]['time_days'], new[new['high_exp'] == True]['patient_dead'], label="high_exp")
    kmf.plot(ax=ax, ci_show=False)
    
    plt.title(f'{gene_name} KMC')
    plt.xlabel('Time (days)')
    plt.ylabel('Survival probability')

    # save figure
    plt.savefig(f'KMC_{gene_name.replace("|", "_")}.png')
    plt.close()



def main():
    # read the data
    merged = pd.read_csv("processed_data/KIRC_merged_clin_RSEM.csv")

    # load cph
    with open('KIRC_cph.pickle', 'rb') as f:
        cph = load(f)

    # get summary
    summary = cph.summary
    print(summary)

    plot_KMC(merged, 'DONSON|29980', cph)
    plot_KMC(merged, 'DVL3|1857', cph)
    plot_KMC(merged, 'POFUT2|23275', cph)



if __name__ == "__main__":
    main()