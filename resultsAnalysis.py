import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from pickle import load

def plot_KMC(df, gene_name):
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
    plt.savefig(f'plot_out/KMC_{gene_name.replace("|", "_")}.png')
    plt.close()

    # show figure
    # plt.show()

    # # reset figure
    # plt.clf()
    # plt.close()
    




def main():
    # read the data
    merged = pd.read_csv("processed_data/KIRC_merged_clin_RSEM.csv")
    
    # read KIRC cph
    with open('KIRC_cph.pickle', 'rb') as f:
        cph_list = load(f)

    # sort cph list by -log2(p)
    cph_list.sort(key=lambda x: x['-log2(p)'].to_list()[0], reverse=True)

    # print top 10 cph
    for i in range(10):
        print(cph_list[i])


    # plot KMC for top 10 cph
    for i in range(10):
        plot_KMC(merged, cph_list[i].index[0])

if __name__ == "__main__":
    main()