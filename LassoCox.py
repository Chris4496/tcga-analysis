import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from pickle import load, dump

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
    
def LassoCoxRegression(df, gene_names):
    # create new df that have the nessecary columns
    new = df[['time_days', 'patient_dead', *gene_names]]

    # fit lasso cox regression
    lasso = CoxPHFitter(penalizer=0.1, l1_ratio=1.0)
    lasso.fit(new, duration_col="time_days", event_col="patient_dead", show_progress=True)

    # print results
    print(lasso.summary)

    # save results
    with open(f'result_cache/lasso_cox_regression_top1000.pickle', 'wb') as f:
        dump(lasso.summary, f)



def main():
    # read the data
    merged = pd.read_csv("processed_data/KIRC_merged_clin_RSEM.csv")
    
    # read KIRC cph
    with open('result_cache/KIRC_cph.pickle', 'rb') as f:
        cph_list = load(f)

    cph_list_sorted = sorted(cph_list, key=lambda df: df['-log2(p)'].iloc[0], reverse=True)

    # Extract the top 100 genes
    top_100_genes = cph_list_sorted[:50]

    # Extract the top 100 genes names
    top_100_genes_names = [x.index[0] for x in top_100_genes]

    LassoCoxRegression(merged, top_100_genes_names)




if __name__ == "__main__":
    main()