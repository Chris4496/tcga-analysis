import pandas as pd
import numpy as np
from pickle import load, dump
import matplotlib.pyplot as plt


def main():
    # read top 1000 genes
    with open('models/cox_ph_top30_penalty1_l1_ratio0.5.pkl', 'rb') as f:
        cph = load(f)

    lasso_results = cph.summary

    # sort by coef
    lasso_results_sorted = lasso_results.sort_values(by='coef', ascending=True)

    # top x
    top = lasso_results_sorted.head(30)

    possible_genes = ['C15orf42|90381', 'CCNF|899', 'DONSON|29980', 'DVL3|1857', 'POFUT2|23275']

    # Create the plot
    plt.figure(figsize=(10, 6))  # Set the figure size

    # Bar plot
    plt.bar(top.index, top['p'], color='skyblue')

    # Add labels and title
    plt.xlabel('Gene', fontsize=12)
    plt.ylabel('Coefficient (coef)', fontsize=12)
    plt.title('Top 20 Genes by Coefficient', fontsize=14)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)

    # Show the plot
    plt.tight_layout()  # Adjust layout to prevent label cutoffs
    plt.show()

if __name__ == "__main__":
    main()

# def plot_KMC(df, gene_name):
#     # create new df that have the nessecary columns
#     new = df[['time_days', 'patient_dead', gene_name]]

#     kmf = KaplanMeierFitter()

#     median_exp = df[gene_name].median()
#     new['high_exp'] = df[gene_name] > median_exp

#     # fit for low expression
#     kmf.fit(new[new['high_exp'] == False]['time_days'], new[new['high_exp'] == False]['patient_dead'], label="low_exp")
#     ax = kmf.plot(ci_show=False)

#     # fit for high expression
#     kmf.fit(new[new['high_exp'] == True]['time_days'], new[new['high_exp'] == True]['patient_dead'], label="high_exp")
#     kmf.plot(ax=ax, ci_show=False)
    
#     plt.title(f'{gene_name} KMC')
#     plt.xlabel('Time (days)')
#     plt.ylabel('Survival probability')

#     # save figure
#     plt.savefig(f'plot_out/KMC_{gene_name.replace("|", "_")}.png')
#     plt.close()

#     # show figure
#     # plt.show()

#     # # reset figure
#     # plt.clf()
#     # plt.close()