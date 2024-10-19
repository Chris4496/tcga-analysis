import pandas as pd
import numpy as np
from pickle import load, dump
import matplotlib.pyplot as plt


def main():
    # Hyperparameters
    top_x = [10, 20, 30, 50, 100, 200, 300, 400, 500, 1000]
    penalties = [10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001]
    l1_ratios = [1.0, 0.5, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001]

    for tx in top_x:
        c_indices = list()
        for penalty in penalties:
            penrow = list()
            for l1_ratio in l1_ratios:
                try:
                    with open(f'models/cox_ph_top{tx}_penalty{penalty}_l1_ratio{l1_ratio}.pkl', 'rb') as f:
                        model = load(f)
                    c_index = model.concordance_index_
                except:
                    print(f"Failed to load model for penalty={penalty}, l1_ratio={l1_ratio}, top={tx}")
                    c_index = np.nan
                penrow.append(c_index)
            c_indices.append(penrow)

        # plot heatmap
        plt.figure(figsize=(10, 6))  # Set the figure size
        plt.imshow(c_indices, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.xticks(range(len(l1_ratios)), [f'l1_ratio={l1_ratio}' for l1_ratio in l1_ratios], rotation=90)
        plt.yticks(range(len(penalties)), [f'penalty={penalty}' for penalty in penalties])
        plt.xlabel('L1 ratio')
        plt.ylabel('Penalty')
        plt.title(f'CoxPH Concordance Index Top {tx}')
        plt.tight_layout()  # Adjust layout to prevent label cutoffs
        plt.show()

        # reset figure
        plt.clf()
        plt.close()


if __name__ == "__main__":
    main()