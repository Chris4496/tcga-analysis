import pandas as pd
import numpy as np
from pickle import load, dump
import matplotlib.pyplot as plt


def main():
    # read data
    with open('cache/c_index_top30_penalty_vs_l1_ratio.pkl', 'rb') as f:
        c_index = load(f)

    print(c_index)

    penalties = [10, 5, 1, 0.5, 0.1]
    l1_ratios = [1.0, 0.8, 0.5, 0.3, 0.1, 0]

    # Create the heatmap
    plt.imshow(c_index, cmap='viridis', interpolation='nearest')

    # Add colorbar
    plt.colorbar(label='Value')

    # Set the ticks and labels for the x-axis and y-axis
    plt.xticks(ticks=np.arange(len(l1_ratios)), labels=l1_ratios)
    plt.yticks(ticks=np.arange(len(penalties)), labels=penalties)

    # Add labels and title
    plt.xlabel('L1 Ratios')
    plt.ylabel('Penalties')
    plt.title('Heatmap with Penalties and L1 Ratios')

    # Save image
    plt.savefig('output/log_likihood_heatmap_penalty_vs_l1_ratio.png', dpi=300)

if __name__ == "__main__":
    main()