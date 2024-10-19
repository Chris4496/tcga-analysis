import pandas as pd
import numpy as np
from pickle import load, dump
import matplotlib.pyplot as plt

p_threshold = 0.05

def main():
    # read univariate results
    with open('cache/univariate_cph_results.pkl', 'rb') as f:
        univariate_results = load(f)

    # drop the 19987 item
    univariate_results.pop(19987)
        
    # filter results
    univariate_results_filtered = [x for x in univariate_results if x['p'].iloc[0] < p_threshold]

    # store results
    print('Saving results')
    with open('cache/univariate_cph_results_filtered.pkl', 'wb') as f:
        dump(univariate_results_filtered, f)

if __name__ == "__main__":
    main()