import pandas as pd
import numpy as np
from pickle import load, dump


def main():
    # read the lasso cox regression results
    with open('result_cache/lasso_cox_regression_top1000.pickle', 'rb') as f:
        lasso_cox_regression = load(f)

    lasso_cox_regression_sored = lasso_cox_regression.sort_values(by='-log2(p)', ascending=False)

    print(lasso_cox_regression_sored.head())

    print(list(lasso_cox_regression_sored.index))


if __name__ == "__main__":
    main()