import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
import pickle
import os
import time

def cox_univariate_screening(df, clin, cache_name=None, retrieve_from_cache=True):
    # check if cache exists
    if retrieve_from_cache:
        # search for cache file
        for file in os.listdir("cache"):
            if f"{cache_name}_screening_result" in file:
                print(f"{cache_name} screening result cache found: cache/{file}")
                # load from cache
                with open(f'cache/{file}', 'rb') as f:
                    return pickle.load(f)
                    
    print(f"{cache_name} screening result cache not found")
    print("start screening")

    # merge the clinical data with the expression data
    merged = pd.merge(df, clin, on='patient_barcode', how='inner')

    # remove the 'patient_barcode' column
    del merged['patient_barcode']

    # remove columns that the values are all constant
    constant_columns = [col for col in merged.columns if merged[col].nunique() <= 1]
    merged = merged.drop(columns=constant_columns)

    # seperate predictor and outcome
    predictors = merged.drop(columns=['dead', 'time(day)'])
    outcome = merged[['dead', 'time(day)']]

    # a list to store the results
    univariate_results = []

    # progess counter
    counter = 0
    total = len(predictors.columns)

    # gene that failed to fit
    failed_features = []

    # loop through each gene
    for feature in predictors.columns:
        # add counter
        counter += 1
        
        # create new df that have the nessecary columns
        temp_df = pd.DataFrame({
            'T': outcome['time(day)'],
            'E': outcome['dead'],
            feature: predictors[feature]
        })
        
        # fit univariate model
        cph = CoxPHFitter()
        try:
            cph.fit(temp_df, duration_col="T", event_col="E")
        except ConvergenceError:
            print(f"Failed to fit {feature}")
            failed_features.append(feature)
            continue

        # store results
        if not cph.summary.empty:
            univariate_results.append(cph.summary)

        # indicate progress
        print(f"Progress: {counter} / {total}, -> Fitting {feature}", end="\r", flush=True)
        
    # drop failed genes
    print(f"Removed {len(failed_features)} features that failed to fit")

    # check for empty df in result list and delete
    for result in univariate_results:
        if result.empty:
            print("empty df")
            print(result)

    # save to cache
    with open(f'cache/{cache_name}_screening_result_{time.strftime("%Y%m%d_%H%M%S")}.pkl', 'wb') as f:
        pickle.dump(univariate_results, f)

    return univariate_results


def extract_top_features(df, screening_result, top=500):
    # extract the topX features
    top_features = [x.index[0] for x in sorted(screening_result, key=lambda x: x['-log2(p)'].iloc[0], reverse=True)[:top]]
    
    # extract the topX features from the original dataframe
    top_df = df[['patient_barcode', *top_features]]

    # reset the index
    top_df.reset_index(drop=True, inplace=True)

    return top_df