import pandas as pd
import pickle
import time
import os

def process_rppa_data_from_raw(retrieve_from_cache=True):
    if retrieve_from_cache:
        # search for cache file
        for file in os.listdir("cache"):
            if "KIRC_rppa_processed" in file:
                print(f"RPPA cache found: cache/{file}")
                # load from cache
                with open(f'cache/{file}', 'rb') as f:
                    return pickle.load(f)

        print("RPPA cache not found")
        print("start processing")

    df = pd.read_csv("data/RPPA_RBN", sep="\t", low_memory=False).transpose()
    
    df = df.reset_index()

    df.columns = df.iloc[0].rename(None)

    df.columns = ['patient_barcode', *df.columns[1:].to_list()]

    df = df.drop(df.index[0])

    # For the column "patient_barcode", we only need the first 12 characters
    df['patient_barcode'] = df['patient_barcode'].str[:12]

    df.reset_index(drop=True, inplace=True)
    
    # save to cache (include date time)
    with open(f'cache/KIRC_rppa_processed_{time.strftime("%Y%m%d_%H%M%S")}.pkl', 'wb') as f:
        pickle.dump(df, f)

    return df


if __name__ == "__main__":
    df = process_rppa_data_from_raw()
    print(df.head())