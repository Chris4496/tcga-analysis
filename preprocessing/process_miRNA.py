import pandas as pd
import pickle
import time
import os
from sklearn.impute import KNNImputer

def process_mirna_data_from_raw(retrieve_from_cache=True):
    if retrieve_from_cache:
        # search for cache file
        for file in os.listdir("cache"):
            if "KIRC_mirna_processed" in file:
                print(f"miRNA cache found: cache/{file}")
                # load from cache
                with open(f'cache/{file}', 'rb') as f:
                    return pickle.load(f)

        print("miRNA cache not found")
        print("start processing")

    df = pd.read_csv("data/miRNA_HiSeq_gene", sep="\t", low_memory=False).transpose()
    
    df = df.reset_index()

    df.columns = df.iloc[0].rename(None)

    df.columns = ['patient_barcode', *df.columns[1:].to_list()]

    df = df.drop(df.index[0])

    df.reset_index(drop=True, inplace=True)

    # For the column "patient_barcode", we only need the first 12 characters
    df['patient_barcode'] = df['patient_barcode'].str[:12]

    df.reset_index(drop=True, inplace=True)

    # Or remove microRNAs with >30% missing values
    threshold = 30
    df = df.loc[:, df.isnull().mean() * 100 < threshold]

    patient_barcode = df['patient_barcode']

    data_for_imputation = df.drop('patient_barcode', axis=1)
    
    # Impute missing values with KNN
    imputer = KNNImputer(n_neighbors=5)
    imputed_data = imputer.fit_transform(data_for_imputation)

    # Convert back to DataFrame with column names
    df = pd.DataFrame(imputed_data, columns=data_for_imputation.columns)

    # Add back the patient_barcode column
    df.insert(0, 'patient_barcode', patient_barcode)
    
    # delete duplicate rows
    df = df.drop_duplicates(subset=['patient_barcode'], keep="last")

    # save to cache (include date time)
    with open(f'cache/KIRC_mirna_processed_{time.strftime("%Y%m%d_%H%M%S")}.pkl', 'wb') as f:
        pickle.dump(df, f)


    return df


if __name__ == "__main__":
    df = process_mirna_data_from_raw()
    print(df.head())