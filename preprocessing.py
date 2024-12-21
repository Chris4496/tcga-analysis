import pandas as pd
from sklearn.impute import KNNImputer
import functools
import os
import hashlib
import pickle
from caching_script import cache_result


@cache_result(verbose=0)
def process_clinical_data_from_raw():
    df = pd.read_csv("data/survival_KIRC_survival.txt", sep="\t")
    
    df = df[['_PATIENT', 'OS', 'OS.time']]

    # rename columns
    df.columns = ['patient_barcode', 'dead', 'time(day)']

    # delete duplicate rows
    df = df.drop_duplicates(subset=['patient_barcode'], keep="last")

    return df

@cache_result(verbose=0)
def process_copyno_data_from_raw():
    df = pd.read_csv("data/Gistic2_CopyNumber_Gistic2_all_data_by_genes", sep="\t", low_memory=False).transpose()
    
    df = df.reset_index()

    df.columns = df.iloc[0].rename(None)

    df.columns = ['patient_barcode', *df.columns[1:].to_list()]

    df = df.drop(df.index[0])

    # For the column "patient_barcode", we only need the first 12 characters
    df['patient_barcode'] = df['patient_barcode'].str[:12]
  
    df.reset_index(drop=True, inplace=True)

    # delete duplicate rows
    df = df.drop_duplicates(subset=['patient_barcode'], keep="last")

    return df

@cache_result(verbose=0)
def process_mirna_data_from_raw():
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

    return df

@cache_result(verbose=0)
def process_rppa_data_from_raw():
    df = pd.read_csv("data/RPPA_RBN", sep="\t", low_memory=False).transpose()
    
    df = df.reset_index()

    df.columns = df.iloc[0].rename(None)

    df.columns = ['patient_barcode', *df.columns[1:].to_list()]

    df = df.drop(df.index[0])

    # For the column "patient_barcode", we only need the first 12 characters
    df['patient_barcode'] = df['patient_barcode'].str[:12]

    df.reset_index(drop=True, inplace=True)

    # delete duplicate rows
    df = df.drop_duplicates(subset=['patient_barcode'], keep="last")

    return df

@cache_result(verbose=0)
def process_rsem_data_from_raw():
    df = pd.read_csv("data/HiSeqV2", sep="\t", low_memory=False).transpose()

    df = df.reset_index()

    df.columns = df.iloc[0].rename(None)

    df.columns = ['patient_barcode', *df.columns[1:].to_list()]

    df = df.drop(df.index[0])

    # For the column "patient_barcode", we only need the first 12 characters
    df['patient_barcode'] = df['patient_barcode'].str[:12]

    df.reset_index(drop=True, inplace=True)

    # delete duplicate rows
    df = df.drop_duplicates(subset=['patient_barcode'], keep="last")

    return df
