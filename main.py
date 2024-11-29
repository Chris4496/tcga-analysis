import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# import functions
from preprocessing.process_clinical import process_clinical_data_from_raw
from preprocessing.process_RSEM import process_rsem_data_from_raw
from preprocessing.process_copyno import process_copyno_data_from_raw
from preprocessing.process_miRNA import process_mirna_data_from_raw
from preprocessing.process_RPPA import process_rppa_data_from_raw
from CoxUniScreening import cox_univariate_screening
from CoxUniScreening import extract_top_features

# import models
from sksurv.linear_model import CoxnetSurvivalAnalysis
from pprint import pprint

def custom_penalizer(feature_range, penalties):
    # create custom penalizer
    alpha = np.zeros(feature_range[-1])

    # for feature 0-500 have penalty of 0.1
    for i in range(feature_range[0], feature_range[1]):
        alpha[i] = penalties[0]

    # for feature 500-1000 have penalty of 0.5
    for i in range(feature_range[1], feature_range[2]):
        alpha[i] = penalties[1]

    # for feature 1507-1638 have penalty of 0.3
    for i in range(feature_range[2], feature_range[3]):
        alpha[i] = penalties[2]

    # for feature 1638-2000 have penalty of 0.4
    for i in range(feature_range[3], feature_range[4]):
        alpha[i] = penalties[3]

    return alpha


def main():
    clin = process_clinical_data_from_raw()
    rsem = process_rsem_data_from_raw()
    copyno = process_copyno_data_from_raw()
    mirna = process_mirna_data_from_raw()
    rppa = process_rppa_data_from_raw()

    # print out the features size of each dataset
    print(f"Clinical: {clin.shape}")
    print(f"RSEM: {rsem.shape}")
    print(f"CopyNo: {copyno.shape}")
    print(f"miRNA: {mirna.shape}")
    print(f"RPPA: {rppa.shape}")
    # Clinical: (944, 3)
    # RSEM: (606, 20531)
    # CopyNo: (528, 24777)
    # miRNA: (311, 508)
    # RPPA: (454, 132)

    # run cox univariate screening
    rsem_screening_result = cox_univariate_screening(rsem, clin, "KIRC_RSEM")
    copyno_screening_result = cox_univariate_screening(copyno, clin, "KIRC_COPYNO")

    # extract top 500 features
    rsem_top500 = extract_top_features(rsem, rsem_screening_result, top=500)
    copyno_top500 = extract_top_features(copyno, copyno_screening_result, top=500)

    # delete repeated rows
    rsem_top500 = rsem_top500.drop_duplicates(subset=['patient_barcode'], keep="last")
    copyno_top500 = copyno_top500.drop_duplicates(subset=['patient_barcode'], keep="last")
    mirna = mirna.drop_duplicates(subset=['patient_barcode'], keep="last")
    rppa = rppa.drop_duplicates(subset=['patient_barcode'], keep="last")

    # get column indices of each data type
    rsem_cols = rsem_top500.shape[1] - 1
    copyno_cols = copyno_top500.shape[1] - 1
    mirna_cols = mirna.shape[1] - 1
    rppa_cols = rppa.shape[1] - 1

    print(f"rsem_cols: {rsem_cols}")
    print(f"copyno_cols: {copyno_cols}")
    print(f"mirna_cols: {mirna_cols}")
    print(f"rppa_cols: {rppa_cols}")

    # merge all genomic features
    merged = clin.merge(rsem_top500, on='patient_barcode', how='inner')
    merged = merged.merge(copyno_top500, on='patient_barcode', how='inner')
    merged = merged.merge(mirna, on='patient_barcode', how='inner')
    merged = merged.merge(rppa, on='patient_barcode', how='inner')

    # delete repeated rows
    merged = merged.drop_duplicates(subset=['patient_barcode'], keep="last")

    print(merged.head())

    # create custom penalizer
    penalty_factors = custom_penalizer([0, 500, 1000, 1507, 1638], [0.1, 0.5, 0.3, 0.4])

    # seperate predictor and outcome
    X = merged.drop(columns=['dead', 'time(day)', 'patient_barcode'])
    y = merged[['dead', 'time(day)']]

    # convert the first column of y to boolean
    y = y.replace({'dead': {0: False, 1: True}})

    # perform cox regression with custom penalizer
    cph = CoxnetSurvivalAnalysis(penalty_factor=penalty_factors)
    cph.fit(X, y)

    cph.get_params()


    
if __name__ == "__main__":
    main()