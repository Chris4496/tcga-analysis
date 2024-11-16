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

    # run cox univariate screening
    rsem_screening_result = cox_univariate_screening(rsem, clin)




if __name__ == "__main__":
    main()