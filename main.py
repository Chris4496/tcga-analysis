import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from pickle import dump

def main():
    # read the data
    merged = pd.read_csv("processed_data/KIRC_merged_clin_RSEM.csv")
    
    # get shape
    shape = merged.shape
    print(shape)

    # extract and remove the 'patient_barcode' column
    patientBarcode = merged['patient_barcode']
    del merged['patient_barcode']

    # remove "gene_id" column
    del merged['gene_id']

    # make new dataframe that contains only the columns of "time_days" and "patient_dead"
    new = merged[['time_days', 'patient_dead', 'C15orf42|90381', 'CCNF|899', 'DONSON|29980', 'DVL3|1857', 'POFUT2|23275']]


    # initialize cph
    cph = CoxPHFitter()
    cph.fit(new, duration_col="time_days", event_col="patient_dead")

    # save cph
    with open('KIRC_cph.pickle', 'wb') as f:
        dump(cph, f)

    # print summary
    print(cph.summary)
    
if __name__ == "__main__":
    main()