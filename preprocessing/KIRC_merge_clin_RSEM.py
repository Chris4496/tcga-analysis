import pandas as pd

def main():
    clin = pd.read_csv("data/processed/KIRC_clinical.csv")
    rsem = pd.read_csv("data/processed/KIRC_RSEM.csv")

    rsem['patient_barcode'] = rsem['patient_barcode'].apply(lambda x: x[:12])

    clin['patient_barcode'] = clin['patient_barcode'].apply(lambda x: x.upper())

    df = pd.merge(rsem, clin, on="patient_barcode", how="inner")
    return df


if __name__ == "__main__":
    df = main()
    
    # save to csv
    df.to_csv("data/processed/KIRC_merged_clin_RSEM.csv", index=False)