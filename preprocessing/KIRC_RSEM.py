import pandas as pd

def main():
    df = pd.read_csv("data/raw/HiSeqV2", sep="\t", low_memory=False).transpose()
    # set first row as column names
    df.columns = df.iloc[0]

    # drop first row
    df = df.drop(df.index[0])
    
    # make index as column
    df.reset_index(inplace=True)

    # rename first column to patient barcode
    df.rename(columns={"index": "patient_barcode"}, inplace=True)


    return df


if __name__ == "__main__":
    df = main()
    df.to_csv("data/processed/KIRC_RSEM.csv", index=False)