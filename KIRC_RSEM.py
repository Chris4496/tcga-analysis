import pandas as pd

def main():
    df = pd.read_csv("raw_data/KIRCRN~1.txt", sep="\t", low_memory=False).transpose()

    df = df.reset_index()

    # remove first row and set first row as column names
    df.columns = df.iloc[0]  # Set the new first row as header
    df = df[1:].reset_index(drop=True)  # Remove the new first row and reset the index

    return df


if __name__ == "__main__":
    df = main()
    df.to_csv("processed_data/KIRC_RSEM.csv", index=False)