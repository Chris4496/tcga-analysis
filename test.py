import pandas as pd
import numpy as np

df = pd.read_csv("processed_data/KIRC_merged_clin_RSEM.csv")


gene = df['C20orf185|359710']


# check if gene contains nan values
print(gene.isna().sum())