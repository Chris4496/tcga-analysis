from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data/processed/KIRC_merged_clin_RSEM.csv")

T = list(df["time_days"])

E = list(df["patient_dead"])

kmf = KaplanMeierFitter()
kmf.fit(T, E)

#Create survival function, cumultative density data, plot visualisations
print(kmf.survival_function_)
print(kmf.cumulative_density_)
kmf.plot_survival_function()
# plt.savefig('KMC2.png')
plt.show()