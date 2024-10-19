from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

from preprocessing_util.KIRC_clin import main as getDf

df = getDf()

T = list(df["time_days"])

E = list(df["patient_dead"])

kmf = KaplanMeierFitter()
kmf.fit(T, E)

#Create survival function, cumultative density data, plot visualisations
print(kmf.survival_function_)
print(kmf.cumulative_density_)
kmf.plot_survival_function()
plt.savefig('KMC2.png')