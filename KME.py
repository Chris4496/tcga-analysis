from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

from main import main as getDf

df = getDf()

T = list()
for index, row in df.iterrows():
    if row["patient_dead"] == 1:
        time = row["days_to_death"]
    else:
        time = row["days_to_last_followup"]
    T.append(time)

E = list(df["patient_dead"])

kmf = KaplanMeierFitter()
kmf.fit(T, E)

#Create survival function, cumultative density data, plot visualisations
print(kmf.survival_function_)
print(kmf.cumulative_density_)
kmf.plot_survival_function()
plt.savefig('KMC.png')
