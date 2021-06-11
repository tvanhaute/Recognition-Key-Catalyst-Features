import pandas as pd
from random import random

metal_conc_min = 0.4
metal_conc_max = 65
metal_conc_delta = metal_conc_max - metal_conc_min

acid_conc_min = 114
acid_conc_max = 1458
acid_conc_delta = acid_conc_max - acid_conc_min

gamma_min = 0.2
gamma_max = 0.84
gamma_delta = gamma_max - gamma_min

epsilon_min = 74
epsilon_max = 4500
epsilon_delta = epsilon_max - epsilon_min

datapunten = 100000

columns = ["Metal_conc", "Acid_conc", "Gamma", "Epsilon", "Yield"]
df = pd.DataFrame(columns=columns)


for i in range(datapunten):

    metal_random = random()
    acid_random = random()
    gamma_random = random()
    epsilon_random = random()

    Metal_conc = metal_conc_min + metal_random * metal_conc_delta
    Acid_conc = acid_conc_min + acid_random * acid_conc_delta
    Gamma = gamma_min + gamma_random * gamma_delta
    Epsilon = epsilon_min + epsilon_random * epsilon_delta

    Yield = Gamma / (1 + Epsilon * (Acid_conc / Metal_conc))
    df.loc[i] = [Metal_conc, Acid_conc, Gamma, Epsilon, Yield]

df.to_csv('DatasetsSTDEVMetalConc/SyntheticDataset' + str(datapunten) + '.csv', index=False)
