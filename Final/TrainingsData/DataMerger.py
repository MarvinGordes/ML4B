import pandas as pd

a = pd.read_csv('TrainingsData/k1g.csv')
b = pd.read_csv('TrainingsData/k2g.csv')
c = pd.read_csv('TrainingsData/k3g.csv')
d = pd.read_csv('TrainingsData/k4g.csv')
e = pd.read_csv('TrainingsData/k5g.csv')
i = pd.read_csv('TrainingsData/k6g.csv')
j = pd.read_csv('TrainingsData/k7g.csv')
f = pd.read_csv('TrainingsData/KnockLangM1.csv')
g = pd.read_csv('TrainingsData/KnockLangM2.csv')
h = pd.read_csv('TrainingsData/KnockLangM3.csv')
k = pd.read_csv('TrainingsData/KnockLangM4.csv')
l = pd.read_csv('TrainingsData/KnockLangM5.csv')

# Alle DataFrames in einer Liste speichern
dataframes = [a, b, c, d, e, f, g, h, i, j, k, l]

# DataFrames zusammenführen
merged_data = pd.concat(dataframes)
merged_data.to_csv('TrainingsData/merged_data.csv', index=False)

# Ergebnis anzeigen
print(merged_data)
