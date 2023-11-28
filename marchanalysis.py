import pandas as pd

df1 = pd.read_csv("MNCAATourneyCompactResults.csv", header=0)
print(df1.head())
df2 = pd.read_csv("MNCAATourneySeeds.csv", header=0)
print(df2.head())
df3 = pd.read_csv("MSeasons.csv", header=0)
print(df3.head())
df4 = pd.read_csv("MTeams.csv", header=0)
print(df4.head())

df_combined = pd.concat([df1, df2, df3, df4], ignore_index=True)

df_combined.to_csv('combined_file.csv', index=False)