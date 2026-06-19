import pandas as pd

df = pd.read_csv("alzheimers_disease_data.csv")
print("Original Shape:", df.shape)

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())


df = df.drop_duplicates()
df = df.dropna()

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nCleaned Shape:", df.shape)

df.to_csv("alzheimers_cleaned.csv", index=False)

print("\nDataset cleaned and saved as 'alzheimers_cleaned.csv'")