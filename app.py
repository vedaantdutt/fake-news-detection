
import pandas as pd

data = pd.read_csv("dataset/data.csv")
print(data.head())

data = data.dropna()

data = data.sample(frac=1, random_state=42)
print(data.head())

print(data.head())
print(data.shape)

