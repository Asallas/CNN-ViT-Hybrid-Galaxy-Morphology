import pandas as pd

df = pd.read_csv("data/gz2_hubble_labels.csv")

print("Class distribution:")
counts = df["label"].value_counts()
print(counts)

total = len(df)
print("\nTotal galaxies:", total)

print("\nLabel ratios (fraction of total):")
print((counts / total).round(4))

