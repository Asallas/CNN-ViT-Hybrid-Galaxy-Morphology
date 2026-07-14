import pandas as pd
from pathlib import Path
def analyze_class_distribution():
    df = pd.read_csv(Path("data/processed/gz2_three_class_labels.csv"))

    print("Class distribution:")
    counts = df["label"].value_counts()
    print(counts)

    total = len(df)
    print("\nTotal galaxies:", total)

    print("\nLabel ratios (fraction of total):")
    print((counts / total).round(4))

