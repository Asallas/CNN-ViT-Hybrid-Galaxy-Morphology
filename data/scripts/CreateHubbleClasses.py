import pandas as pd
from pathlib import Path
def create_hubble_classes():
    input_csv = Path("data/raw/gz2_hart16.csv.gz")
    output_csv = Path("data/processed/gz2_three_class_labels.csv")
    df = pd.read_csv(input_csv)


    smooth = df["t01_smooth_or_features_a01_smooth_debiased"]
    disk = df["t01_smooth_or_features_a02_features_or_disk_debiased"]
    spiral = df["t04_spiral_a08_spiral_debiased"]


    df["label"] = "unknown"

    df.loc[(smooth > 0.8) & (disk < 0.2) , "label"] = "elliptical"
    df.loc[(disk > 0.8) & (spiral > 0.8) , "label"] = "spiral"
    df.loc[((df["label"] == "unknown") & ((smooth > 0.5) | (disk > 0.5))) , "label"] = "other"

    df = df[df["label"] != "unknown"]
    df = df[["dr7objid", "ra", "dec", "label"]]

    df.to_csv(output_csv, index=False)

    print(df["label"].value_counts())
    print("Total labeled galaxies:", len(df))

if __name__ == "__main__":
    create_hubble_classes()