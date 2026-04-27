import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/gz2_three_class_labels.csv")

train_df, temp_df = train_test_split(
    df, test_size=0.3, stratify=df["label"], random_state=42
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
)

train_df.to_csv("train_labels2.csv", index=False)
val_df.to_csv("val_labels2.csv", index=False)
test_df.to_csv("test_labels2.csv", index=False)

print("Train set size:", len(train_df))
print("Validation set size:", len(val_df))
print("Test set size:", len(test_df))

print(test_df["label"].value_counts(normalize=True))