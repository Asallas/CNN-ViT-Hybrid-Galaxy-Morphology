import os
import pandas as pd

CSV_FILE = "data/gz2_three_class_labels.csv"
IMAGE_DIR = "data/images3_224"

df = pd.read_csv(CSV_FILE)

missing = []

for objid in df["dr7objid"]:

    img_path = os.path.join(IMAGE_DIR, f"{objid}.jpg")

    if not os.path.exists(img_path):
        missing.append(objid)

print("Total images expected:", len(df))
print("Missing images:", len(missing))

if missing:
    print("\nMissing image IDs:")
    for objid in missing:
        print(objid)