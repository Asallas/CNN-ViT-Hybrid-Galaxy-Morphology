import os
import pandas as pd
import random
import matplotlib.pyplot as plt
from PIL import Image

CSV_FILE = "data/gz2_hubble_labels.csv"
IMAGE_DIR = "data/images_224"

df = pd.read_csv(CSV_FILE)

sample = df.sample(16)

fig, axes = plt.subplots(4,4, figsize=(8,8))

for ax, (_, row) in zip(axes.flatten(), sample.iterrows()):

    objid = row["dr7objid"]
    label = row["label"]

    img_path = os.path.join(IMAGE_DIR, f"{objid}.jpg")

    img = Image.open(img_path)

    ax.imshow(img)
    ax.set_title(label)
    ax.axis("off")

plt.tight_layout()
plt.show()