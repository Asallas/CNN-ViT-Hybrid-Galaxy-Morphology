import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class GalaxyDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        self.label_map = {
            "elliptical": 0,
            "spiral": 1,
            "other": 2
        }
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        objid = row["dr7objid"]
        label_str = row["label"]

        img_path = os.path.join(self.img_dir, f"{objid}.jpg")

        image = Image.open(img_path).convert("RGB")

        label = self.label_map.get(label_str, -1)

        if self.transform:
            image = self.transform(image)

        return image, label