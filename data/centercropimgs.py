import os
import pandas as pd
import cv2
from tqdm import tqdm
import concurrent.futures
import threading

RAW_DIR = "images_raw"
OUT_DIR = "images3_224"
CSV_FILE = "gz2_three_class_labels.csv"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_FILE)

lock = threading.Lock()

def process_image(objid):
    with lock:
        pbar.update(1)
    
    img_path = f"{RAW_DIR}/{objid}.jpg"
    out_path = f"{OUT_DIR}/{objid}.jpg"
    
    img = cv2.imread(img_path)
    
    if img is None:
        return objid  # failed to read
    
    h, w = img.shape[:2]
    
    startx = w//2 - 112
    starty = h//2 - 112
    
    crop = img[starty:starty+224, startx:startx+224]
    
    crop = crop.astype("float32") / 255.0
    
    cv2.imwrite(out_path, (crop*255).astype("uint8"))
    
    return None

with tqdm(total=len(df), desc="Processing images") as pbar:
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_image, objid) for objid in df["dr7objid"]]
        failed_objids = [f.result() for f in concurrent.futures.as_completed(futures) if f.result() is not None]

# Remove failed entries from dataframe
if failed_objids:
    df = df[~df["dr7objid"].isin(failed_objids)]
    df.to_csv(CSV_FILE, index=False)
    print(f"Removed {len(failed_objids)} entries from CSV due to missing images.")