import os
import pandas as pd
import cv2
from tqdm import tqdm
import concurrent.futures
import threading
from pathlib import Path

def center_crop_image(csv_file=None, raw_dir=None, out_dir=None, size=224, max_workers=8):
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent

    CSV_FILE = Path(csv_file) if csv_file else data_dir / "processed" / "gz2_three_class_labels.csv"
    RAW_DIR = Path(raw_dir) if raw_dir else data_dir / "images" / "images_raw"
    OUT_DIR = Path(out_dir) if out_dir else data_dir / "images" / "images3_224"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_FILE)
    failed_objids = []

    lock = threading.Lock()

    half_crop = size // 2





    def process_image(objid):

        img_path = f"{RAW_DIR}/{objid}.jpg"
        out_path = f"{OUT_DIR}/{objid}.jpg"
    
        img = cv2.imread(img_path)
    
        if img is None:
            with lock:
                pbar.update(1)
            return objid  # failed to read
    
        h, w = img.shape[:2]
    
        startx = w//2 - half_crop
        starty = h//2 - half_crop
    
        crop = img[starty:starty+size, startx:startx+size]
    
        cv2.imwrite(str(out_path), crop)
    
        with lock:
            pbar.update(1)

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

if __name__ == "__main__":
    center_crop_image()