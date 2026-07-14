import os
import pandas as pd
import requests

def check_images():
    df = pd.read_csv("gz2_hart16.csv.gz", dtype={"dr7objid": str})

    missing = []

    existing = {f.split(".")[0] for f in os.listdir("images")}

    for objid, ra, dec in df[["dr7objid", "ra", "dec"]].values:
        if str(objid) not in existing:
            missing.append((objid, ra, dec))

    print("Number of missing images:", len(missing))
    print("List of missing images(objid, ra, dec):")
    for objid, ra, dec in missing:
        print(objid, ra, dec)

    IMG_SIZE = 424
    for objid, ra, dec in missing:
        url = (
            "https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg"
            f"?ra={ra}&dec={dec}&scale=0.396&width={IMG_SIZE}&height={IMG_SIZE}"
        )

        r = requests.get(url, timeout=10)

    print(objid, r.status_code, r.headers.get("content-type"), len (r.content))

if __name__ == "__main__":
    check_images()
