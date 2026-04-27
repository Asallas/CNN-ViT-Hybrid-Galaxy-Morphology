import pandas as pd
import asyncio
import aiohttp
import os

CSV_FILE = "gz2_hart16.csv.gz"
OUTPUT_DIR = "images_raw"
IMG_SIZE = 424
CONCURRENT_REQUESTS = 40
MAX_RETRIES = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_FILE)
records = df[["dr7objid", "ra", "dec"]].to_dict("records")


async def download_image(session, record):

    objid = record["dr7objid"]
    ra = record["ra"]
    dec = record["dec"]

    filename = f"{OUTPUT_DIR}/{objid}.jpg"

    if os.path.exists(filename):
        return

    url = (
        "https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg"
        f"?ra={ra}&dec={dec}&scale=0.396&width={IMG_SIZE}&height={IMG_SIZE}"
    )

    for attempt in range(MAX_RETRIES):
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.read()
                    if len(data) > 1000:
                        with open(filename, "wb") as f:
                            f.write(data)
                        return

        except Exception:
            pass
        
        await asyncio.sleep(0.5)


async def main():

    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)

    async with aiohttp.ClientSession(connector=connector) as session:

        tasks = [download_image(session, record) for record in records]

        for i in range(0, len(tasks), CONCURRENT_REQUESTS):
            batch = tasks[i:i+CONCURRENT_REQUESTS]
            await asyncio.gather(*batch)
            print(f"{i+len(batch)} images processed")


asyncio.run(main())