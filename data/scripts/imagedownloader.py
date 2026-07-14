import pandas as pd
import asyncio
import aiohttp
from pathlib import Path

async def _download_image(session, record, semaphore, output_dir, img_size, max_retries):

    objid = record["dr7objid"]
    ra = record["ra"]
    dec = record["dec"]

    filename = output_dir / f"{objid}.jpg"

    if filename.exists():
        return

    url = (
        "https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg"
        f"?ra={ra}&dec={dec}&scale=0.396&width={img_size}&height={img_size}"
    )
    async with semaphore:
        for attempt in range(max_retries):
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.read()
                        if len(data) > 1000:
                            with open(filename, "wb") as f:
                                f.write(data)
                            return None

            except aiohttp.ClientError:
                pass
            except asyncio.TimeoutError:
                pass
            
            await asyncio.sleep(0.5)
    
    return objid


async def _download_all_images(records, output_dir, img_size, concurrent_requests, max_retries):

    connector = aiohttp.TCPConnector(limit=concurrent_requests)

    timeout = aiohttp.ClientTimeout(total=30)
    semaphore = asyncio.Semaphore(concurrent_requests)
    failed_objids = []

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:

        tasks = [
            _download_image(session=session, 
                                 record=record, 
                                 semaphore=semaphore, 
                                 output_dir=output_dir, 
                                 img_size=img_size, 
                                 max_retries=max_retries) 
            for record in records]
        total = len(tasks)
        for i,task in enumerate(asyncio.as_completed(tasks), start=1):
            result = await task
            if result is not None:
                failed_objids.append(result)
            if i % concurrent_requests == 0:
                print(f"i: {i:,}/{total:,} images processed")


    return failed_objids

def download_images(csv_file=None, output_dir=None, img_size=224, concurrent_requests=40, max_retries=5):
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent
    csv_file = Path(csv_file) if csv_file else data_dir / "raw" / "gz2_hart16.csv.gz"
    output_dir = Path(output_dir) if output_dir else data_dir / "images" / "images_raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_file)

    required_columns = {"dr7objid", "ra", "dec"}

    missing_cols = (required_columns - set(df.columns))

    if missing_cols:
        raise ValueError(f"CSV file is missing required columns: {missing_cols}")
    
    records = df[["dr7objid", "ra", "dec"]].to_dict("records")

    print(f"Starting download of {len(records):,} images to {output_dir}...")

    failed_objids = asyncio.run(_download_all_images(records, output_dir, img_size, concurrent_requests, max_retries))

    print(f"Download complete. Failed downloads: {len(failed_objids):,}")
    return failed_objids

if __name__ == "__main__":
    download_images()