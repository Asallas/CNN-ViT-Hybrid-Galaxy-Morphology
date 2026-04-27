import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

IMAGE_DIR = "images"
PROGRESS_INTERVAL = 5000
MAX_WORKERS = os.cpu_count()

def check_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return None
    except Exception:
        return path

if __name__ == '__main__':
    files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)]
    total = len(files)

    print(f"Checking {total} images using {MAX_WORKERS} workers...\n")

    bad_images = []
    processed = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = [executor.submit(check_image, f) for f in files]

        for future in as_completed(futures):

            result = future.result()
            processed += 1

            if result is not None:
                bad_images.append(result)

            if processed % PROGRESS_INTERVAL == 0:
                percent = (processed / total) * 100
                print(f"{processed}/{total} ({percent:.2f}%) checked")

    print("\nFinished.")
    print("Corrupted images:", len(bad_images))

    if bad_images:
        print("\nList of corrupted files:")
        for img in bad_images:
            print(img)
        print(img)