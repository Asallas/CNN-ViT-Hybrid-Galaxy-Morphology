import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def check_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return None
    except Exception:
        return Path(image_path)

def validate_images(image_dir=None, progress_interval=5000, max_workers=None):
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent
    IMAGE_DIR = Path(image_dir) if image_dir else data_dir / "images" / "images_raw"
    
    max_workers = max_workers or os.cpu_count() or 8

    files = [path for path in IMAGE_DIR.iterdir() if path.is_file()]

    total = len(files)
    print(f"Checking {total} images with {max_workers} workers...")

    bad_images = []
    processed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        futures = [executor.submit(check_image, f) for f in files]

        for future in as_completed(futures):

            result = future.result()
            processed += 1

            if result is not None:
                bad_images.append(result)

            if (progress_interval > 0 and processed % progress_interval == 0):
                percent = (processed / total) * 100
                print(f"{processed}/{total} ({percent:.2f}%) checked")

    print("\nFinished.")
    print(f"Corrupted images: {len(bad_images):,}")

    if bad_images:
        print("\nList of corrupted files:")
        for img in bad_images:
            print(img)
    return bad_images

if __name__ == "__main__":
    validate_images()
