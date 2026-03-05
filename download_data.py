"""
Download and unzip large data assets from GitHub Release at container startup.
Run this before starting the app.
"""
import os
import urllib.request
import zipfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RELEASE_URL = "https://github.com/sedak326/Proteome-Visualizer-with-RAG-agent/releases/download/v1.0-data"

ASSETS = [
    ("umap_output.zip", "umap_output"),
    ("annotated.zip", "annotated"),
    ("chroma_db.zip", "Marine_RAG"),
    ("animal_pics.zip", "animal_pics"),
    ("prerendered_animations.zip", "prerendered_animations"),
]

def already_downloaded(marker_file):
    return os.path.exists(os.path.join(BASE_DIR, marker_file))

def download_and_unzip(zip_name, check_dir):
    check_path = os.path.join(BASE_DIR, check_dir)
    if os.path.exists(check_path) and os.listdir(check_path):
        print(f"  {check_dir} already present, skipping.")
        return

    url = f"{RELEASE_URL}/{zip_name}"
    dest = os.path.join(BASE_DIR, zip_name)
    print(f"  Downloading {zip_name}...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Extracting {zip_name}...")
    with zipfile.ZipFile(dest, "r") as zf:
        zf.extractall(BASE_DIR)
    os.remove(dest)
    print(f"  Done: {zip_name}")

if __name__ == "__main__":
    print("Downloading data assets...")
    for zip_name, check_dir in ASSETS:
        download_and_unzip(zip_name, check_dir)
    print("All assets ready.")
