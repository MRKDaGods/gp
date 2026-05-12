import os
import urllib.request
import zipfile
from pathlib import Path

def download_and_extract():
    url = "https://github.com/cfzd/AIC23_MTMC_sample/releases/download/v1.0/sample_videos.zip"
    target_dir = Path(r"C:\Users\seift\Downloads\gp\data\samples\cityflow")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = target_dir / "sample_videos.zip"
    
    print(f"Downloading sample videos from {url}...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete. Extracting...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
            
        print(f"Successfully extracted tools and videos to {target_dir}")
        os.remove(zip_path)  # clean up zip
    except Exception as e:
        print(f"Error downloading: {e}")
        print("This link might be down or requires authentication.")

if __name__ == "__main__":
    download_and_extract()
