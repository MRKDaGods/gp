import os
import shutil

# Set token
os.environ["KAGGLE_API_TOKEN"] = "KGAT_b7b65632a8b882d35a6fbe8b074e0a71"

try:
    import kagglehub
    print("Downloading MTMC weights dataset...")
    # Download dataset
    path = kagglehub.dataset_download("mrkdagods/mtmc-weights")
    print(f"Downloaded to: {path}")
    
    # Destination folder for models
    dest = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(dest, exist_ok=True)
    
    # Copy files
    print(f"Copying files from {path} to {dest}...")
    for item in os.listdir(path):
        s = os.path.join(path, item)
        d = os.path.join(dest, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
            
    print("Data download and copy complete!")
except Exception as e:
    print(f"Error: {e}")
