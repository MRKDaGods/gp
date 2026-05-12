import os
import shutil

# Set Kaggle credentials from the provided token
os.environ["KAGGLE_USERNAME"] = "mrkdagods"
os.environ["KAGGLE_KEY"] = "b7b65632a8b882d35a6fbe8b074e0a71"

try:
    import kagglehub
    print("Downloading AI City 2023 Track 2 dataset (thanhnguyenle/data-aicity-2023-track-2)...")
    
    # Download dataset
    path = kagglehub.dataset_download("thanhnguyenle/data-aicity-2023-track-2")
    print(f"Downloaded to: {path}")
    
    # Target folder: c:\Users\seift\Downloads\gp\dataset
    dest = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
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
            
    print("Dataset successfully downloaded and moved to the 'dataset' folder!")
except Exception as e:
    print(f"Error: {e}")
