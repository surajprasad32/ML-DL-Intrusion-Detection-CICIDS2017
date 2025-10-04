"""
download_dataset.py
--------------------------------
Utility script to download the CICIDS2017 dataset from Kaggle.
It expects a 'kaggle.json' API token file in the project root
or uploaded in Google Colab environment.
"""

import os
from pathlib import Path

def setup_kaggle_api():
    """Configure Kaggle credentials"""
    if not Path("~/.kaggle/kaggle.json").expanduser().exists():
        print("⚠️ kaggle.json not found. Please upload it when prompted.")
        from google.colab import files
        uploaded = files.upload()
        os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
        os.system("cp kaggle.json ~/.kaggle/")
        os.system("chmod 600 ~/.kaggle/kaggle.json")
    else:
        print("✅ kaggle.json already configured.")

def download_cicids_dataset():
    """Download and unzip CICIDS2017 dataset from Kaggle"""
    os.makedirs("data", exist_ok=True)
    print("⬇️ Downloading dataset from Kaggle...")
    os.system("pip install kaggle -q")
    os.system("kaggle datasets download -d chethuhn/network-intrusion-dataset -p data/")
    print("📦 Unzipping dataset...")
    os.system("unzip -q data/network-intrusion-dataset.zip -d data/")
    print("✅ Dataset ready in ./data")

if __name__ == "__main__":
    setup_kaggle_api()
    download_cicids_dataset()
