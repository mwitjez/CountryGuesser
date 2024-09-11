import os
from kaggle.api.kaggle_api_extended import KaggleApi


def prepare_data():
    api = KaggleApi()
    api.authenticate()

    dataset = "killusions/street-location-images"
    download_dir = "data/"
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    print(f"Downloading dataset {dataset} to {download_dir}")
    api.dataset_download_files(dataset, path=download_dir, unzip=True)
    print(f"Dataset {dataset} downloaded and unzipped at {download_dir}")
