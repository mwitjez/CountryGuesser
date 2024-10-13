import json
import os
import csv

import lightning as L
import pandas as pd
import reverse_geocoder as rg
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data_preprocesing.large_dataset_preprocessing import LargeDatasetPreprocessor
from src.data_preprocesing.image_dataset import CustomImageDataset


class GeoDataModule(L.LightningDataModule):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.trial_data = config["trial_data"]
        self.batch_size = config["batch_size"]
        self.image_size = (config["image_size"], config["image_size"])

    def setup(self, stage: str) -> None:
        df = self._create_unified_dataframe()
        df.dropna()

        print(df["label"].value_counts())

        train_df, val_test_df = train_test_split(df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=42)
        self.train_dataset = CustomImageDataset(train_df, self.image_size)
        self.val_dataset = CustomImageDataset(val_df, self.image_size)
        self.test_dataset = CustomImageDataset(test_df, self.image_size)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    def _create_unified_dataframe(self) -> pd.DataFrame:
        base_path = "data/trial_data" if self.trial_data else "data/full_data"
        label_mapping_path = f"{base_path}/country_to_index.json"

        with open(label_mapping_path, "r") as f:
            label_mapping = json.load(f)

        street_location_dataset = self._get_street_location_dataset(base_path, label_mapping)
        geolocation_dataset = self._get_geolocation_dataset(base_path, label_mapping)
        street_view_panoramas = self._get_street_view_panoramas(base_path)

        data = street_location_dataset + geolocation_dataset + street_view_panoramas

        return pd.DataFrame(data, columns=["filename", "label"])

    def _get_geolocation_dataset(self, base_path, label_mapping):
        geolocation_dataset_path = f"{base_path}/compressed_dataset"
        data = []
        for label_str in os.listdir(geolocation_dataset_path):
            if not label_str.startswith('.'):
                for img_name in os.listdir(f"{geolocation_dataset_path}/{label_str}"):
                    label = label_mapping.get(label_str)
                    if label:
                        data.append([f"{geolocation_dataset_path}/{label_str}/{img_name}", int(label)])
        return data

    def _get_street_location_dataset(self, base_path, label_mapping):
        street_location_dataset_path = f"{base_path}/data"
        data = []
        for img_name in os.listdir(street_location_dataset_path):
            if img_name.endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(street_location_dataset_path, img_name)
                label_name = (img_name.rsplit(".", 1)[0] + ".json")
                label_path = os.path.join(street_location_dataset_path, label_name)
                with open(label_path, "r") as f:
                    label_data = json.load(f)
                    label_str = label_data["country_name"]
                    label = label_mapping[label_str]
                data.append([img_path, int(label)])
        return data

    def _get_large_dataset_of_images(self, base_path):
        preprocessor = LargeDatasetPreprocessor()
        data = preprocessor.get_data(base_path)
        return data

    def _get_street_view_panoramas(self, base_path):
        with open(f"{base_path}/country_code_to_index.json", "r") as f:
            country_code_to_label = json.load(f)

        with open(f"{base_path}/images.csv", 'r') as f:
            data_dict = {row['id']: (float(row['lat']), float(row['lng'])) for row in csv.DictReader(f)}

        country_codes = rg.RGeocoder(mode=2).query(list(data_dict.values()))
        img_dir = f"{base_path}/images"
        data = []

        for i, id_ in enumerate(data_dict):
            img_path = os.path.join(img_dir, f"{id_}.jpeg")
            if os.path.exists(img_path):
                country_code = country_codes[i]['cc']
                label = country_code_to_label.get(country_code, -1)
                if label != -1:
                    data.append((img_path, label))

        return data
