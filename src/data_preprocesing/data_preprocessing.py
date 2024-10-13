import json
import os

import lightning as L
import pandas as pd
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

        unique_labels = df['label'].unique()
        print("Unique labels: ")
        print(unique_labels)
        print("Number of unique labels: ")
        print(len(unique_labels))
        print("Dataframe dtype: ")
        print(df['label'].dtype)
        print("Null values: ")
        print(df['label'].isnull().sum())

        filtered_data = df.groupby('label').apply(self._filter_and_sample).reset_index(drop=True)
        filtered_data.dropna()

        # Display the results
        print("Original dataset size:", df.shape)
        print(df['label'].value_counts())  # Check the distribution
        print("Filtered dataset size:", filtered_data.shape)
        print(filtered_data['label'].value_counts())  # Check the distribution

        train_df, val_test_df = train_test_split(df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=42)
        self.train_dataset = CustomImageDataset(train_df, self.image_size)
        self.val_dataset = CustomImageDataset(val_df, self.image_size)
        self.test_dataset = CustomImageDataset(test_df, self.image_size)

    def _filter_and_sample(self, group):
        max_samples = 20000
        if len(group) > max_samples:
            return group.sample(max_samples, random_state=42)  # Sample max_samples
        else:
            return group

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    def _create_unified_dataframe(self) -> pd.DataFrame:
        base_path = "data/trial_data" if self.trial_data else "data/full_data"
        street_location_dataset_path = f"{base_path}/data"
        geolocation_dataset_path = f"{base_path}/compressed_dataset"
        label_mapping_path = f"{base_path}/country_to_index.json"

        with open(label_mapping_path, "r") as f:
            label_mapping = json.load(f)

        street_location_dataset = self._get_street_location_dataset(street_location_dataset_path, label_mapping)
        geolocation_dataset = self._get_geolocation_dataset(geolocation_dataset_path, label_mapping)
        large_dataset_of_images = self._get_large_dataset_of_images(base_path)

        data = street_location_dataset + geolocation_dataset + large_dataset_of_images

        return pd.DataFrame(data, columns=["filename", "label"])

    def _get_geolocation_dataset(self, geolocation_dataset_path, label_mapping):
        data = []
        for label_str in os.listdir(geolocation_dataset_path):
            if not label_str.startswith('.'):
                for img_name in os.listdir(f"{geolocation_dataset_path}/{label_str}"):
                    label = label_mapping.get(label_str)
                    if label:
                        data.append([f"{geolocation_dataset_path}/{label_str}/{img_name}", int(label)])
        return data

    def _get_street_location_dataset(self, street_location_dataset_path, label_mapping):
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
