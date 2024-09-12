import json
import os
import pandas as pd
import lightning as L

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split



class CustomImageDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["filename"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = row["label"]
        return image, label


class GeoDataModule(L.LightningDataModule):
    def __init__(self, trial_data: bool=False, batch_size: int = 32):
        super().__init__()
        self.trial_data = trial_data
        self.batch_size = batch_size

    def setup(self, stage: str):
        df = self._create_unified_dataframe(self.trial_data)
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        self.train_dataset = CustomImageDataset(train_df)
        self.val_dataset = CustomImageDataset(val_df)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def _create_unified_dataframe(self, trial_data: bool=False):
        base_path = "data/trial_data" if trial_data else "data/full_data"
        street_location_dataset_path = f"{base_path}/street-location-images/data"
        geolocation_dataset_path = f"{base_path}/compressed_dataset"
        label_mapping_path = f"{base_path}/street-location-images/country_to_index_mapped.json"

        with open(label_mapping_path, "r") as f:
            label_mapping = json.load(f)

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
                data.append([img_path, label])

        for label_str in os.listdir(geolocation_dataset_path):
            for img_name in os.listdir(f"{geolocation_dataset_path}/{label_str}"):
                label = label_mapping.get(label_str)
                if label:
                    data.append([f"{geolocation_dataset_path}/{label_str}/{img_name}", label])

        return pd.DataFrame(data, columns=["filename", "label"])

