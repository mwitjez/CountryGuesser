import json
import os

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class CustomImageDataset(Dataset):
    def __init__(self, folder_path, label_mapping_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(folder_path) if f.endswith((".jpg", ".png", ".jpeg"))]
        with open(label_mapping_path, "r") as f:
            self.label_mapping = json.load(f)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.folder_path, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label_name = (img_name.rsplit(".", 1)[0] + ".json")
        label_path = os.path.join(self.folder_path, label_name)

        with open(label_path, "r") as f:
            label_data = json.load(f)
            label_str = label_data["country_name"]
            label = self.label_mapping[label_str]

        return image, label


class DataLoaders:
    def __init__(self, trial_data: bool = True):
        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self._label_mapping_path = "../data/country_to_index.json"
        self._base_path = "../data/trial_data" if trial_data else "../data/full_data"

        self._train_dataloader = self._create_dataloader("train")
        self._test_dataloader = self._create_dataloader("test")

    def _create_dataloader(self, data_type: str) -> DataLoader:
        path = f"{self._base_path}/{data_type}"
        dataset = CustomImageDataset(
            folder_path=path,
            label_mapping_path=self._label_mapping_path,
            transform=self._transform,
        )
        return DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    def get_train_dataloader(self) -> DataLoader:
        return self._train_dataloader

    def get_test_dataloader(self) -> DataLoader:
        return self._test_dataloader