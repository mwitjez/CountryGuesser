import json
import os

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


class CustomImageDataset(Dataset):
    def __init__(self, folder_path, image_filenames, label_mapping_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_filenames = image_filenames
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
    def __init__(self, trial_data: bool = True, test_size: float = 0.2):
        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self._label_mapping_path = "data/country_to_index_mapped.json"
        self._base_path = "data/trial_data" if trial_data else "data/full_data"

        all_image_filenames = [f for f in os.listdir(f"{self._base_path}/") if f.endswith((".jpg", ".png", ".jpeg"))]
        train_filenames, val_filenames = train_test_split(all_image_filenames, test_size=test_size)
        self._train_dataloader = self._create_dataloader(train_filenames)
        self._val_dataloader = self._create_dataloader(val_filenames)

    def _create_dataloader(self, image_filenames: list) -> DataLoader:
        dataset = CustomImageDataset(
            folder_path=self._base_path,
            image_filenames=image_filenames,
            label_mapping_path=self._label_mapping_path,
            transform=self._transform,
        )
        return DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    def get_train_dataloader(self) -> DataLoader:
        return self._train_dataloader

    def get_val_dataloader(self) -> DataLoader:
        return self._val_dataloader
