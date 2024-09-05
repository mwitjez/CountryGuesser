from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CustomImageDataset(Dataset):
    def __init__(self, folder_path, label_mapping_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]  # Adjust file extensions as needed
        with open(label_mapping_path, 'r') as f:
            self.label_mapping = json.load(f) 

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.folder_path, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label_name = img_name.rsplit('.', 1)[0] + '.json'  # Replace image extension with .json
        label_path = os.path.join(self.folder_path, label_name)

        with open(label_path, 'r') as f:
            label_data = json.load(f)
            label_str = label_data['country_name']
            label = self.label_mapping[label_str]

        return image, label


def get_dataloader(test_data: bool=True) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    if test_data:
        path = "../data/test_data"
    else:
        path = "data/full_data"
    dataset = CustomImageDataset(folder_path=path, label_mapping_path="../data/country_to_index.json", transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    return dataloader
