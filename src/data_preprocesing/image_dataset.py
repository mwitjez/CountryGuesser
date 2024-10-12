from typing import Tuple

import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: Tuple[int, int] = (224, 224)) -> None:
        self.df = df
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        row = self.df.iloc[idx]
        img_path = row["filename"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = row["label"]
        return image, label
