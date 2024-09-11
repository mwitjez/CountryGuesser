import json
import os
import shutil
import torch

from huggingface_hub import Repository
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from src.data_preprocessing import DataLoaders
from src.models.tiny_vit_lightning import TinyVitLightning

def train():
    loaders = DataLoaders(trial_data=True)
    train_dataloader = loaders.get_train_dataloader()
    val_dataloader = loaders.get_val_dataloader()

    with open("data/country_to_index_mapped.json", "r") as f:
        num_classes = len(json.load(f))

    model = TinyVitLightning(num_classes)
    wandb_logger = WandbLogger()
    trainer = Trainer(
        max_epochs=10,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
        logger=wandb_logger
    )
    trainer.fit(model, train_dataloader, val_dataloader)

    model_save_path = "data/models/"
    repo = Repository(local_dir=model_save_path, clone_from="mwitjez/geoguessr_tiny_ViT")

    input_sample = torch.randn(1, 3, 224, 224)
    model.to_onnx(f"{model_save_path}/model.onxx", input_sample, export_params=True)
    torch.save(model.state_dict(), f"{model_save_path}/geoguessr_model.bin")

    repo.push_to_hub()

    if os.path.exists(model_save_path):
        shutil.rmtree(model_save_path)
        print(f"'{model_save_path}' has been removed successfully.")
    else:
        print(f"'{model_save_path}' does not exist.")
