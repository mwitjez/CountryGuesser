import json

import lightning as L
import torch
from data_preprocessing import GeoDataModule
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from models.tiny_vit_lightning import TinyVitLightning


def train() -> L.LightningModule:
    with open("src/config/model_config.json", "r") as f:
        config = json.load(f)

    torch.set_float32_matmul_precision("medium")
    data_module = GeoDataModule(config)
    model = TinyVitLightning(config)
    wandb_logger = WandbLogger(project="geoguessr AI")
    trainer = Trainer(
        max_epochs=config["num_epochs"],
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min"),
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=wandb_logger,
    )
    trainer.fit(model, data_module)
    return model
