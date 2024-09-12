from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from data_preprocessing import GeoDataModule
from models.tiny_vit_lightning import TinyVitLightning


def train():
    data_module = GeoDataModule(trial_data=True)
    model = TinyVitLightning()
    wandb_logger = WandbLogger()
    trainer = Trainer(
        max_epochs=10,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
        logger=wandb_logger
    )
    trainer.fit(model, data_module)
    return model
