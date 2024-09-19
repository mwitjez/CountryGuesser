from typing import List, Tuple

import lightning as L
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchmetrics.classification import Accuracy, MulticlassF1Score
from timm.models import create_model


class FastVitLightning(L.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.pretrained_model = create_model("fastvit_ma36", pretrained=True, num_classes=config["num_classes"])

        self.accuracy = Accuracy(task="multiclass", num_classes=config["num_classes"])
        self.f1_score = MulticlassF1Score(num_classes=config["num_classes"])

        self.loss = nn.CrossEntropyLoss()
        self.lr = 1e-3

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pretrained_model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)

        acc = self.accuracy(preds, y)
        f1 = self.f1_score(preds, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", acc, on_step=False, on_epoch=True)
        self.log("train_f1", f1, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)

        acc = self.accuracy(preds, y)
        f1 = self.f1_score(preds, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        self.log("val_f1", f1, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]:
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=1e-8)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.2)
        return [optimizer], [scheduler] 
