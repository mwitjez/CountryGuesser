import torch
import torch.nn as nn
import lightning as L
from src.models.tiny_vit import tiny_vit_21m_224
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchmetrics.classification import MulticlassF1Score, Accuracy

class TinyVitLightning(L.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.pretrained_model = tiny_vit_21m_224(pretrained=True, num_classes=num_classes)

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.f1_score = MulticlassF1Score(num_classes=num_classes)

        self.loss = nn.CrossEntropyLoss()
        self.lr = 1e-3

        self.save_hyperparameters()

    def forward(self, x):
        return self.pretrained_model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)

        acc = self.accuracy(preds, y)
        f1 = self.f1_score(preds, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", acc, on_step=False, on_epoch=True)
        self.log("train_f1", f1, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)

        acc = self.accuracy(preds, y)
        f1 = self.f1_score(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        self.log("val_f1", f1, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]
