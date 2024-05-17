import torch
from torch import nn
from pytorch_lightning import LightningModule
from torchmetrics.classification import BinaryAccuracy
import mlflow

class LitTraining(LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super(LitTraining, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = self.dice_loss
        self.accuracy = BinaryAccuracy()

    def dice_loss(self, pred, target):
        smooth = 1.
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2,3))
        loss = 1 - ((2. * intersection + smooth) / (pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + smooth))
        return loss.mean()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy((y_hat > 0.5).float(), y)
        self.log('train_loss', loss)
        self.log('train_acc', acc, prog_bar=True)
        mlflow.log_metric('train_loss', loss.item())
        mlflow.log_metric('train_acc', acc.item())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy((y_hat > 0.5).float(), y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        mlflow.log_metric('val_loss', loss.item())
        mlflow.log_metric('val_acc', acc.item())

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy((y_hat > 0.5).float(), y)
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        mlflow.log_metric('test_loss', loss.item())
        mlflow.log_metric('test_acc', acc.item())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
