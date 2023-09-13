import torch
import wandb
from torch import nn, optim
import pytorch_lightning as pl
import torchvision.models as models


def accuracy(pred, target):
    correct = torch.sum(torch.argmax(pred, 1) == target)
    return correct / len(target)


class CNN(pl.LightningModule):
    def __init__(self, finetune=True):
        super().__init__()
        self.backbone = models.resnet50(weights="DEFAULT")
        num_filters = self.backbone.fc.in_features
        layers = list(self.backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        num_target_classes = 100
        self.classifier = nn.Linear(num_filters, num_target_classes)
        self.finetune = finetune
        self.loss_fn = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.finetune:
            self.feature_extractor.eval()
            with torch.no_grad():
                representations = self.feature_extractor(x).flatten(1)
        else:
            representations = self.feature_extractor(x).flatten(1)
        y_hat = self.classifier(representations)
        train_loss = self.loss_fn(y_hat, y)
        train_acc = accuracy(y_hat, y)
        wandb.log({'train_loss': train_loss})
        wandb.log({'train_accuracy': train_acc})
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.finetune:
            self.feature_extractor.eval()
            with torch.no_grad():
                representations = self.feature_extractor(x).flatten(1)
        else:
            representations = self.feature_extractor(x).flatten(1)
        y_hat = self.classifier(representations)
        val_loss = self.loss_fn(y_hat, y)
        val_acc = accuracy(y_hat, y)
        wandb.log({'val_loss': val_loss})
        wandb.log({'val_accuracy': val_acc})
        return val_loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
