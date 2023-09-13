import torch
from torch import nn, optim
import pytorch_lightning as pl
import torchvision.models as models

def accuracy(pred, target):
    correct = torch.sum(torch.argmax(pred, 1) == target)
    return correct / len(target)


class CNN1D(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.ModuleList()
        for i in range(6):
            in_channels = 1 if i == 0 else out_channels
            out_channels = (2 * (i + 1))  # Double the filters in each layer
            # print(in_channels, out_channels)
            self.feature_extractor.append(
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=7, stride=3)
            )
            self.feature_extractor.append(nn.ReLU())
        self.feature_extractor = nn.Sequential(*self.feature_extractor)

        num_target_classes = 6
        self.classifier = nn.Sequential(
            nn.Linear(468, 100),
            nn.ReLU(),
            nn.Linear(100, num_target_classes),
            nn.Sigmoid(),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.unsqueeze(1)
        representations = self.feature_extractor(x)
        y_hat = self.classifier(representations.flatten(1))
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        train_loss = self.loss_fn(y_hat, y.type(torch.long))
        train_acc = accuracy(y_hat, y)
        self.log(name='train_loss', value=train_loss.item())
        self.log(name='train_accuracy', value=train_acc.item())
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        val_loss = self.loss_fn(y_hat, y.type(torch.long))
        val_acc = accuracy(y_hat, y)
        self.log(name='val_loss', value=val_loss.item())
        self.log(name='val_accuracy', value=val_acc.item())
        return val_loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-4)