# import wandb
import pytorch_lightning as pl
from torchvision import transforms
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split
from dataset_1d import CovidDataset1D
from cnn_1d import CNN1D
from dataset import test_dataset

# wandb.login(key="a053eb0835a86520fa9a0bba095fabfac12a10c5")


# main
data_path = r"/mnt/chromeos/MyFiles/university/8/Computer Architectures/data"
# test_dataset()
# wandb_logger = WandbLogger()

dataset = CovidDataset1D(data_path)


length = len(dataset)
t, v = int(0.8 * length), int(0.1 * length)
train, val, test = random_split(dataset, [t, v, length - t - v])
train_loader = DataLoader(train, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val, batch_size=64, shuffle=False, num_workers=0)
test_loader = DataLoader(test, batch_size=64, shuffle=False, num_workers=0)
# trainer = pl.Trainer(max_epochs=50, accelerator="gpu", devices=1,
#                      logger=wandb_logger)
trainer = pl.Trainer(max_epochs=50)
model = CNN1D()
trainer.fit(model, train_loader, val_loader)
