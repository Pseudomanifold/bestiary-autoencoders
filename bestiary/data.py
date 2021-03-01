"""Data sets for use with `pytorch_lightning`."""

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from pytorch_lightning import LightningDataModule

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import os


class MNISTDataModule(LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()

        self.batch_size = batch_size
        self.root = os.path.join(os.getcwd(), 'data')

    def prepare_data(self):
        MNIST(self.root, train=True, download=True, transform=ToTensor())
        MNIST(self.root, train=False, download=True, transform=ToTensor())

    def setup(self):
        train_data = MNIST(self.root, train=True, transform=ToTensor())
        test_data = MNIST(self.root, train=False, transform=ToTensor())

        # MNIST is sufficiently large so that we can pull this off!
        train_data, val_data = random_split(train_data, [55000, 5000])

        self.train_dataset = train_data
        self.val_dataset = val_data
        self.test_dataset = test_data

        # We could also hard-code this, but let's get dimensions and
        # other information from the data.

        batch, labels = next(iter(self.train_dataloader()))
        self.input_dim = batch.view(self.batch_size, -1).size()[1]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
