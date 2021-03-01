"""Demonstrate the use of different autoencoder models for MNIST."""

from .data import MNISTDataModule
from .models import LinearAutoencoder

from pytorch_lightning import Trainer


if __name__ == '__main__':
    dm = MNISTDataModule()
    dm.prepare_data()
    dm.setup()

    model = LinearAutoencoder(input_dim=dm.input_dim)
    trainer = Trainer()
    trainer.fit(model, dm)
