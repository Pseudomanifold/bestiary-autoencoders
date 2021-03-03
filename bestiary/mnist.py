"""Demonstrate the use of different autoencoder models for MNIST."""

from .callbacks import SaveLatentSpaceCallback
from .data import MNISTDataModule
from .models import LinearAutoencoder

from pytorch_lightning import Trainer


if __name__ == '__main__':
    dm = MNISTDataModule()
    dm.prepare_data()
    dm.setup()

    model = LinearAutoencoder(input_dim=dm.input_dim)

    trainer = Trainer(
        max_epochs=50,
        callbacks=SaveLatentSpaceCallback(output_dir='./output')
    )
    trainer.fit(model, dm)
