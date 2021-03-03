"""Demonstrate the use of different autoencoder models for MNIST."""

import matplotlib.pyplot as plt

import numpy as np

import torch

from .data import MNISTDataModule
from .models import LinearAutoencoder

from pytorch_lightning import Trainer

from torchvision.utils import make_grid


if __name__ == '__main__':
    dm = MNISTDataModule()
    dm.prepare_data()
    dm.setup()

    model = LinearAutoencoder(input_dim=dm.input_dim)

    trainer = Trainer(
        max_epochs=10
    )
    trainer.fit(model, dm)

    with torch.no_grad():
        x, y = next(iter(dm.test_dataloader()))
        x_hat, z, loss = model(x.view(x.size(0), -1))

        batch_size, dim = x_hat.size()
        width, height = int(np.sqrt(dim)), int(np.sqrt(dim))
        x_hat = x_hat.view(batch_size, 1, height, width)

    img = make_grid(x_hat)
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()
