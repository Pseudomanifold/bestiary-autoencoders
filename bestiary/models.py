"""Implementation of some autoencoder architectures."""

import torch

import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl


class LinearAutoencoder(pl.LightningModule):
    """A simple linear autoencoder architecture."""

    def __init__(self, input_dim, bottleneck_dim=2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, input_dim)
        )

        self.loss_fn = F.mse_loss

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Unravel all image dimensions into a single one. For MNIST, we
        # will go from (b, 1, 28, 28) to (b, 1*28*28).
        x = x.view(x.size(0), -1)

        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = self.loss_fn(x_hat, x)

        self.log('train_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
