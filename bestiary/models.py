"""Implementation of some autoencoder architectures."""

import torch

import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl


class LinearAutoencoder(pl.LightningModule):
    """A simple linear autoencoder architecture."""

    def __init__(self, input_dim, bottleneck_dim=2, lr=0.01):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, input_dim)
        )

        self.lr = 0.01
        self.loss_fn = F.mse_loss

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)

        loss = self.loss_fn(x_hat, x)
        return x_hat, z, loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Unravel all image dimensions into a single one. For MNIST, we
        # will go from (b, 1, 28, 28) to (b, 1*28*28).
        x = x.view(x.size(0), -1)

        # We don't care about anything but the loss here. The other
        # return values of the model will be more relevant for some
        # post-processing tasks.
        _, _, loss = self(x)

        self.log('train_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class BetaVAE(pl.LightningModule):
    """A Beta VAE architecture."""

    def __init__(self, input_dim, bottleneck_dim=32, beta=1, lr=1e-3):
        super(BetaVAE, self).__init__()

        self.bottleneck_dim = bottleneck_dim
        self.beta = beta
        self.lr = lr

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Linear(32, 16),
        )

        # Final layer for encoding the parameters of the distribution
        self.fc_mu = nn.Linear(16, bottleneck_dim)
        self.fc_logvar = nn.Linear(16, bottleneck_dim)

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.Linear(32, input_dim),
            nn.Sigmoid(),
        )

        # For sampling values that parametrise our distribution
        self.fc_z = nn.Linear(bottleneck_dim, 16)

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # e^(1/2 * log(std^2))
        eps = torch.randn_like(std)    # random ~ N(0, 1)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = self.fc_z(z)
        z = z.view(z.shape[0], -1)
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, self.loss_fn(x_hat, x, mu, logvar) 

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        _, _, loss = self(x)

        self.log('train_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def loss_fn(self, x_hat, x, mu, logvar):

        reconstruction_loss = F.mse_loss(
            x_hat,
            x,
            reduction='sum' # We reduce it ourselves later on!
        )

        # This follows the loss provided by Kingma and Welling in
        # 'Auto-Encoding Variational Bayes', Appendix B. The loss
        # assumes that all distributions are Gaussians.
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return (reconstruction_loss + self.beta * kl) / x.shape[0]
