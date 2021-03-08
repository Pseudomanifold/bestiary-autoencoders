"""Demonstrate the use of different autoencoder models for MNIST."""

import argparse
import torch

from .callbacks import SaveLatentSpaceCallback
from .data import MNISTDataModule

from .models import LinearAutoencoder
from .models import BetaVAE

from pytorch_lightning import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-m', '--model',
        type=str,
        choices=['linear', 'vae'],
        help='Model to use for training',
        default='linear'
    )

    parser.add_argument(
        '--max-epochs',
        type=int,
        # This default value is rather low; it is sufficient to obtain
        # some 'okayish' results for MNIST, but you want adjust this.
        default=50,
        help='Maximum number of epochs to train'
    )

    args = parser.parse_args()

    dm = MNISTDataModule()
    dm.prepare_data()
    dm.setup()

    if args.model == 'linear':
        model = LinearAutoencoder(
            input_dim=dm.input_dim,
            bottleneck_dim=2,
            lr=1e-2  # should be made configurable in your own code :-)

        )
    elif args.model == 'vae':
        model = BetaVAE(
            input_dim=dm.input_dim,
            bottleneck_dim=2,
            lr=1e-3  # should be made configurable in your own code :-)
        )

    # We will use this to save our sample later on (when using a VAE),
    # and of course to save all reconstructions during training.
    save_cb = SaveLatentSpaceCallback(output_dir=f'./output/{args.model}')

    trainer = Trainer(
        max_epochs=args.max_epochs,
        callbacks=SaveLatentSpaceCallback(output_dir=f'./output/{args.model}')
    )
    trainer.fit(model, dm)

    # We can only sample images if we are training a VAE. This is
    # achieved by first sampling a parameterisation of the latent
    # space, then decode it!
    if args.model == 'vae':
        with torch.no_grad():
            x, y = next(iter(dm.val_dataloader()))
            batch_size = x.size(0)
            mu, logvar = model.encode(x.view(batch_size, -1))

            p = torch.distributions.Normal(
                torch.zeros_like(mu), torch.ones_like(logvar)
            )
            z = p.rsample()
            x_hat = model.decode(z)

            save_cb.save(x_hat, f'./output/{args.model}/Sample.png')
