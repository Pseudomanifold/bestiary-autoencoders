"""Callback modules for executing tasks during training."""

from pytorch_lightning.callbacks.base import Callback

from torchvision.utils import save_image

import numpy as np

import torch
import os


class SaveLatentSpaceCallback(Callback):

    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

    def save(self, x_hat, filename):
        batch_size, dim = x_hat.size()
        width, height = int(np.sqrt(dim)), int(np.sqrt(dim))
        x_hat = x_hat.view(batch_size, 1, height, width)

        save_image(x_hat, filename)


    def on_epoch_end(self, trainer, pl_module):
        with torch.no_grad():
            x, y = next(iter(pl_module.val_dataloader()))
            x_hat, z, loss = pl_module(x.view(x.size(0), -1))

            epoch = trainer.current_epoch

            self.save(
                x_hat,
                os.path.join(self.output_dir, f'R_{epoch:04d}.png')
            )

            np.savetxt(
                os.path.join(self.output_dir, f'L_{epoch:04d}.txt'),
                z.numpy(),
                fmt='%.8f'
            )
