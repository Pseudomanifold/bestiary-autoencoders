# A Bestiary of Autoencoders

This is repository illustrating some simple autoencoder architectures.
To run the example, follow these steps:

1. Install [poetry](https://python-poetry.org), a framework for
   simplifying the management of virtual environments.
2. Issue `poetry install` to set up and install all dependencies.
3. To run the model with default parameters, issue `poetry run python -m bestiary.mnist`.

This should result in the following output:

```
  | Name    | Type   | Params
-----------------------------------
0 | encoder | Linear | 1.6 K
1 | decoder | Linear | 2.4 K
-----------------------------------
3.9 K     Trainable params
0         Non-trainable params
3.9 K     Total params
0.016     Total estimated model params size (MB)
Epoch 1:  62%|█████████████████████████████████████████████████████████████████▊                                        | 534/860 [00:06<00:04, 80.91it/s, loss=0.0557, v_num=127]
```

Afterwards, see the folder `output` for some latent space reconstructions. To see available options, run `poetry run python -m bestiary.mnist --help`.
The current options of the code are as follows:

```
usage: mnist.py [-h] [-m {linear,vae}] [--max-epochs MAX_EPOCHS]

optional arguments:
  -h, --help            show this help message and exit
  -m {linear,vae}, --model {linear,vae}
                        Model to use for training
  --max-epochs MAX_EPOCHS
                        Maximum number of epochs to train
```

If you are interested in autoencoders, check out these awesome other
projects:

- [latent-lego](https://github.com/quadbiolab/latent-lego): a Python library for building autoencoders for single-cell genomics, developed by the Quantitative Developmental Biology Lab of D-BSSE. 
- [topological-autoencoders](https://github.com/BorgwardtLab/topological-autoencoders): a new type of architecture for preserving topological properties in your data, developed by the Machine Learning and Computational Bioloy Lab of D-BSSE.
