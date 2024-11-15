import abc
from abc import ABC

import torch
import numpy as np

from torch.nn import functional as F
from torch import optim, nn
import lightning as L
import pytorch_lightning as pl
import mltools.layers.torch as layers

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau



class LitTransmissionMatrix(L.LightningModule):

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 train_loss_fn = nn.MSELoss(),
                 ):
        super().__init__()
        self.linear_list = nn.ModuleList([
            layers.Linear(
                in_features=input_size,
                out_features=output_size,
                bias=False,
                dtype=torch.cfloat,
                real2_domain=False,
            )
        ])
        self.abs = layers.Abs()
        self.loss_fn = train_loss_fn

    def forward(self, batch):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = 0
        for layer in self.linear_list:
            y_hat = y_hat + self.abs(layer(x))
        return y_hat, y

    def training_step(self, batch):
        # training_step defines the train loop.
        # it is independent of forward
        y_hat, y = self.forward(batch)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch):
        # training_step defines the train loop.
        # it is independent of forward
        y_hat, y = self.forward(batch)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer




class MLP(L.LightningModule):
    _default_optimizer = torch.optim.Adam
    _default_init_lr: float = 1e-3
    _default_scheduler = optim.lr_scheduler.ReduceLROnPlateau
    _default_scheduler_kwargs: dict = dict(
        factor=0.5,
        patience=10,
        threshold=0.01,
        cooldown=3,
        min_lr=1e-7,
    )

    def __init__(
            self,
            input_size: int,
            hidden_layers_sizes: list[int],
            output_size: int,
            hidden_activation_layers = nn.LeakyReLU(),
            last_activation_layer = None,
            train_loss_fn = nn.MSELoss(),
            add_dropout: bool = False,
            dropout_prob: float = 0.2,
            init_lr: float = None,
            scheduler_kwargs: dict = None
            ):
        super().__init__()
        self.save_hyperparameters()
        self._init_lr = init_lr if init_lr is not None else LitMLP._default_init_lr
        self._scheduler_kwargs = scheduler_kwargs if scheduler_kwargs is not None else LitMLP._default_scheduler_kwargs
        
        # Define the layers of the MLP
        layers = []
        previous_size = input_size
        self.dropout_prob = dropout_prob

        # Add hidden layers
        for hidden_size in hidden_layers_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(hidden_activation_layers)  # Activation function (ReLU)
            if add_dropout:
                layers.append(nn.Dropout(dropout_prob))
            previous_size = hidden_size

        # Add output layer
        layers.append(nn.Linear(previous_size, output_size))
        if last_activation_layer is not None:
            layers.append(last_activation_layer)
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
        
        # Define loss function
        self.loss_fn = train_loss_fn


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # optimizer = LitMLP._default_optimizer(self.parameters(), lr=self._init_lr, weight_decay=1e-5)
        optimizer = MLP._default_optimizer(self.parameters(), lr=self._init_lr)
        lr_scheduler = MLP._default_scheduler(optimizer, **self._scheduler_kwargs)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
            }
        }
        
        
        

