"""This code imlpements early stopping and was taken from
https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py"""

import numpy as np
from copy import deepcopy

import torch
from torch import nn


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a
    given patience.

    Args:
        patience: how long to wait after last time validation loss improved. Defaults to 10.
        verbose: if True, prints a message for each validation loss improvement. Defaults to False.
        delta: minimum change in the monitored quantity to qualify as an improvement. Defaults to 1e-4.
    """

    def __init__(self, patience: int = 12, verbose: bool = False, delta: float = 1e-4):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.checkpoint = None

    def __call__(self, val_loss: torch.Tensor, model: nn.Module):
        """Perform a check for early stopping

        Args:
            val_loss: validation loss
            model: model that is being trained
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: torch.Tensor, model: nn.Module):
        """Save the model"""
        self.checkpoint = deepcopy(model.state_dict())
        self.val_loss_min = val_loss
