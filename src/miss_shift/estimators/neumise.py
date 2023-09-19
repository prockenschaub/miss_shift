# Based on https://github.com/marineLM/NeuMiss_sota

import torch
from torch import Tensor, nn
from torch.nn import Sequential, Linear, ReLU
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
from sklearn.base import BaseEstimator

from ..misc.pytorchtools import EarlyStopping
from ..networks.neumise import NeuMISEBlock


class NeuMISEMLP(nn.Module):
    """A NeuMISE block followed by an MLP.
    
    Args:
        n_features : dimension of inputs.
        neumise_depth : number of layers in the NeuMISE block.
        mlp_depth : number of hidden layers in the MLP.
        mlp_width : width of the MLP. If None take mlp_width=n_features. Default: None.
        dtype : Pytorch dtype for the parameters. Default: torch.float.    
    """

    def __init__(self, n_features: int, neumise_depth: int, mlp_depth: int,
                 mlp_width: int = None, dtype = torch.float):
        super().__init__()
        self.n_features = n_features
        self.neumise_depth = neumise_depth
        self.mlp_depth = mlp_depth
        self.dtype = dtype
        mlp_width = n_features if mlp_width is None else mlp_width
        self.mlp_width = mlp_width

        b = int(mlp_depth >= 1)
        last_layer_width = mlp_width if b else n_features
        self.layers = Sequential(
            NeuMISEBlock(n_features, neumise_depth, dtype),
            *[Linear(n_features, mlp_width, dtype=dtype), ReLU()]*b,
            *[Linear(mlp_width, mlp_width, dtype=dtype), ReLU()]*b*(mlp_depth-1),
            *[Linear(last_layer_width, 1, dtype=dtype)],
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.layers(x)
        return out.squeeze()

class NeuMISE(BaseEstimator):
    """Predict with a NeuMISE block followed by an MLP.

    Args:
        depth: the depth of the NeuMISE block.
        n_epochs: the maximum number of epochs.
        batch_size: the batch size.
        lr: the learning rate.
        weight_decay: the weight decay parameter.
        early_stopping: if True, early stopping is used based on the validaton set.
        optimizer: one of `sgd`or `adam`.
        mlp_depth: the depth of the MLP stacked on top of the NeuMISE iterations.
        width_factor: the width of the MLP stacked on top of the NeuMISE block is calculated
            as width_factor times n_features.
        add_mask: if True, the mask is concatenated to the output of the NeuMISE block.
        verbose: flag to print detailed information about training to the console. 
    """

    def __init__(self, depth: int, n_epochs: int = 1000, batch_size: int = 100, 
                 lr: float = 0.01, weight_decay: float = 1e-4, early_stopping: bool = False, 
                 optimizer: str = 'sgd', mlp_depth: int = 0, width_factor: int = 1,
                 add_mask: bool = False, verbose: bool = False):
        self.depth = depth
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.early_stop = early_stopping
        self.optimizer = optimizer
        self.mlp_depth = mlp_depth
        self.width_factor = width_factor
        self.add_mask = add_mask
        self.verbose = verbose

        self.r2_train = []
        self.mse_train = []
        self.r2_val = []
        self.mse_val = []

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Jointly train the NeuMISE block and the MLP

        Args:
            X: original (n, d) covariates w/ missingness
            y: original (n, ) outcomes 
            X_val: optional covariates w/ missingness that are passively imputed. Defaults to None.
            y_val: optional outcomes that may be used for passively imputed. Defaults to None.
        """
        n_samples, n_features = X.shape

        X = torch.as_tensor(X, dtype=torch.double)
        y = torch.as_tensor(y, dtype=torch.double)

        if X_val is not None:
            X_val = torch.as_tensor(X_val, dtype=torch.double)
            y_val = torch.as_tensor(y_val, dtype=torch.double)

        self.net = NeuMISEMLP(n_features=n_features,
                           neumise_depth=self.depth,
                           mlp_depth=self.mlp_depth,
                           mlp_width=self.width_factor, 
                           dtype=torch.double)

        if len(list(self.net.parameters())) > 0:
            # Create parameter groups
            group_wd = []
            group_no_wd = []
            for name, param in self.net.named_parameters():
                if name in ['mu', 'b']:
                    group_no_wd.append(param)
                else:
                    group_wd.append(param)

            if self.optimizer == 'sgd':
                self.optimizer = optim.SGD(
                    [{'params': group_wd, 'weight_decay': self.weight_decay},
                     {'params': group_no_wd, 'weight_decay': 0}],
                    lr=self.lr)
            elif self.optimizer == 'adam':
                self.optimizer = optim.Adam(
                    [{'params': group_wd, 'weight_decay': self.weight_decay},
                     {'params': group_no_wd, 'weight_decay': 0}],
                    lr=self.lr)

            self.scheduler = ReduceLROnPlateau(
                            self.optimizer, mode='min', factor=0.2,
                            patience=10, threshold=1e-4)

        if self.early_stop and X_val is not None:
            early_stopping = EarlyStopping(verbose=self.verbose)

        criterion = nn.MSELoss()

        # Train the network
        for i_epoch in range(self.n_epochs):
            if self.verbose:
                print("epoch nb {}".format(i_epoch))

            # Shuffle tensors to have different batches at each epoch
            ind = torch.randperm(n_samples)
            X = X[ind]
            y = y[ind]

            xx = torch.split(X, split_size_or_sections=self.batch_size, dim=0)
            yy = torch.split(y, split_size_or_sections=self.batch_size, dim=0)

            param_group = self.optimizer.param_groups[0]
            lr = param_group['lr']
            if self.verbose:
                print("Current learning rate is: {}".format(lr))
            if lr < 1e-4:
                break

            for bx, by in zip(xx, yy):

                self.optimizer.zero_grad()

                y_hat = self.net(bx)

                loss = criterion(y_hat, by)
                loss.backward()

                # Take gradient step
                self.optimizer.step()

            # Evaluate the train loss
            with torch.no_grad():
                y_hat = self.net(X)
                loss = criterion(y_hat, y)
                mse = loss.item()
                self.mse_train.append(mse)

                var = ((y - y.mean())**2).mean()
                r2 = 1 - mse/var
                self.r2_train.append(r2)

                if self.verbose:
                    print("Train loss - r2: {}, mse: {}".format(r2, mse))

            # Evaluate the validation loss
            if X_val is not None:
                with torch.no_grad():
                    y_hat = self.net(X_val)
                    loss_val = criterion(y_hat, y_val)
                    mse_val = loss_val.item()
                    self.mse_val.append(mse_val)

                    var = ((y_val - y_val.mean())**2).mean()
                    r2_val = 1 - mse_val/var
                    self.r2_val.append(r2_val)
                    if self.verbose:
                        print("Validation loss is: {}".format(r2_val))

                if self.early_stop:
                    early_stopping(mse_val, self.net)
                    if early_stopping.early_stop:
                        if self.verbose:
                            print("Early stopping")
                        break

                self.scheduler.step(mse_val)

        # load the last checkpoint with the best model
        if self.early_stop and early_stopping.early_stop:
            self.net.load_state_dict(early_stopping.checkpoint)

    def predict(self, X):
        """Predict the outcome from partially-observed data.

        Args:
            X: original (n, d) covariates w/ missingness

        Returns:
            predicted outcomes (n, d)
        """
        X = torch.as_tensor(X, dtype=torch.double)

        with torch.no_grad():
            y_hat = self.net(X)

        return np.array(y_hat)
    