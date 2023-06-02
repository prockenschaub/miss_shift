# Based on https://github.com/marineLM/NeuMiss_sota

import torch
from torch import Tensor, nn
from torch.nn import Sequential, Linear, ReLU
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import math
from math import sqrt, pi
from scipy.stats import norm
import numpy as np
from sklearn.base import BaseEstimator

from misc.pytorchtools import EarlyStopping
from networks.neumiss import NeuMissBlock


def f_star(X, beta, curvature, link='linear'):
    dot_product = X.matmul(beta[1:]) + beta[0]

    if link == 'linear':
        y = dot_product
    elif link == 'square':
        y = curvature*(dot_product-1)**2
    elif link == 'cube':
        y = beta[0] + curvature*dot_product**3
        linear_coef = torch.pow(3*torch.sqrt(3)/2*torch.sqrt(curvature)*4, 2/3)
        y -= linear_coef*dot_product
    elif link == 'stairs':
        y = dot_product - 1
        for a, b in zip([2, -4, 2], [-0.8, -1, -1.2]):
            tmp = torch.sqrt(torch.tensor(pi)/8)*curvature*(dot_product + b)
            y += a*(1 + torch.erf(tmp / torch.sqrt(torch.tensor(2))))/2
    elif link == 'discontinuous_linear':
        y = dot_product + (dot_product > 1)*3

    return y


class NeuMissOracle(BaseEstimator):

    def __init__(self, data_params, depth, n_epochs=1000, batch_size=100, lr=0.01,
                 weight_decay=1e-4, early_stopping=False, optimizer='sgd',
                 init_type='normal', verbose=False):
        self.data_params = data_params
        self.depth = depth
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.early_stop = early_stopping
        self.optimizer = optimizer
        self.init_type = init_type
        self.verbose = verbose

        self.r2_train = []
        self.mse_train = []
        self.r2_val = []
        self.mse_val = []

    def fit(self, X, y, X_val=None, y_val=None):
        (_, _, _, beta, _, _, link, curvature) = self.data_params
        beta = torch.tensor(beta, dtype=torch.double)

        n_samples, n_features = X.shape

        X = torch.as_tensor(X, dtype=torch.double)
        y = torch.as_tensor(y, dtype=torch.double)

        if X_val is not None:
            X_val = torch.as_tensor(X_val, dtype=torch.double)
            y_val = torch.as_tensor(y_val, dtype=torch.double)

        self.net = NeuMissBlock(n_features, self.depth, dtype=torch.double)

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

                rep = self.net(bx)
                y_hat = f_star(rep, beta, curvature, link)

                loss = criterion(y_hat, by)
                loss.backward()

                # Take gradient step
                self.optimizer.step()

            # Evaluate the train loss
            with torch.no_grad():
                rep = self.net(X)
                y_hat = f_star(rep, beta, curvature, link)
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
                    rep = self.net(X_val)
                    y_hat = f_star(rep, beta, curvature, link)
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
        (_, _, _, beta, _, _, link, curvature) = self.data_params
        X = torch.as_tensor(X, dtype=torch.double)
        beta = torch.tensor(beta, dtype=torch.double)

        with torch.no_grad():
            rep = self.net(X)
            y_hat = f_star(rep, beta, curvature, link)

        return np.array(y_hat)


class NeumissLegacy(nn.Module):
    def __init__(self, n_features, depth,
                 init_type):
        super().__init__()
        self.depth = depth
        self.n_features = n_features
        self.relu = nn.ReLU()

        # Create the parameters of the network
        l_W = [torch.empty(n_features, n_features, dtype=torch.double)
               for _ in range(self.depth)]
        Wc = torch.empty(n_features, n_features, dtype=torch.double)
        beta = torch.empty(1*n_features, dtype=torch.double)
        mu = torch.empty(n_features, dtype=torch.double)
        b = torch.empty(1, dtype=torch.double)

        # Initialize the parameters of the network
        if init_type == 'normal':
            for W in l_W:
                nn.init.xavier_normal_(W)
            nn.init.xavier_normal_(Wc)
            nn.init.normal_(beta)
            nn.init.normal_(mu)
            nn.init.normal_(b)

        elif init_type == 'uniform':
            bound = 1 / math.sqrt(n_features)
            for W in l_W:
                nn.init.kaiming_uniform_(W, a=math.sqrt(5))
            nn.init.kaiming_uniform_(Wc, a=math.sqrt(5))
            nn.init.uniform_(beta, -bound, bound)
            nn.init.uniform_(mu, -bound, bound)
            nn.init.normal_(b)

        # Make tensors learnable parameters
        self.l_W = [torch.nn.Parameter(W) for W in l_W]
        for i, W in enumerate(self.l_W):
            self.register_parameter('W_{}'.format(i), W)
        self.Wc = torch.nn.Parameter(Wc)
        self.beta = torch.nn.Parameter(beta)
        self.mu = torch.nn.Parameter(mu)
        self.b = torch.nn.Parameter(b)
        
    def forward(self, x, m, phase='train'):
        """
        Parameters:
        ----------
        x: tensor, shape (batch_size, n_features)
            The input data imputed by 0.
        m: tensor, shape (batch_size, n_features)
            The missingness indicator (0 if observed and 1 if missing).
        """

        h0 = x + m*self.mu
        h = x - (1-m)*self.mu
        h_res = x - (1-m)*self.mu

        if len(self.l_W) > 0:
            S0 = self.l_W[0]
            h = torch.matmul(h, S0)*(1-m)

        for W in self.l_W[1:self.depth]:
            h = torch.matmul(h, W)*(1-m)
            h += h_res

        h = torch.matmul(h, self.Wc)*m + h0
        return h



class NeumissLegacyOracle(BaseEstimator):
    """The Neumiss + MLP neural network

    Parameters
    ----------

    mode: str
        One of:
        * 'baseline': The weight matrices for the Neumann iteration are not
        shared.
        * 'shared': The weight matrices for the Neumann iteration are shared.
        * 'shared_accelerated': The weight matrices for the Neumann iteration
        are shared and one corefficient per residual connection can be learned
        for acceleration.

    depth: int
        The depth of the NeuMiss block.

    n_epochs: int
        The maximum number of epochs.

    batch_size: int
        The batch size.

    lr: float
        The learning rate.

    weight_decay: float
        The weight decay parameter.

    early_stopping: boolean
        If True, early stopping is used based on the validaton set.

    optimizer: srt
        One of `sgd`or `adam`.

    residual_connection: boolean
        If True, the residual connection of the Neumann network are
        implemented.

    mlp_depth: int
        The depth of the MLP stacked on top of the Neumann iterations.

    width_factor: int
        The width of the MLP stacked on top of the NeuMiss layer is calculated
        as width_factor times n_features.

    init_type: str
        The type of initialisation for the parameters. Either 'normal',
        'uniform', or 'custom_normal'. If 'custom_normal', the values provided
        for the parameter `Sigma`, `mu`, `L` (and `coefs` if accelerated) are
        used to initialise the Neumann block.

    add_mask: boolean
        If True, the mask is concatenated to the output of the NeuMiss block.

    verbose: boolean
    """

    def __init__(self, data_params, depth, n_epochs=1000, batch_size=100, lr=0.01,
                 weight_decay=1e-4, early_stopping=False, optimizer='sgd',
                 init_type='normal', add_mask=False, Sigma=None, mu=None,
                 beta=None, beta0=None, L=None, tmu=None, tsigma=None,
                 coefs=None, verbose=False):
        self.data_params = data_params
        self.depth = depth
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.early_stop = early_stopping
        self.optimizer = optimizer
        self.init_type = init_type
        self.add_mask = add_mask
        self.Sigma = Sigma
        self.mu = mu
        self.beta = beta
        self.beta0 = beta0
        self.L = L
        self.tmu = tmu
        self.tsigma = tsigma
        self.coefs = coefs
        self.verbose = verbose

        self.r2_train = []
        self.mse_train = []
        self.r2_val = []
        self.mse_val = []

    def fit(self, X, y, X_val=None, y_val=None):
        (_, _, _, beta, _, _, link, curvature) = self.data_params
        beta = torch.tensor(beta, dtype=torch.double)

        M = np.isnan(X)
        X = np.nan_to_num(X)

        n_samples, n_features = X.shape

        if X_val is not None:
            M_val = np.isnan(X_val)
            X_val = np.nan_to_num(X_val)

        M = torch.as_tensor(M, dtype=torch.double)
        X = torch.as_tensor(X, dtype=torch.double)
        y = torch.as_tensor(y, dtype=torch.double)

        if X_val is not None:
            M_val = torch.as_tensor(M_val, dtype=torch.double)
            X_val = torch.as_tensor(X_val, dtype=torch.double)
            y_val = torch.as_tensor(y_val, dtype=torch.double)

        self.net = NeumissLegacy(n_features=n_features,
                           depth=self.depth,
                           init_type=self.init_type)

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
            M = M[ind]
            y = y[ind]

            xx = torch.split(X, split_size_or_sections=self.batch_size, dim=0)
            mm = torch.split(M, split_size_or_sections=self.batch_size, dim=0)
            yy = torch.split(y, split_size_or_sections=self.batch_size, dim=0)

            param_group = self.optimizer.param_groups[0]
            lr = param_group['lr']
            if self.verbose:
                print("Current learning rate is: {}".format(lr))
            if lr < 1e-4:
                break

            for bx, bm, by in zip(xx, mm, yy):

                self.optimizer.zero_grad()

                rep = self.net(bx, bm)
                y_hat = f_star(rep, beta, curvature, link)

                loss = criterion(y_hat, by)
                loss.backward()

                # Take gradient step
                self.optimizer.step()

            # Evaluate the train loss
            with torch.no_grad():
                rep = self.net(X, M, phase='test')
                y_hat = f_star(rep, beta, curvature, link)
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
                    rep = self.net(X_val, M_val, phase='test')
                    y_hat = f_star(rep, beta, curvature, link)
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
        (_, _, _, beta, _, _, link, curvature) = self.data_params
        beta = torch.tensor(beta, dtype=torch.double)
        M = np.isnan(X)
        X = np.nan_to_num(X)

        M = torch.as_tensor(M, dtype=torch.double)
        X = torch.as_tensor(X, dtype=torch.double)

        with torch.no_grad():
            rep = self.net(X, M, phase='test')
            y_hat = f_star(rep, beta, curvature, link)

        return np.array(y_hat)