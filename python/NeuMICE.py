
import torch
from torch import Tensor, nn
from torch.nn import Linear, Parameter, ReLU, Sequential
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
from sklearn.base import BaseEstimator

from pytorchtools import EarlyStopping


class Mask(nn.Module):
    """A mask non-linearity."""
    mask: Tensor

    def __init__(self, input: Tensor):
        super(Mask, self).__init__()
        self.mask = torch.isnan(input)

    def forward(self, input: Tensor, invert=False) -> Tensor:
        if invert: 
            return ~self.mask * input
        return self.mask*input


class SkipConnection(nn.Module):
    """A skip connection operation."""
    value: Tensor

    def __init__(self, value: Tensor):
        super(SkipConnection, self).__init__()
        self.value = value

    def forward(self, input: Tensor) -> Tensor:
        return input + self.value


class NeuMICEBlock(nn.Module):
    """The NeuMICE block inspired by "What’s a good imputation to predict with
    missing values?" by Marine Le Morvan, Julie Josse, Erwan Scornet,
    Gael Varoquaux."""

    def __init__(self, n_features: int, depth: int,
                 dtype = torch.float) -> None:
        """
        Parameters
        ----------
        n_features : int
            Dimension of inputs and outputs of the NeuMiss block.
        depth : int
            Number of layers (Neumann iterations) in the NeuMiss block.
        dtype : _dtype
            Pytorch dtype for the parameters. Default: torch.float.

        """
        super().__init__()
        self.depth = depth
        self.dtype = dtype
        self.mu = Parameter(torch.empty(n_features, dtype=dtype))
        self.linear = Linear(n_features, n_features, bias=False, dtype=dtype)
        self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        x = x.type(self.dtype)  # Cast tensor to appropriate dtype
        mask = Mask(x)  # Initialize mask non-linearity
        x = torch.nan_to_num(x)  # Fill missing values with 0
        s0 = x - self.mu
        h = x - mask(self.mu, invert=True)  # Subtract masked parameter mu
        skip = SkipConnection(h)  # Initialize skip connection with this value
        
        layer = [self.linear, mask, skip]  # One Neumann iteration
        layers = Sequential(*(layer*self.depth))  # Neumann block

        return layers(s0)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.mu)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.5)

    def extra_repr(self) -> str:
        return 'depth={}'.format(self.depth)


class NeuMICEMLP(nn.Module):
    """A NeuMICE block followed by a MLP."""

    def __init__(self, n_features: int, neumiss_depth: int, mlp_depth: int,
                 mlp_width: int = None, dtype = torch.float) -> None:
        """
        Parameters
        ----------
        n_features : int
            Dimension of inputs.
        neumiss_depth : int
            Number of layers in the NeuMiss block.
        mlp_depth : int
            Number of hidden layers in the MLP.
        mlp_width : int
            Width of the MLP. If None take mlp_width=n_features. Default: None.
        dtype : _dtype
            Pytorch dtype for the parameters. Default: torch.float.

        """
        super().__init__()
        self.n_features = n_features
        self.neumiss_depth = neumiss_depth
        self.mlp_depth = mlp_depth
        self.dtype = dtype
        mlp_width = n_features if mlp_width is None else mlp_width
        self.mlp_width = mlp_width

        b = int(mlp_depth >= 1)
        last_layer_width = mlp_width if b else n_features
        self.layers = Sequential(
            NeuMICEBlock(n_features, neumiss_depth, dtype),
            *[Linear(n_features, mlp_width, dtype=dtype), ReLU()]*b,
            *[Linear(mlp_width, mlp_width, dtype=dtype), ReLU()]*b*(mlp_depth-1),
            *[Linear(last_layer_width, 1, dtype=dtype)],
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.layers(x)
        return out.squeeze()



class NeuMICE_mlp(BaseEstimator):
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

    def __init__(self, mode, depth, n_epochs=1000, batch_size=100, lr=0.01,
                 weight_decay=1e-4, early_stopping=False, optimizer='sgd',
                 residual_connection=False, mlp_depth=0, width_factor=1,
                 init_type='normal', add_mask=False, Sigma=None, mu=None,
                 beta=None, beta0=None, L=None, tmu=None, tsigma=None,
                 coefs=None, verbose=False):
        self.mode = mode
        self.depth = depth
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.early_stop = early_stopping
        self.optimizer = optimizer
        self.residual_connection = residual_connection
        self.mlp_depth = mlp_depth
        self.width_factor = width_factor
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

        n_samples, n_features = X.shape

        X = torch.as_tensor(X, dtype=torch.double)
        y = torch.as_tensor(y, dtype=torch.double)

        if X_val is not None:
            X_val = torch.as_tensor(X_val, dtype=torch.double)
            y_val = torch.as_tensor(y_val, dtype=torch.double)

        self.net = NeuMICEMLP(n_features=n_features,
                           neumiss_depth=self.depth,
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
        X = torch.as_tensor(X, dtype=torch.double)

        with torch.no_grad():
            y_hat = self.net(X)

        return np.array(y_hat)
    