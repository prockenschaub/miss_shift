import numpy as np
from sklearn.base import BaseEstimator

import torch
from torch import nn
import torch.distributions as td
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..misc.pytorchtools import EarlyStopping
from ..networks.mlp import MLP_reg

def weights_init(layer) -> None:
    if type(layer) == nn.Linear:
        torch.nn.init.orthogonal_(layer.weight)


class MIWAE(nn.Module):
    def __init__(self, n_inputs, width, latent_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.n_inputs = n_inputs
        self.latent_size = latent_size

        self.p_z = td.Independent(
            td.Normal(
                loc=torch.zeros(self.latent_size),
                scale=torch.ones(self.latent_size),
            ),
            1,
        )

        self.encoder = nn.Sequential(
            torch.nn.Linear(n_inputs, width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, width),
            torch.nn.ReLU(),
            torch.nn.Linear(
                width, 2 * latent_size
            ),  # the encoder will output both the mean and the diagonal covariance
        )

        self.decoder = nn.Sequential(
            torch.nn.Linear(latent_size, width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, width),
            torch.nn.ReLU(),
            torch.nn.Linear(
                width, 3 * n_inputs
            ),  # the decoder will output both the mean, the scale, and the number of degrees of freedoms (hence the 3*p)
        )

        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)

    def forward(self, iota_x, mask, L):
        out_encoder = self.encoder(iota_x)
        q_zgivenxobs = td.Independent(
            td.Normal(
                loc=out_encoder[..., : self.latent_size],
                scale=torch.nn.Softplus()(
                    out_encoder[..., self.latent_size : (2 * self.latent_size)]
                ),
            ),
            1,
        )

        zgivenx = q_zgivenxobs.rsample([L])
        zgivenx_flat = zgivenx.reshape([-1, self.latent_size])

        out_decoder = self.decoder(zgivenx_flat)
        all_means_obs_model = out_decoder[..., :self.n_inputs]
        all_scales_obs_model = (
            torch.nn.Softplus()(out_decoder[..., self.n_inputs : (2 * self.n_inputs)]) + 0.001
        )
        all_degfreedom_obs_model = (
            torch.nn.Softplus()(out_decoder[..., (2 * self.n_inputs) : (3 * self.n_inputs)]) + 3
        )

        data_flat = torch.Tensor.repeat(iota_x, [L, 1]).reshape([-1, 1])
        tiledmask = torch.Tensor.repeat(mask, [L, 1])

        all_log_pxgivenz_flat = torch.distributions.StudentT(
            loc=all_means_obs_model.reshape([-1, 1]),
            scale=all_scales_obs_model.reshape([-1, 1]),
            df=all_degfreedom_obs_model.reshape([-1, 1]),
        ).log_prob(data_flat)
        all_log_pxgivenz = all_log_pxgivenz_flat.reshape([-1, self.n_inputs])

        logpxobsgivenz = torch.sum(all_log_pxgivenz * tiledmask, 1).reshape(
            [L, -1]
        )
        logpz = self.p_z.log_prob(zgivenx)
        logq = q_zgivenxobs.log_prob(zgivenx)

        xgivenz = td.Independent(
            td.StudentT(
                loc=all_means_obs_model,
                scale=all_scales_obs_model,
                df=all_degfreedom_obs_model,
            ),
            1,
        )

        return xgivenz, logpxobsgivenz, logpz, logq



class MIWAERegressor(BaseEstimator):
    """Imputes and then runs a MLP (Pytorch based, same as for NeuMiss)
    on the imputed data.

    Parameters
    ----------

    add_mask: bool
        Whether or not to concatenate the mask with the data.

    imputation_type: str
        One of 'mean' or 'MICE'.

    est_params: dict
        The dictionary containing the parameters for the MLP.
    """

    def __init__(self, latent_size, encoder_width, K=20, n_draws=5, device='cpu', early_stopping=False, verbose=False, **mlp_params):

        self.latent_size = latent_size
        self.encoder_width = encoder_width
        self.K = K

        self.n_epochs = mlp_params.get('n_epochs', 100)
        self.batch_size = mlp_params.get('batch_size', 32)
        self.weight_decay = mlp_params.get('weight_decay', 0)
        self.lr = mlp_params.get('lr', 1.e-3)

        self.mlp_params = mlp_params
        self.n_draws = n_draws

        self.verbose = verbose
        self.device = device
        self.early_stop = early_stopping

        self._reg = MLP_reg(is_mask=False, early_stopping=early_stopping, verbose=verbose, **self.mlp_params)

    def concat_mask(self, X, T):
        if self.imputation_type == 'MultiMICE':
            # replicate the mask, because T is now of shape [n_samples, n_draws, n_features]
            M = np.isnan(X)
            M = np.repeat(M, self.n_draws, axis=0).reshape(T.shape)
            T = np.concatenate((T, M), axis=2)
        else:
            M = np.isnan(X)
            T = np.hstack((T, M))
        return T

    def _miwae_loss(self, iota_x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _, logpxobsgivenz, logpz, logq = self._imp(iota_x, mask, self.K)
        neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpz - logq, 0))

        return neg_bound

    def _miwae_impute(
        self, iota_x: torch.Tensor, mask: torch.Tensor, L: int
    ) -> torch.Tensor:
        batch_size = iota_x.shape[0]
        p = iota_x.shape[1]
        xgivenz, logpxobsgivenz, logpz, logq = self._imp(iota_x, mask, L)

        imp_weights = torch.nn.functional.softmax(
            logpxobsgivenz + logpz - logq, 0
        )  # these are w_1,....,w_L for all observations in the batch
        xms = xgivenz.sample().reshape([L, batch_size, p])
        xm = torch.einsum("ki,kij->ij", imp_weights, xms)

        return xm

    def _train_imputer(self, X, X_val):
        X = torch.from_numpy(X).float().to(self.device)
        mask = np.isfinite(X.cpu()).bool().to(self.device)

        xhat_0 = torch.clone(X)
        xhat_0[np.isnan(X.cpu()).bool()] = 0

        X_val = torch.from_numpy(X_val).float().to(self.device)
        mask_val = np.isfinite(X_val.cpu()).bool().to(self.device)

        xhat_0_val = torch.clone(X_val)
        xhat_0_val[np.isnan(X_val.cpu()).bool()] = 0

        n = X.shape[0]  # number of observations
        p = X.shape[1]  # number of features

        self._imp = MIWAE(p, self.encoder_width, self.latent_size)
        self._imp.to(self.device)

        optimizer = optim.Adam(self._imp.parameters(), lr=self.lr,)

        xhat = torch.clone(xhat_0)  # This will be out imputed data matrix

        self.scheduler = ReduceLROnPlateau(
                            optimizer, mode='min', factor=0.2,
                            patience=10, threshold=1e-4)

        if self.early_stop and X_val is not None:
            early_stopping = EarlyStopping(verbose=self.verbose)

        bs = min(self.batch_size, n)

        for ep in range(1, self.n_epochs):
            perm = np.random.permutation(
                n
            )  # We use the "random reshuffling" version of SGD
            batches_data = np.array_split(
                xhat_0[
                    perm,
                ],
                int(n / bs),
            )
            batches_mask = np.array_split(
                mask[
                    perm,
                ],
                int(n / bs),
            )
            for it in range(len(batches_data)):
                optimizer.zero_grad()
                self._imp.zero_grad()
                b_data = batches_data[it]
                b_mask = batches_mask[it].float()
                loss = self._miwae_loss(iota_x=b_data, mask=b_mask)
                loss.backward()
                optimizer.step()

            # Evaluate the train loss
            with torch.no_grad():
                loss = self._miwae_loss(iota_x=xhat_0, mask=mask).cpu().data.numpy()
                if self.verbose:
                    print("Epoch {} - MIWAE training loss: {}".format(ep, loss))

            # Evaluate the validation loss
            if X_val is not None:
                with torch.no_grad():
                    loss_val = self._miwae_loss(iota_x=xhat_0_val, mask=mask_val).cpu().data.numpy()
            
                    if self.verbose:
                        print("Epoch {} - MIWAE validation loss is: {}".format(ep, loss_val))

                if self.early_stop:
                    early_stopping(loss_val, self._imp)
                    if early_stopping.early_stop:
                        if self.verbose:
                            print("Early stopping")
                        break

                self.scheduler.step(loss_val)

        return self

    def impute(self, X):
        X = torch.from_numpy(X).float().to(self.device)
        mask = np.isfinite(X.cpu()).bool().to(self.device)
        
        xhat_0 = torch.clone(X)
        xhat_0[np.isnan(X.cpu()).bool()] = 0

        T = []
        for _ in range(self.n_draws):
            with torch.no_grad():
                xhat = torch.clone(xhat_0)
                xhat[~mask] = self._miwae_impute(iota_x=xhat_0, mask=mask, L=10,)[~mask]
                T.append(xhat.numpy())
        return np.stack(T)

    def fit(self, X, y, X_val=None, y_val=None, X_true=None):
        self._train_imputer(X, X_val)
        T = self.impute(X)
        T_val = self.impute(X_val)

        self._reg.fit(T, y, X_val=T_val, y_val=y_val)
        return self

    def predict(self, X):
        T = self.impute(X)
    
        return self._reg.predict(T)
