import numpy as np
from sklearn.base import BaseEstimator

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..misc.pytorchtools import EarlyStopping
from ..networks.miwae import MIWAE
from ..networks.mlp import MLP_reg


class MIWAEMLP(BaseEstimator):
    """Imputes with MIWAE and then runs an MLP on the imputed data.

    Args:
        input_size: number of covariates
        encoder_width: size of the intermediate layers of the autoencoder
        latent_size: size of the latent space
        K: number of samples to draw during training
        n_draws: number of imputations to draw during inference
        add_mask: whether or not to concatenate the mask with the data
        mode: training mode. If 'joint', MIWAE and the MLP are trained together.
            However, this is computationally intensive, since for each MLP
            configuration, a new MIWAE must be trained. Since the same MIWAE may
            be used for different MLP configurations, we can also train MIWAE
            separately and then mix and match. One can do so via 'imputer-only'
            or 'predictor-only'. Defaults to 'joint'.
        save_dir: if mode == 'imputer-only', file path where to store the pretrained
            MIWAE model
        device: device on which to train. Defaults to 'cpu'.
        mlp_params: the dictionary containing the parameters for the MLP
    """

    def __init__(
        self,
        input_size: int,
        encoder_width: int,
        latent_size: int,
        K: int = 20,
        n_draws: int = 5,
        add_mask: bool = False,
        mode: str = "joint",
        save_dir: str = "tmp",
        device: str = "cpu",
        **mlp_params,
    ):
        self.input_size = input_size
        self.encoder_width = encoder_width
        self.latent_size = latent_size
        self.K = K
        self.n_draws = n_draws

        self.mlp_params = mlp_params
        self.n_epochs = mlp_params.get("n_epochs", 100)
        self.batch_size = mlp_params.get("batch_size", 32)
        self.weight_decay = mlp_params.get("weight_decay", 0)
        self.lr = mlp_params.get("lr", 1.0e-3)
        self.verbose = mlp_params.get("verbose", False)
        self.early_stop = mlp_params.get("early_stopping", False)

        self.add_mask = add_mask

        self.mode = mode
        self.save_path = (
            f"{save_dir}/miwae_{input_size}_{encoder_width}_{latent_size}_{K}.pt"
        )
        self.device = device

        self._imp = MIWAE(input_size, encoder_width, latent_size)
        self._reg = MLP_reg(is_mask=False, **mlp_params)

    def _miwae_loss(self, iota_x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Calculate the MIWAE loss used during training

        Args:
            iota_x: zero-imputed covariates
            mask: mask indicating the missing values in the covariates

        Returns:
            loss (negative MIWAE bound)
        """
        _, logpxobsgivenz, logpz, logq = self._imp(iota_x, mask, self.K)
        neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpz - logq, 0))

        return neg_bound

    def _miwae_impute(
        self, iota_x: torch.Tensor, mask: torch.Tensor, L: int
    ) -> torch.Tensor:
        """Impute using MIWAE

        Args:
            iota_x: zero-imputed covariates
            mask: mask indicating the missing values in the covariates
            L: number of candidates to draw for imputation, which will be combined
                based on their importance weights

        Returns:
            imputed data
        """
        batch_size = iota_x.shape[0]
        p = iota_x.shape[1]
        xgivenz, logpxobsgivenz, logpz, logq = self._imp(iota_x, mask, L)

        imp_weights = torch.nn.functional.softmax(
            logpxobsgivenz + logpz - logq, 0
        )  # these are w_1,....,w_L for all observations in the batch
        xms = xgivenz.sample().reshape([L, batch_size, p])
        xm = torch.einsum("ki,kij->ij", imp_weights, xms)

        return xm

    def _conditional_impute(self, iota_x: torch.Tensor, mask: torch.Tensor):
        """Impute using AE (i.e., the conditional expectation)

        Args:
            iota_x: zero-imputed covariates
            mask: mask indicating the missing values in the covariates

        Returns:
            imputed data
        """
        return self._imp(iota_x, mask, 0)

    def _train_imputer(self, X: np.ndarray, X_val: np.ndarray = None):
        """Training routine for the MIWAE imputer

        Args:
            X: original (n, d) covariates w/ missingness
            X_val: optional covariates w/ missingness that are passively imputed. Defaults to None.
        """
        X = torch.from_numpy(X).float().to(self.device)
        mask = np.isfinite(X.cpu()).bool().to(self.device)

        xhat_0 = torch.clone(X)
        xhat_0[np.isnan(X.cpu()).bool()] = 0

        if X_val is not None:
            X_val = torch.from_numpy(X_val).float().to(self.device)
            mask_val = np.isfinite(X_val.cpu()).bool().to(self.device)

            xhat_0_val = torch.clone(X_val)
            xhat_0_val[np.isnan(X_val.cpu()).bool()] = 0

        n = X.shape[0]  # number of observations
        p = X.shape[1]  # number of features

        optimizer = optim.Adam(
            self._imp.parameters(),
            lr=self.lr,
        )

        self.scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=5, threshold=1e-4
        )

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
                    loss_val = (
                        self._miwae_loss(iota_x=xhat_0_val, mask=mask_val)
                        .cpu()
                        .data.numpy()
                    )

                    if self.verbose:
                        print(
                            "Epoch {} - MIWAE validation loss is: {}".format(
                                ep, loss_val
                            )
                        )

                if self.early_stop:
                    early_stopping(loss_val, self._imp)
                    if early_stopping.early_stop:
                        if self.verbose:
                            print("Early stopping")
                        break

                self.scheduler.step(loss_val)

    def concat_mask(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Concatenate the missingness indicators to the imputed covariates

        Args:
            X: original (n, d) covariates w/ missingness
            T: imputed (n, d) covariates w/o missingness (or (n, n_draws, d) in the case of MICE)

        Returns:
            concatenated (n, 2*d) data
        """
        if self.n_draws > 0:
            # replicate the mask, because T is now of shape [n_samples, n_draws, n_features]
            M = np.isnan(X)
            M = np.repeat(M, self.n_draws, axis=0).reshape(T.shape)
            T = np.concatenate((T, M), axis=2)
        else:
            M = np.isnan(X)
            T = np.hstack((T, M))
        return T

    def impute(self, X: np.ndarray) -> np.ndarray:
        """Perform imputation using a trained imputer

        Args:
            X: original (n, d) covariates w/ missingness

        Returns:
            imputed (n, d) or (n, n_draws, d) covariates w/o missingness
        """
        X = torch.from_numpy(X).float().to(self.device)
        mask = np.isfinite(X.cpu()).bool().to(self.device)

        xhat_0 = torch.clone(X)
        xhat_0[np.isnan(X.cpu()).bool()] = 0

        if self.n_draws == 0:
            # Perform imputation with the conditional expectation
            with torch.no_grad():
                xhat_0[~mask] = self._conditional_impute(iota_x=xhat_0, mask=mask)[
                    ~mask
                ]
            return xhat_0
        else:
            # Perform imputation by drawing from the conditional distribution
            T = []
            for _ in range(self.n_draws):
                with torch.no_grad():
                    xhat = torch.clone(xhat_0)
                    xhat[~mask] = self._miwae_impute(
                        iota_x=xhat_0,
                        mask=mask,
                        L=10,
                    )[~mask]
                    T.append(xhat.numpy())
            return np.stack(T)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ):
        """First train MIWAE (or load a pretrained model) and then train the MLP

        Args:
            X: original (n, d) covariates w/ missingness
            y: original (n, ) outcomes
            X_val: optional covariates w/ missingness that are passively imputed. Defaults to None.
            y_val: optional outcomes that may be used for passively imputed. Defaults to None.
        """
        if self.mode in ["join", "imputer-only"]:
            self._train_imputer(X, X_val)
            if self.mode == "imputer-only":
                torch.save(self._imp.state_dict(), self.save_path)
        else:
            self._imp.load_state_dict(torch.load(self.save_path))

        if self.mode in ["joint", "predictor-only"]:
            T = self.impute(X)
            T_val = self.impute(X_val)

            if self.add_mask:
                T = self.concat_mask(X, T)
                T_val = self.concat_mask(X_val, T_val)

            self._reg.fit(T, y, X_val=T_val, y_val=y_val)

    def predict(self, X):
        """Predict the outcome from partially-observed data.

        Note: for MIWAE, the prediction is averaged across the `n_draw`s

        Args:
            X: original (n, d) covariates w/ missingness

        Returns:
            predicted outcomes (n, d)
        """
        T = self.impute(X)
        if self.add_mask:
            T = self.concat_mask(X, T)
        return self._reg.predict(T)
