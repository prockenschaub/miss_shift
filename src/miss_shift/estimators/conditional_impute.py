import numpy as np
from sklearn.base import BaseEstimator
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

from ..networks.mlp import MLP_reg
from ..misc.iterativeimputer import FastIterativeImputer


class ImputeMLP(BaseEstimator):
    """Imputes with mean or iterative imputation and then runs an MLP on the imputed data.

    Args:
        add_mask: whether or not to concatenate the mask with the data
        imputation_type: one of 'mean', 'ICE', or 'MICE'
        n_draws: number of imputations to draw (only relevant for MICE)
        use_y_for_impute: should the outcome be used for imputation (only relevant for ICE and MICE)
        verbose: flag to print detailed information about training to the console.
        mlp_params: the dictionary containing the parameters for the MLP
    """

    def __init__(
        self,
        add_mask: bool,
        imputation_type: str,
        n_draws=5,
        use_y_for_impute=False,
        verbose=False,
        **mlp_params
    ):

        self.add_mask = add_mask
        self.imputation_type = imputation_type
        self.mlp_params = mlp_params
        self.n_draws = n_draws
        self.use_y_for_impute = use_y_for_impute

        if self.imputation_type == "mean":
            self._imp = SimpleImputer(missing_values=np.nan, strategy="mean")
        elif self.imputation_type == "ICE":
            self._imp = IterativeImputer(random_state=0, verbose=2 * int(verbose))
        elif self.imputation_type == "MICE":
            self._imp = FastIterativeImputer(
                random_state=0,
                sample_posterior=True,
                max_iter=5,
                verbose=2 * int(verbose),
            )

        self._reg = MLP_reg(is_mask=add_mask, verbose=verbose, **self.mlp_params)

    def concat_mask(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Concatenate the missingness indicators to the imputed covariates

        Args:
            X: original (n, d) covariates w/ missingness
            T: imputed (n, d) covariates w/o missingness (or (n, n_draws, d) in the case of MICE)

        Returns:
            concatenated (n, 2*d) data
        """
        if self.imputation_type == "MICE":
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
        if self.imputation_type == "MICE":
            T = []
            for _ in range(self.n_draws):
                T.append(self._imp.transform(X))
            return np.stack(T)
        else:
            return self._imp.transform(X)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ):
        """First train the imputer and then the MLP

        Args:
            X: original (n, d) covariates w/ missingness
            y: original (n, ) outcomes
            X_val: optional covariates w/ missingness that are passively imputed. Defaults to None.
            y_val: optional outcomes that may be used for passively imputed. Defaults to None.
        """
        if self.use_y_for_impute:
            # Add the outcome to the dataset to use it during imputation
            X = np.c_[X, y]
            X_val = np.c_[X_val, y_val]

        self._imp.fit(X)
        T = self.impute(X)
        T_val = self.impute(X_val)

        if self.use_y_for_impute:
            # Remove the outcome from all datasets to fit the regressor
            X = X[..., :-1]
            X_val = X_val[..., :-1]
            T = T[..., :-1]
            T_val = T_val[..., :-1]

        if self.add_mask:
            T = self.concat_mask(X, T)
            T_val = self.concat_mask(X_val, T_val)

        self._reg.fit(T, y, X_val=T_val, y_val=y_val)

        if self.use_y_for_impute:
            # finally, refit the imputation for prediction at test time
            self._imp.fit(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the outcome from partially-observed data.

        Note: for MICE, the prediction is averaged across the `n_draw`s

        Args:
            X: original (n, d) covariates w/ missingness

        Returns:
            predicted outcomes (n, d)
        """
        T = self.impute(X)
        if self.add_mask:
            T = self.concat_mask(X, T)
        return self._reg.predict(T)
