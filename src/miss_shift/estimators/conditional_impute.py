import numpy as np
from sklearn.base import BaseEstimator
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

from ..networks.mlp import MLP_reg
from ..misc.iterativeimputer import FastIterativeImputer

class ImputeMLPPytorch(BaseEstimator):
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

    def __init__(self, add_mask, imputation_type, n_draws=5, verbose=False, **mlp_params):

        self.add_mask = add_mask
        self.imputation_type = imputation_type
        self.mlp_params = mlp_params
        self.n_draws = n_draws

        if self.imputation_type == 'mean':
            self._imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        elif self.imputation_type == 'MICE':
            self._imp = IterativeImputer(random_state=0, verbose=2*int(verbose))
        elif self.imputation_type == 'MultiMICE':
            self._imp = FastIterativeImputer(random_state=0, sample_posterior=True, max_iter=5, verbose=2*int(verbose))

        self._reg = MLP_reg(is_mask=add_mask, verbose=verbose, **self.mlp_params)

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

    def impute(self, X):
        if self.imputation_type == 'MultiMICE':
            T = []
            for _ in range(self.n_draws):
                T.append(self._imp.transform(X))
            return np.stack(T)
        else:
            return self._imp.transform(X)

    def fit(self, X, y, X_val=None, y_val=None):
        self._imp.fit(X)
        T = self.impute(X)
        T_val = self.impute(X_val)

        if self.add_mask:
            T = self.concat_mask(X, T)
            T_val = self.concat_mask(X_val, T_val)
        self._reg.fit(T, y, X_val=T_val, y_val=y_val)
        return self

    def predict(self, X):
        T = self.impute(X)
        if self.add_mask:
            T = self.concat_mask(X, T)
        return self._reg.predict(T)
