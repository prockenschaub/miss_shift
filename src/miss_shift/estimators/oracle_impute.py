import numpy as np
from sklearn.base import BaseEstimator

from ..networks.mlp import MLP_reg


class OracleImputeMLP(BaseEstimator):
    """Imputes using an oracle and then runs a MLP on the imputed data.
       The oracle is for Gaussian data with M(C)AR or Gaussian self-masking
       missing data mechanisms.

    Parameters
    ----------

    add_mask: bool
        Whether or not to concatenate the mask with the data.

    mdm: str
        The missing data mechanism: either 'MCAR', 'MAR' or 'gaussian_sm'.

    mu: array-like, shape (n_features, )
        Mean of the Gaussian distribution.

    Sigma: array-like, shape (n_features, n_features)
        Covariance matrix of the Gaussian distribution.

    tmu: array-like, shape (n_features, )
        Mean of the Gaussian self-masking distribution. Only used if
        `mdm=gaussian_sm'.

    tsigma2: array-like, shape (n_features, )
        Variances of the Gaussian self-masking distributions. Only used if
        `mdm=gaussian_sm'.

    est_params: dict
        The dictionary containing the parameters for the MLP.
    """

    def __init__(self, data_params, add_mask, **mlp_params):
        self.data_params = data_params
        self.add_mask = add_mask
        self.mlp_params = mlp_params

        self._reg = MLP_reg(is_mask=add_mask, **self.mlp_params)

    def oracle_impute(self, X):
        (_, mu, Sigma, _, masking_params, _, _, _, _) = self.data_params
        
        T = X.copy()
        for t in T:
            m = np.isnan(t)
            obs = np.where(~m)[0]
            mis = np.where(m)[0]

            sigma_obs = Sigma[np.ix_(obs, obs)]
            sigma_obs_inv = np.linalg.inv(sigma_obs)
            sigma_misobs = Sigma[np.ix_(mis, obs)]
            sigma_mis = Sigma[np.ix_(mis, mis)]

            mu_mis_obs = mu[mis] + sigma_misobs.dot(
                sigma_obs_inv).dot(t[obs] - mu[obs])

            if masking_params['mdm'] in ['MCAR', 'MAR_logistic']:
                t[mis] = mu_mis_obs

            elif masking_params['mdm'] == 'gaussian_sm':
                sm_params = masking_params['sm_param']
                tsigma2 = sm_params['sigma2_tilde']
                tmu = sm_params['tmu']

                sigma_mis_obs = sigma_mis - \
                                sigma_misobs.dot(sigma_obs_inv).dot(sigma_misobs.T)
                sigma_mis_obs_inv = np.linalg.inv(sigma_mis_obs)

                D_mis_inv = np.diag(1 / tsigma2[mis])

                S = np.linalg.inv(D_mis_inv + sigma_mis_obs_inv)
                s = S.dot(D_mis_inv.dot(tmu[mis]) +
                          sigma_mis_obs_inv.dot(mu_mis_obs))

                t[mis] = s

            else:
                raise ValueError('`mdm`shouyld be one of `MCAR`, `MAR`, or' +
                                 '`gaussian_sm`')
        return T

    def fit(self, X, y, X_val=None, y_val=None):
        M = np.isnan(X)
        M_val = np.isnan(X_val)
        T = self.oracle_impute(X)
        T_val = self.oracle_impute(X_val)
        if self.add_mask:
            T = np.hstack((T, M))
            T_val = np.hstack((T_val, M_val))
        self._reg.fit(T, y, X_val=T_val, y_val=y_val)
        return self

    def predict(self, X):
        M = np.isnan(X)
        T = self.oracle_impute(X)
        if self.add_mask:
            T = np.hstack((T, M))
        return self._reg.predict(T)