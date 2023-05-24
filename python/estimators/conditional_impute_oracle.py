import numpy as np
from sklearn.base import BaseEstimator
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from networks.mlp import MLP_reg

from misc.iterativeimputer import FastIterativeImputer

from scipy.stats import norm
from math import sqrt, pi

def f_star(X, beta, curvature, link='linear'):
    dot_product = X.dot(beta[1:]) + beta[0]

    if link == 'linear':
        y = dot_product
    elif link == 'square':
        y = curvature*(dot_product-1)**2
    elif link == 'cube':
        y = beta[0] + curvature*dot_product**3
        linear_coef = pow(3*sqrt(3)/2*sqrt(curvature)*4, 2/3)
        y -= linear_coef*dot_product
    elif link == 'stairs':
        y = dot_product - 1
        for a, b in zip([2, -4, 2], [-0.8, -1, -1.2]):
            tmp = sqrt(pi/8)*curvature*(dot_product + b)
            y += a*norm.cdf(tmp)
    elif link == 'discontinuous_linear':
        y = dot_product + (dot_product > 1)*3

    return y


class MICEOracle(BaseEstimator):
    
    def __init__(self, data_params, n_draws=5):
        self.data_params = data_params
        self.n_draws = n_draws

        self._imp = FastIterativeImputer(random_state=0, sample_posterior=True, max_iter=5)

    def impute(self, X):
        T = []
        for _ in range(self.n_draws):
            T.append(self._imp.transform(X))
        return np.stack(T).reshape((self.n_draws * X.shape[0], X.shape[1]), order='F')

    def fit(self, X, y, X_val=None, y_val=None):
        self._imp.fit(X)
        return self

    def predict(self, X):
        (_, _, _, beta, _, _, link, curvature) = self.data_params
        T = self.impute(X)
        y_pred = f_star(T, beta, curvature, link)  # y_pred [nb_samples*n_draws]
        y_pred = np.reshape(y_pred, [X.shape[0], -1])
        y_pred = np.mean(y_pred, axis=-1)
        return y_pred
