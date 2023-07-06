from sklearn.base import BaseEstimator
import numpy as np
from scipy.stats import norm
from math import sqrt, pi


class ProbabilisticBayesPredictor(BaseEstimator):
    """Imputes using an oracle distribution and then runs the true function f* on the imputed data.

    Parameters
    ----------
    mdm: str
        The missing data mechanism: either 'MCAR', 'MAR' or 'gaussian_sm'.

    mu: array-like, shape (n_features, )
        Mean of the Gaussian distribution.

    Sigma: array-like, shape (n_features, n_features)
        Covariance matrix of the Gaussian distribution.
    """

    def __init__(self, data_params, n_draws=10):
        self.data_params = data_params
        self.n_draws = n_draws

    def oracle_impute(self, X):
        (_, mu, Sigma, _, masking_params, _, _, _) = self.data_params
        
        mdm = masking_params['mdm']

        if mdm not in ['MCAR', 'MAR_logistic', 'MAR_monotone_logistic', 'gaussian_sm']:
            raise ValueError('`mdm` should be one of `MCAR`, `MAR_logistic`, `MAR_monotone_logistic`,or `gaussian_sm`')
        elif mdm == 'gaussian_sm':
            sm_params = masking_params['sm_param']
            tsigma2 = sm_params['sigma2_tilde']
            tmu = sm_params['tmu']
        
        T = X.copy()
        for t in T:
            m = np.isnan(t)
            obs = np.where(~m)[0]
            mis = np.where(m)[0]

            if len(mis) == 0:
                continue

            sigma_obs = Sigma[np.ix_(obs, obs)]
            sigma_obs_inv = np.linalg.inv(sigma_obs)
            sigma_misobs = Sigma[np.ix_(mis, obs)]
            sigma_mis = Sigma[np.ix_(mis, mis)]

            mu_cond = mu[mis] + sigma_misobs.dot(sigma_obs_inv).dot(t[obs] - mu[obs])
            sigma_cond = sigma_mis - sigma_misobs.dot(sigma_obs_inv).dot(sigma_misobs.T) + np.diag(np.repeat(1e-6, repeats=len(mis)))

            if mdm in ['MCAR', 'MAR_logistic', 'MAR_monotone_logistic']:
                    t[mis] = np.random.multivariate_normal(mu_cond, sigma_cond)

            elif mdm == 'gaussian_sm':
                sigma_cond_inv = np.linalg.inv(sigma_cond)

                D_mis_inv = np.diag(1/tsigma2[mis])

                S = np.linalg.inv(D_mis_inv + sigma_cond_inv)
                m = S.dot(D_mis_inv.dot(tmu[mis]) + sigma_cond_inv.dot(mu_cond))

                t[mis] = np.random.multivariate_normal(m, S)

        return T

    def fit(self, X, y, X_val=None, y_val=None):
        return self

    def predict_f_star(self, X):
        (_, _, _, beta, _, _, link, curvature) = self.data_params
        
        pred = []
        for x in X:
            dot_product = beta[0] + beta[1:].dot(x)

            if link == 'linear':
                predx = dot_product
            else:
                if link == 'square':
                    predx = curvature*(dot_product-1)**2
                elif link == 'cube':
                    linear_coef = pow(3*sqrt(3)/2*sqrt(curvature)*4, 2/3)
                    predx = (beta[0] + curvature*dot_product**3 -
                              linear_coef*dot_product)
                elif link == 'stairs':
                    predx = dot_product - 1
                    for a, b in zip([2, -4, 2], [-0.8, -1, -1.2]):
                        predx += a*norm.cdf(
                            sqrt(pi/8)*curvature*(dot_product + b))
                elif link == 'discontinuous_linear':
                    predx = dot_product + (dot_product > 1)*3

            pred.append(predx)

        return np.array(pred)

    def predict(self, X):
        pred = []
        for _ in range(self.n_draws):
            T = self.oracle_impute(X)
            predy = self.predict_f_star(T)
            pred.append(predy)
        return np.mean(np.stack(pred, axis=1), axis=1)
    
