# Minimally adapted from Le Morvan et al. (file ground_truth.py)
from sklearn.base import BaseEstimator
import numpy as np
from scipy.stats import norm
from math import sqrt, pi

class BayesPredictor_GSM_nonlinear(BaseEstimator):
    """This is the Bayes predicor for Gaussian data with a Gaussian
    selfmasking missing data mechanism"""

    def __init__(self, data_params, order0=False):

        _, _, _, _, masking_params, _, _, _ = data_params

        if masking_params['sm_type'] == 'probit':
            raise ValueError('This Bayes predictor is only valid for' +
                             'Gaussian selfmasking and not probit selfmasking')

        if masking_params['perm']:
            raise ValueError('The Bayes predictor is not available for' +
                             'perm = True')

        self.data_params = data_params
        self.order0 = order0

    def fit(self, X, y):
        return self

    def predict(self, X):

        _, mu, cov, beta, masking_params, _, link, curvature = self.data_params

        sm_params = masking_params['sm_param']
        tsigma2 = sm_params['sigma2_tilde']
        tmu = sm_params['tmu']

        pred = []
        for x in X:
            mis = np.where(np.isnan(x))[0]
            obs = np.where(~np.isnan(x))[0]

            D_mis_inv = np.diag(1/tsigma2[mis])

            cov_misobs = cov[np.ix_(mis, obs)]
            cov_obs_inv = np.linalg.inv(cov[np.ix_(obs, obs)])
            cov_mis = cov[np.ix_(mis, mis)]

            mu_mis_obs = mu[mis] + cov_misobs.dot(cov_obs_inv).dot(
                x[obs] - mu[obs])
            cov_mis_obs = cov_mis - cov_misobs.dot(cov_obs_inv).dot(
                cov_misobs.T)
            cov_mis_obs_inv = np.linalg.inv(cov_mis_obs)

            S = np.linalg.inv(D_mis_inv + cov_mis_obs_inv)
            s = S.dot(D_mis_inv.dot(tmu[mis]) +
                      cov_mis_obs_inv.dot(mu_mis_obs))

            dot_product = beta[0] + beta[obs + 1].dot(x[obs]) + \
                beta[mis + 1].dot(s)

            if link == 'linear':
                predx0 = dot_product
                predx = predx0
            else:
                var_Tmis = beta[mis + 1].dot(S).dot(beta[mis + 1])
                if link == 'square':
                    predx0 = curvature*(dot_product-1)**2
                    predx = predx0 + curvature*var_Tmis
                elif link == 'cube':
                    predx0 = (beta[0] + curvature*dot_product**3 -
                              3*dot_product)
                    predx = predx0 + 3*curvature*var_Tmis*dot_product
                elif link == 'stairs':
                    predx0 = dot_product - 1
                    predx = dot_product - 1
                    for a, b in zip([2, -4, 2], [-0.8, -1, -1.2]):
                        predx += a*norm.cdf((dot_product + b)/sqrt(
                            1/(pi/8*curvature**2) + var_Tmis))
                        predx0 += a*norm.cdf(
                            sqrt(pi/8)*curvature*(dot_product + b))
                elif link == 'discontinuous_linear':
                    predx0 = dot_product + (dot_product > 1)*3
                    predx = dot_product + 3*(1-norm.cdf(
                        1, loc=dot_product, scale=sqrt(var_Tmis)))

            if self.order0:
                pred.append(predx0)
            else:
                pred.append(predx)

        return np.array(pred)
    



class BayesPredictor_MCAR_MAR_nonlinear(BaseEstimator):
    """This is the Bayes predictor for multivariate Gaussian data, MCAR or
    MAR missing data mechanisms."""

    def __init__(self, data_params, order0=False):
        self.data_params = data_params
        self.order0 = order0

    def fit(self, X, y):
        return self

    def predict(self, X):
        (_, mu, sigma, beta, _, _, link, curvature) = self.data_params

        pred = []
        for x in X:
            m = ''.join([str(mj) for mj in np.isnan(x).astype(int)])

            obs = np.where(np.array(list(m)).astype(int) == 0)[0]
            mis = np.where(np.array(list(m)).astype(int) == 1)[0]

            dot_product = beta[0]
            if len(mis) > 0:
                dot_product += beta[mis + 1].dot(mu[mis])
            if len(obs) > 0:
                dot_product += beta[obs + 1].dot(x[obs])
            if len(obs) * len(mis) > 0:
                sigma_obs = sigma[np.ix_(obs, obs)]
                sigma_obs_inv = np.linalg.inv(sigma_obs)
                sigma_misobs = sigma[np.ix_(mis, obs)]

                dot_product += beta[mis + 1].dot(sigma_misobs).dot(
                    sigma_obs_inv).dot(x[obs] - mu[obs])

            if link == 'linear':
                predx0 = dot_product
                predx = predx0
            else:
                if len(mis)*len(obs) > 0:
                    sigma_mismis = sigma[np.ix_(mis, mis)]
                    sigma_mis_obs = sigma_mismis - sigma_misobs.dot(
                        sigma_obs_inv).dot(sigma_misobs.T)
                    var_Tmis = beta[mis + 1].dot(
                        sigma_mis_obs).dot(beta[mis + 1])
                elif len(obs) > 0:
                    var_Tmis = 0
                elif len(mis) > 0:
                    sigma_mismis = sigma[np.ix_(mis, mis)]
                    var_Tmis = beta[mis + 1].dot(sigma_mismis).dot(
                        beta[mis + 1])
                if link == 'square':
                    predx0 = curvature*(dot_product-1)**2
                    predx = predx0 + curvature*var_Tmis
                elif link == 'cube':
                    linear_coef = pow(3*sqrt(3)/2*sqrt(curvature)*4, 2/3)
                    predx0 = (beta[0] + curvature*dot_product**3 -
                              linear_coef*dot_product)
                    predx = predx0 + 3*curvature*var_Tmis*dot_product
                elif link == 'stairs':
                    predx0 = dot_product - 1
                    predx = dot_product - 1
                    for a, b in zip([2, -4, 2], [-0.8, -1, -1.2]):
                        predx += a*norm.cdf((dot_product + b)/sqrt(
                            1/(pi/8*curvature**2) + var_Tmis))
                        predx0 += a*norm.cdf(
                            sqrt(pi/8)*curvature*(dot_product + b))
                elif link == 'discontinuous_linear':
                    predx0 = dot_product + (dot_product > 1)*3
                    predx = dot_product + 3*(1-norm.cdf(
                        1, loc=dot_product, scale=sqrt(var_Tmis)))

            if self.order0:
                pred.append(predx0)
            else:
                pred.append(predx)

        return np.array(pred)



class BayesPredictor_MAR_y(BaseEstimator):
    """This is the Bayes predictor for multivariate Gaussian data, MCAR or
    MAR missing data mechanisms."""

    def __init__(self, data_params, order0=False):
        self.data_params = data_params
        self.order0 = order0

    def fit(self, X, y):
        return self

    def predict(self, X):
        (_, mu, sigma, beta, _, _, link, curvature) = self.data_params

        pred = []
        for x in X:
            m = ''.join([str(mj) for mj in np.isnan(x).astype(int)])

            obs = np.where(np.array(list(m)).astype(int) == 0)[0]
            mis = np.where(np.array(list(m)).astype(int) == 1)[0]

            dot_product = beta[0]
            if len(mis) > 0:
                dot_product += beta[mis + 1].dot(mu[mis])
            if len(obs) > 0:
                dot_product += beta[obs + 1].dot(x[obs])
            if len(obs) * len(mis) > 0:
                sigma_obs = sigma[np.ix_(obs, obs)]
                sigma_obs_inv = np.linalg.inv(sigma_obs)
                sigma_misobs = sigma[np.ix_(mis, obs)]

                dot_product += beta[mis + 1].dot(sigma_misobs).dot(
                    sigma_obs_inv).dot(x[obs] - mu[obs])

            if link == 'linear':
                predx0 = dot_product
                predx = predx0
            else:
                if len(mis)*len(obs) > 0:
                    sigma_mismis = sigma[np.ix_(mis, mis)]
                    sigma_mis_obs = sigma_mismis - sigma_misobs.dot(
                        sigma_obs_inv).dot(sigma_misobs.T)
                    var_Tmis = beta[mis + 1].dot(
                        sigma_mis_obs).dot(beta[mis + 1])
                elif len(obs) > 0:
                    var_Tmis = 0
                elif len(mis) > 0:
                    sigma_mismis = sigma[np.ix_(mis, mis)]
                    var_Tmis = beta[mis + 1].dot(sigma_mismis).dot(
                        beta[mis + 1])
                if link == 'square':
                    predx0 = curvature*(dot_product-1)**2
                    predx = predx0 + curvature*var_Tmis
                elif link == 'cube':
                    linear_coef = pow(3*sqrt(3)/2*sqrt(curvature)*4, 2/3)
                    predx0 = (beta[0] + curvature*dot_product**3 -
                              linear_coef*dot_product)
                    predx = predx0 + 3*curvature*var_Tmis*dot_product
                elif link == 'stairs':
                    predx0 = dot_product - 1
                    predx = dot_product - 1
                    for a, b in zip([2, -4, 2], [-0.8, -1, -1.2]):
                        predx += a*norm.cdf((dot_product + b)/sqrt(
                            1/(pi/8*curvature**2) + var_Tmis))
                        predx0 += a*norm.cdf(
                            sqrt(pi/8)*curvature*(dot_product + b))
                elif link == 'discontinuous_linear':
                    predx0 = dot_product + (dot_product > 1)*3
                    predx = dot_product + 3*(1-norm.cdf(
                        1, loc=dot_product, scale=sqrt(var_Tmis)))

            if self.order0:
                pred.append(predx0)
            else:
                pred.append(predx)

        return np.array(pred)