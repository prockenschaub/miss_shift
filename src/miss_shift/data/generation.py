"""
Data generation.
"""
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from scipy.stats import norm
from math import sqrt, log, pi
from scipy.optimize import root_scalar

from .amputation import MCAR, MAR_logistic, MAR_monotone_logistic, MAR_on_y, MNAR_logistic, MNAR_logistic_uniform, gaussian_sm

def _validate_masking_params(masking_params):
    mdm = masking_params['mdm']
        
    missing_rate = masking_params.get('missing_rate', -1)
    if missing_rate > 1 or missing_rate < 0:
        raise ValueError("missing_rate must be >= 0 and <= 1, got %s" % missing_rate)
    
    if mdm in ['MAR_logistic', 'MAR_monotone_logistic']: 
        prop_for_masking = masking_params.get('prop_for_masking', -1)
        if (prop_for_masking > 1 or prop_for_masking < 0):
            raise ValueError("prop_for_masking should be between 0 and 1")
    if 'gaussian_sm' in mdm: 
        if not masking_params.get('sm_type'):
            raise ValueError("sm_type must be specified for self-masking")
        if not masking_params.get('sm_param'):
            raise ValueError("sm_param must be specified for self-masking")
        


def gen_params(n_features, prop_latent, masking_params, snr, 
               link='linear',data_path=None, curvature=1, seed_data=None, seed_ampute=None):
    _validate_masking_params(masking_params)
    
    if prop_latent > 1 or prop_latent < 0:
        raise ValueError("prop_latent should be between 0 and 1")

    rng_data = check_random_state(seed_data)
    rng_ampute = check_random_state(seed_ampute)

    # Generate covariance and mean
    # ---------------------------
    B = rng_data.randn(n_features, int(prop_latent*n_features))
    cov = B.dot(B.T) + np.diag(
        rng_data.uniform(low=0.01, high=0.1, size=n_features))

    mean = np.zeros(n_features) + rng_data.randn(n_features)

    # For self-masking, adapt the remaining parameters to obtain the
    # desired missing rate
    # ---------------------
    if masking_params['mdm'] == 'gaussian_sm':
        sm_params = {}
        missing_rate = masking_params['missing_rate']

        if masking_params['sm_type'] == 'probit':
            lam = masking_params['sm_param']
            sm_params['lambda'] = lam
            sm_params['c'] = np.zeros(n_features)
            for i in range(n_features):
                sm_params['c'][i] = lam*(mean[i] - norm.ppf(missing_rate)*np.sqrt(
                    1/lam**2+cov[i, i]))

        elif masking_params['sm_type'] == 'gaussian':
            k = masking_params['sm_param']
            sm_params['k'] = k
            sm_params['sigma2_tilde'] = np.zeros(n_features)

            min_x = missing_rate**2/(1-missing_rate**2)

            def f(x):
                y = -2*(1+x)*log(missing_rate*sqrt(1/x+1))
                return y

            for i in range(n_features):
                max_x = min_x
                while f(max_x) < k**2:
                    max_x += 1
                sol = root_scalar(lambda x: f(x) - k**2, method='bisect',
                                bracket=(max_x-1, max_x), xtol=1e-3)

                sm_params['sigma2_tilde'][i] = sol.root*cov[i, i]
            
            sm_params['tmu'] = mean + k*np.sqrt(np.diag(cov))
        
            masking_params['sm_param'] = sm_params

            if masking_params.get('perm', False):
                masking_params['perm'] = rng_ampute.permutation(n_features)

    # Generate beta
    beta = np.repeat(1., n_features + 1)
    var = beta[1:].dot(cov).dot(beta[1:])
    beta[1:] *= 1/sqrt(var)

    return (n_features, mean, cov, beta, masking_params, snr, link, data_path, curvature)

def gen_y(X, snr, link, curvature, beta, n_samples, seed_data=None):
    rng_data = check_random_state(seed_data)

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

    var_y = np.mean((y - np.mean(y))**2)
    sigma2_noise = var_y/snr

    noise = rng_data.normal(
        loc=0, scale=sqrt(sigma2_noise), size=n_samples)
    y += noise
    return y

def gen_mask(X,mean, cov, masking_params, seed_ampute=None):
    rng_ampute = check_random_state(seed_ampute)

    masking = masking_params['mdm']
    missing_rate = masking_params['missing_rate']

    if masking == 'MCAR':
        M = MCAR(X, missing_rate, rng_ampute)
    elif masking == 'MAR_logistic':
        prop_for_masking = masking_params['prop_for_masking']
        sample_vars = masking_params.get('sample_vars', False)
        M = MAR_logistic(X, missing_rate, prop_for_masking,
                                rng_ampute, sample_vars=sample_vars)
    elif masking == 'MAR_monotone_logistic':
        prop_for_masking = masking_params['prop_for_masking']
        sample_vars = masking_params.get('sample_vars', False)
        M = MAR_monotone_logistic(X, missing_rate, prop_for_masking,
                                rng_ampute, sample_vars=sample_vars)
    elif masking == 'MNAR_logistic':
        M = MNAR_logistic(X, missing_rate, rng_ampute)
    elif masking == 'MNAR_logistic_uniform':
        M = MNAR_logistic_uniform(X, missing_rate,
                                        prop_for_masking, rng_ampute)
    elif masking == 'gaussian_sm':
        sm_type = masking_params['sm_type']
        sm_param = masking_params['sm_param']
        perm = masking_params.get('perm', False)
        M = gaussian_sm(X, sm_type, sm_param, mean, cov, rng_ampute)
        if perm:
            M = M[:, perm]

    Xm = np.copy(X)
    np.putmask(Xm, M, np.nan)
    return Xm

def gen_data(n_sizes, data_params, seed_data=None, seed_ampute=None):

    rng_data = check_random_state(seed_data)

    (n_features, mean, cov, beta, masking_params, snr, link, data_path, curvature) = data_params
    X = np.empty((0, n_features))
    Xm = np.empty((0, n_features))
    y = np.empty((0, ))
    
    current_size = 0

    for _, n_samples in enumerate(n_sizes):
        if data_path=='None':
            current_X = rng_data.multivariate_normal(
                    mean=mean, cov=cov,
                    size=n_samples-current_size,
                    check_valid='raise')
        else:
            real_data=pd.read_csv(data_path)
            current_X = real_data.values.astype(float)
            
            if n_samples > current_X.shape[0]:
                raise ValueError
        current_y=gen_y(current_X, snr, link, curvature, beta, n_samples, current_size,seed_data)
        current_Xm=gen_mask(current_X,mean, cov, masking_params, seed_ampute)

        X = np.vstack((X, current_X))
        Xm = np.vstack((Xm, current_Xm))
        y = np.hstack((y, current_y))

        current_size = n_samples

        yield X, Xm, y
    
    
