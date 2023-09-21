"""This file implements amputation procedures according to various missing
data mechanisms. It was inspired from
https://github.com/BorisMuzellec/MissingDataOT/blob/master/utils.py
"""

import numpy as np
from sklearn.utils import check_random_state
from scipy.optimize import fsolve
from scipy.stats import norm
from math import sqrt


def sigmoid(x):
    """Calculate the sigmoid"""
    return 1 / (1 + np.exp(-x))


def MCAR(
    X: np.ndarray, p: float, random_state: None | int | np.random.RandomState = None
) -> np.ndarray:
    """
    Simulate missingness according to a missing completely at random (MCAR) mechanism.

    Args:
        X: data for which missing values will be simulated.
        p: proportion of missing values to generate for variables which will have
        missing values.
        random_state:
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.

    Returns:
        mask: boolean mask of generated missing values (True if the value is missing).
    """

    rng = check_random_state(random_state)

    n, d = X.shape
    mask = np.zeros((n, d))

    ber = rng.rand(n, d)
    mask = ber < p

    return mask


def MAR_logistic(
    X: np.ndarray,
    p: float,
    p_obs: float,
    random_state: None | int | np.random.RandomState = None,
    sample_vars: bool = True,
) -> np.ndarray:
    """
    Simulate missingness according to a missing at random mechanism with a logistic masking model.
    First, a subset of variables with *no* missing values is randomly selected. The remaining
    variables have missing values according to a logistic model with random weights, re-scaled so
    as to attain the desired proportion of missing values on those variables.

    Args:
        X: data for which missing values will be simulated.
        p: proportion of missing values to generate for variables which will have
            missing values.
        p_obs: proportion of variables with *no* missing values that will be used for
            the logistic masking model.
        random_state:
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.

    Returns:
        mask: boolean mask of generated missing values (True if the value is missing).
    """
    rng = check_random_state(random_state)

    n, d = X.shape
    mask = np.zeros((n, d))

    # number of variables that will have no missing values
    # (at least one variable)
    d_obs = max(int(p_obs * d), 1)
    # number of variables that will have missing values
    d_na = d - d_obs

    # Sample variables that will all be observed, and those with missing values
    if sample_vars:
        idxs_obs = rng.choice(d, d_obs, replace=False)
        idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])
    else:
        idxs_obs = np.arange(d_obs)
        idxs_nas = np.arange(d_obs, d)

    # Other variables will have NA proportions that depend on those observed
    # variables, through a logistic model. The parameters of this logistic
    # model are random, and adapted to the scale of each variable.
    mu = X.mean(0)
    cov = (X - mu).T.dot(X - mu) / n
    cov_obs = cov[np.ix_(idxs_obs, idxs_obs)]
    coeffs = rng.randn(d_obs, d_na)
    v = np.array([coeffs[:, j].dot(cov_obs).dot(coeffs[:, j]) for j in range(d_na)])
    steepness = rng.uniform(low=0.1, high=0.5, size=d_na)
    coeffs /= steepness * np.sqrt(v)

    # Move the intercept to have the desired amount of missing values
    intercepts = np.zeros((d_na))
    for j in range(d_na):
        w = coeffs[:, j]

        def f(b):
            s = sigmoid(X[:, idxs_obs].dot(w) + b) - p
            return s.mean()

        res = fsolve(f, x0=0)
        intercepts[j] = res[0]

    ps = sigmoid(X[:, idxs_obs].dot(coeffs) + intercepts)
    ber = rng.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    return mask


def MAR_monotone_logistic(
    X: np.ndarray,
    p: float,
    p_obs: float,
    random_state: None | int | np.random.RandomState = None,
    sample_vars: bool = True,
) -> np.ndarray:
    """
    Simulate missingness according to a monotone missing at random mechanism with a logistic masking model.
    First, a subset of variables with *no* missing values is masked via MCAR. The remaining
    variables have missing values according to a logistic model with random weights, re-scaled so
    as to attain the desired proportion of missing values on those variables.

    Args:
        X: data for which missing values will be simulated.
        p: proportion of missing values to generate for variables which will have
            missing values.
        p_obs: proportion of variables with *no* missing values that will be used for
            the logistic masking model.
        random_state:
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.

    Returns:
        mask: boolean mask of generated missing values (True if the value is missing).
    """

    rng = check_random_state(random_state)

    n, d = X.shape
    mask = np.zeros((n, d))

    # number of variables that will have MCAR missing values
    # (at least one variable)
    d_obs = max(int(p_obs * d), 1)
    # number of variables that will have monotone MAR missing values
    d_na = d - d_obs

    # Sample variables that will have MCAR and MAR values
    if sample_vars:
        idxs_obs = rng.choice(d, d_obs, replace=False)
        idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])
    else:
        idxs_obs = np.arange(d_obs)
        idxs_nas = np.arange(d_obs, d)

    # Sample the MCAR
    mask[:, idxs_obs] = MCAR(X[:, idxs_obs], p, rng)

    # Other variables will have NA proportions that depend on those remaining observed
    # variables, through a logistic model. The parameters of this logistic
    # model are random, and adapted to the scale of each variable.
    mu = X.mean(0)
    cov = (X - mu).T.dot(X - mu) / n
    cov_obs = cov[np.ix_(idxs_obs, idxs_obs)]
    coeffs = rng.randn(d_obs, d_na)
    v = np.array([coeffs[:, j].dot(cov_obs).dot(coeffs[:, j]) for j in range(d_na)])
    steepness = rng.uniform(low=0.1, high=0.5, size=d_na)
    coeffs /= steepness * np.sqrt(v)

    # Move the intercept to have the desired amount of missing values
    Xpartial = X.copy()
    np.putmask(Xpartial, mask, 0)
    intercepts = np.zeros((d_na))
    for j in range(d_na):
        w = coeffs[:, j]

        def f(b):
            s = sigmoid(Xpartial[:, idxs_obs].dot(w) + b) - p
            return s.mean()

        res = fsolve(f, x0=0)
        intercepts[j] = res[0]

    ps = sigmoid(Xpartial[:, idxs_obs].dot(coeffs) + intercepts)
    ber = rng.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    return mask


def MAR_on_y(
    X: np.ndarray,
    y: np.ndarray,
    p: float,
    random_state: None | int | np.random.RandomState = None,
) -> np.ndarray:
    """
    Simulate missingness according to a missing at random mechanism depending on y through a logistic masking model.

    Args:
        X: data for which missing values will be simulated.
        p: proportion of missing values to generate for variables which will have
            missing values.
        p_obs: proportion of variables with *no* missing values that will be used for
            the logistic masking model.
        random_state:
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.

    Returns:
        mask: boolean mask of generated missing values (True if the value is missing).
    """
    rng = check_random_state(random_state)

    n, d = X.shape
    mask = np.zeros((n, d))

    mu = y.mean(0)
    var = (y - mu).T.dot(y - mu) / n
    coeffs = rng.randn(d)
    v = np.array([coeffs[j] ** 2 * var for j in range(d)])
    coeffs /= np.sqrt(v)

    # Move the intercept to have the desired amount of missing values
    intercepts = np.zeros((d))
    for j in range(d):
        w = coeffs[j]

        def f(b):
            s = sigmoid(y * w + b) - p
            return s.mean()

        res = fsolve(f, x0=0)
        intercepts[j] = res[0]

    ps = sigmoid(y[:, np.newaxis].dot(coeffs[np.newaxis]) + intercepts)
    ber = rng.rand(n, d)
    mask = ber < ps

    return mask


def gaussian_sm(
    X: np.ndarray,
    sm_type: str,
    sm_params: dict,
    mean: np.ndarray,
    cov: np.ndarray,
    random_state: None | int | np.random.RandomState = None,
) -> np.ndarray:
    """
    Simulate missingness according to a Gaussian self-masking model.

    Args:
        X: data for which missing values will be simulated.
        sm_type: type of selfmasking function used. One of `gaussian` or `probit`.
        sm_param: parameter for the selfmasking function.
            If `sm_type == 'gaussian'`, then `sm_param` is the parameter called
            `k` in the paper that controls the mean of the Gaussian selfmasking
            function as well as `tmu` and `sigma2_tilde`, which can be derived
            from `k`. See also `gen_params`.
            If `sm_type == 'probit'`, then `sm_param`is the parameter called
            lambda`in the paper that controls the slope of the probit selfmasking
            function, as well as `c`, which can be derived from `k`.
            See also https://github.com/marineLM/Impute_then_Regress/blob/master/python/ground_truth.py
        mean: means of the data for which missing values will be simulated
        cov: covariance of the data for which missing values will be simulated
        random_state:
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.

    Returns:
        mask: boolean mask of generated missing values (True if the value is missing).
    """
    n, d = X.shape
    mask = np.zeros((n, d))

    rng = check_random_state(random_state)

    for j in range(d):
        X_j = X[:, j]
        if sm_type == "probit":
            lam = sm_params["lambda"]
            c = sm_params["c"][j]
            prob = norm.cdf(lam * X_j - c)
        elif sm_type == "gaussian":
            k = sm_params["k"]
            sigma2_tilde = sm_params["sigma2_tilde"][j]
            mu_tilde = mean[j] + k * sqrt(cov[j, j])
            prob = np.exp(-0.5 * (X_j - mu_tilde) ** 2 / sigma2_tilde)

        mask[:, j] = rng.binomial(n=1, p=prob, size=len(X_j))

    return mask
