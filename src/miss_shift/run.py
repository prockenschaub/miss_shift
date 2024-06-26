from copy import deepcopy
import os
from typing import List

import torch
from sklearn.ensemble import HistGradientBoostingRegressor

from .data.generation import gen_params, gen_data
from .estimators.neumiss import Neumiss
from .estimators.neumise import NeuMISE
from .estimators.conditional_impute import ImputeMLP
from .estimators.miwae import MIWAEMLP
from .oracles.conditional import (
    BayesPredictor_GSM_nonlinear,
    BayesPredictor_MCAR_MAR_nonlinear,
)
from .oracles.probabilistic import ProbabilisticBayesPredictor


def run_one(
    data_desc: dict,
    method: str,
    method_params: dict,
    it: int,
    n_train: List[int],
    n_test: int,
    n_val: int,
    mdm: str,
    tmp_dir: str = "tmp",
) -> List[dict]:
    """Run a single experiment, i.e., a single estimator for a given set of data and hyperparameters.

    Note: a single run may include multiple data sizes.

    Args:
        data_desc: parameters describing the data generation.
        method: method/estimator to use.
        method_params: hyperparameters of the method.
        it: number of the current trial.
        n_train: list of one or more training sample sizes in increasing order.
        n_test: number of training samples.
        n_val: number of validation samples.
        mdm: missing data mechanism. One of "MCAR", "MAR_monotone_logistic", "MAR_on_y", or "gaussian_sm".
        tmp_dir: path to a directory used to store temporary data. Defaults to 'tmp'.

    Returns:
        performance summary including R2 and MSE
    """
    if not isinstance(n_train, list):
        n_train = [n_train]
    n_tot = [n_train + n_test + n_val for n_train in n_train]

    # Generate the data
    orig_desc = deepcopy(data_desc)
    orig_desc["masking_params"] = orig_desc.pop("miss_orig")
    orig_desc.pop("miss_shift")
    orig_params = gen_params(
        **orig_desc,
        seed_data=it,
        seed_ampute=orig_desc["masking_params"].get("seed", it),
    )
    gen_orig = gen_data(
        n_tot,
        orig_params,
        seed_data=it,
        seed_ampute=orig_desc["masking_params"].get("seed", it),
    )

    shift_desc = deepcopy(data_desc)
    shift_desc["masking_params"] = shift_desc.pop("miss_shift")
    shift_desc.pop("miss_orig")
    shift_params = gen_params(
        **shift_desc,
        seed_data=it,
        seed_ampute=shift_desc["masking_params"].get("seed", it * 42),
    )  # Change the ampute seed for the shift
    gen_shift = gen_data(
        n_tot,
        shift_params,
        seed_data=it,
        seed_ampute=shift_desc["masking_params"].get("seed", it * 42),
    )  # Change the ampute seed for the shift

    # Get method name and initialize estimator
    if method == "bayes":
        if mdm == "gaussian_sm":
            est = BayesPredictor_GSM_nonlinear
        elif mdm in ["MCAR", "MAR_logistic", "MAR_monotone_logistic"]:
            est = BayesPredictor_MCAR_MAR_nonlinear
        else:
            raise ValueError(
                f"No conditional Bayes predictor has been implemented for {mdm}."
            )
    elif method == "prob_bayes":
        if mdm in ["MCAR", "MAR_logistic", "MAR_monotone_logistic", "gaussian_sm"]:
            est = ProbabilisticBayesPredictor
        else:
            raise ValueError(
                f"No probabilistic Bayes predictor has been implemented for {mdm}."
            )
    elif "neumiss" in method:
        est = Neumiss
    elif "neumise" in method:
        est = NeuMISE
    elif "miwae" in method:
        est = MIWAEMLP
    elif (
        ("mean_impute" in method)
        or ("ice_impute" in method)
        or ("mice_impute" in method)
    ):
        est = ImputeMLP
    elif "gbrt" in method:
        est = HistGradientBoostingRegressor
    else:
        raise ValueError("{} is not a known method".format(method))

    # A list of dictionaries that give the MSE an R2 for each n and train, test
    # and validation sets.
    results = []

    # Loop over the different dataset sizes
    for (X, Xm, y), (_, Xs, ys) in zip(gen_orig, gen_shift):
        n, p = X.shape
        n = n - n_val - n_test
        n_val_half = n_val // 2

        # test data
        X_test = X[0:n_test]  # fully observed
        Xm_test = Xm[0:n_test]  # partially observed
        y_test = y[0:n_test]
        # validation data for choosing the best model
        Xm_val = Xm[n_test : (n_test + n_val_half)]
        y_val = y[n_test : (n_test + n_val_half)]
        # validation data for earlystopping
        Xm_val_es = Xm[(n_test + n_val_half) : (n_test + n_val)]
        y_val_es = y[(n_test + n_val_half) : (n_test + n_val)]
        # train data
        Xm_train = Xm[(n_test + n_val) :]
        y_train = y[(n_test + n_val) :]

        # Set the torch seed
        torch.manual_seed(0)
        if method in ["gbrt"]:
            # For these methods the validation data for early stopping should
            # be given as a fraction of the training set.
            method_params["validation_fraction"] = n_val_half / (n + n_val_half)
            X_train_val_es = Xm[(n_test + n_val_half) :]
            y_train_val_es = y[(n_test + n_val_half) :]
            reg = est(**method_params)
            reg.fit(X_train_val_es, y_train_val_es)
        elif method in ["bayes", "prob_bayes"]:
            reg = est(orig_params, **method_params)
            reg.fit(Xm_train, y_train)
        elif method in ["oracle_impute"]:
            reg = est(orig_params, **method_params)
            reg.fit(Xm_train, y_train, X_val=Xm_val_es, y_val=y_val_es)
        elif "miwae" in method:
            save_dir = os.path.join(
                tmp_dir,
                mdm,
                "miwae",
                f"n={n}",
                f'prop_latent={data_desc["prop_latent"]}',
                f"iter={it}",
            )
            os.makedirs(save_dir, exist_ok=True)

            reg = est(**method_params, save_dir=save_dir)
            reg.fit(Xm_train, y_train, X_val=Xm_val_es, y_val=y_val_es)

            if method_params["mode"] == "imputer-only":
                continue
        else:
            # For these methods the validatin data for early stopping should be
            # given as standalone data.
            reg = est(**method_params)
            reg.fit(Xm_train, y_train, X_val=Xm_val_es, y_val=y_val_es)

        pred_train = reg.predict(Xm_train)
        pred_val = reg.predict(Xm_val)
        pred_test = reg.predict(X_test)
        pred_test_m = reg.predict(Xm_test)

        mse_train = ((y_train - pred_train) ** 2).mean()
        mse_val = ((y_val - pred_val) ** 2).mean()
        mse_test = ((y_test - pred_test) ** 2).mean()
        mse_test_m = ((y_test - pred_test_m) ** 2).mean()

        var_train = ((y_train - y_train.mean()) ** 2).mean()
        var_val = ((y_val - y_val.mean()) ** 2).mean()
        var_test = ((y_test - y_test.mean()) ** 2).mean()

        r2_train = 1 - mse_train / var_train
        r2_val = 1 - mse_val / var_val
        r2_test = 1 - mse_test / var_test
        r2_test_m = 1 - mse_test_m / var_test

        # Evaluate the shifted data
        Xs_test = Xs[0:n_test]  # fully observed
        ys_test = ys[0:n_test]
        pred_test_s = reg.predict(Xs_test)
        mse_test_s = ((ys_test - pred_test_s) ** 2).mean()
        var_test_s = ((ys_test - ys_test.mean()) ** 2).mean()
        r2_test_s = 1 - mse_test_s / var_test_s

        res = {
            "iter": it,
            "n": n,
            "R2_train": r2_train,
            "R2_val": r2_val,
            "R2_test": r2_test,
            "R2_test_m": r2_test_m,
            "R2_test_s": r2_test_s,
            "mse_train": mse_train,
            "mse_val": mse_val,
            "mse_test": mse_test,
            "mse_test_m": mse_test_m,
            "mse_test_s": mse_test_s,
        }

        results.append(res)

    return results
