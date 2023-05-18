import os
from copy import deepcopy
import numpy as np
import pandas as pd
from collections import namedtuple
from joblib import Parallel, delayed
from data.generation import gen_params, gen_data

from estimators.neumiss import Neumiss
from estimators.neumice import NeuMICE
from estimators.conditional_impute import ImputeMLPPytorch
from estimators.oracle_impute import OracleImputeMLPPytorch
from sklearn.ensemble import HistGradientBoostingRegressor
from oracles.conditional import BayesPredictor_GSM_nonlinear, BayesPredictor_MCAR_MAR_nonlinear
from oracles.probabilistic import ProbabilisticBayesPredictor

from tqdm import tqdm
import torch


# Result item to create the DataFrame in a consistent way.
fields = ['iter', 'method', 'n', 'mse_train', 'mse_val', 'mse_test', 'mse_test_m', 'mse_test_s',
          'R2_train', 'R2_val', 'R2_test', 'R2_test_m', 'R2_test_s', 
          'early_stopping', 'optimizer', 'depth',
          'n_epochs', 'learning_rate', 'lr', 'weight_decay', 'batch_size',
          'type_width', 'width', 'n_draws', 'n_iter_no_change',
          'verbose', 'mlp_depth', 'init_type', 'max_iter', 'order0',
          'n_trials_no_change', 'add_mask', 'imputation_type', 
          'n_features', 'prop_latent', 'snr', 'miss_orig', 'miss_shift',
          'link', 'curvature', 'width_factor', 'max_leaf_nodes', 'min_samples_leaf']

ResultItem = namedtuple('ResultItem', fields)
ResultItem.__new__.__defaults__ = (np.nan, )*len(ResultItem._fields)


def run(n_trials, n_train, n_val, n_test, mdm, data_descs, methods_params,
        out_dir, n_jobs=1):

    for nm, scope in methods_params.items():
        runs = []
        for params in scope:
            for data_desc in data_descs.itertuples(index=False):
                data_desc = dict(data_desc._asdict())
                for it in range(n_trials):
                    runs.append([data_desc, nm, params, it])

        # results = Parallel(n_jobs=n_jobs)(
        #     delayed(run_one)(data_desc, method, method_params, it, n_train,
        #                     n_test, n_val, mdm)
        #     for data_desc, method, method_params, it in tqdm(runs)
        # )
        results = []
        for data_desc, method, method_params, it in tqdm(runs):
            results.append(run_one(data_desc, method, method_params, it, n_train,
                            n_test, n_val, mdm))

        # combined_results is a list of all result items that combine the obtained
        # performances and the corresponding data and method parameters.
        # Note that results has the same size as store_params (correspondance)
        combined_results = []
        for i in range(len(runs)):
            data_desc, method, method_params, _ = runs[i]
            result = results[i]
            for result_n in result:
                result_item = ResultItem(
                    method=method, **result_n, **data_desc, **method_params)
                combined_results.append(result_item)

        combined_results = pd.DataFrame(combined_results)
        combined_results.to_csv(os.path.join(out_dir, '{}.csv'.format(nm)), index=False)

def run_one(data_desc, method, method_params, it, n_train, n_test, n_val, mdm):

    if not isinstance(n_train, list):
        n_train = [n_train]
    n_tot = [n_train + n_test + n_val for n_train in n_train]

    # Generate the data
    orig_desc = deepcopy(data_desc)
    orig_desc['masking_params'] = orig_desc.pop('miss_orig')
    orig_desc.pop('miss_shift')
    orig_params = gen_params(**orig_desc, random_state=it)
    gen_orig = gen_data(n_tot, orig_params, random_state=it)

    shift_desc = deepcopy(data_desc)
    shift_desc['masking_params'] = shift_desc.pop('miss_shift')
    shift_desc.pop('miss_orig')
    shift_params = gen_params(**shift_desc, random_state=it)
    gen_shift = gen_data(n_tot, shift_params, random_state=it*42)

    # Get method name and initialize estimator
    if method == 'bayes':
        if mdm == 'gaussian_sm':
            est = BayesPredictor_GSM_nonlinear
        elif mdm in ['MCAR', 'MAR_logistic']:
            est = BayesPredictor_MCAR_MAR_nonlinear
    elif method == "prob_bayes":
        est = ProbabilisticBayesPredictor
    elif 'neumiss' in method:
        est = Neumiss
    elif 'neumice' in method:
        est = NeuMICE
    elif 'oracle_impute' in method:
        est = OracleImputeMLPPytorch
    elif ('mean_impute' in method) or ('mice_impute' in method)  or ('multimice_impute' in method):
        est = ImputeMLPPytorch
    elif 'gbrt' in method:
        est = HistGradientBoostingRegressor
    else:
        raise ValueError('{} is not a known method'.format(method))

    # A list of dictionaries that give the MSE an R2 for each n and train, test
    # and validation sets.
    results = []

    # Loop over the different dataset sizes
    for (X, Xm, y), (_, Xs, ys) in zip(gen_orig, gen_shift):
        n, p = X.shape
        n = n - n_val - n_test
        n_val_half = n_val//2

        print('method: {}, dim: {}, it: {}'.format(method, (n, p), it))

        # test data
        X_test = X[0:n_test] # fully observed
        Xm_test = Xm[0:n_test] # partially observed
        y_test = y[0:n_test]
        # validation data for choosing the best model
        Xm_val = Xm[n_test:(n_test + n_val_half)]
        y_val = y[n_test:(n_test + n_val_half)]
        # validation data for earlystopping
        Xm_val_es = Xm[(n_test + n_val_half):(n_test + n_val)]
        y_val_es = y[(n_test + n_val_half):(n_test + n_val)]
        # train data
        Xm_train = Xm[(n_test + n_val):]
        y_train = y[(n_test + n_val):]

        # Set the torch seed
        torch.manual_seed(0)
        if method in ['gbrt']:
            # For these methods the validation data for early stopping should
            # be given as a fraction of the training set.
            method_params['validation_fraction'] = (
                n_val_half/(n + n_val_half))
            X_train_val_es = Xm[(n_test + n_val_half):]
            y_train_val_es = y[(n_test + n_val_half):]
            reg = est(**method_params)
            reg.fit(X_train_val_es, y_train_val_es)
        elif method in ['bayes', 'prob_bayes']:
            reg = est(orig_params, **method_params)
            reg.fit(Xm_train, y_train)
        elif method in ['oracle_impute']:
            reg = est(orig_params, **method_params)
            reg.fit(Xm_train, y_train, X_val=Xm_val_es, y_val=y_val_es)
        else:
            # For these methods the validatin data for early stopping should be
            # given as standalone data.
            reg = est(**method_params)
            reg.fit(Xm_train, y_train, X_val=Xm_val_es, y_val=y_val_es)

        pred_test = reg.predict(X_test)
        pred_test_m = reg.predict(Xm_test)
        pred_test_s = reg.predict(Xs)
        pred_train = reg.predict(Xm_train)
        pred_val = reg.predict(Xm_val)

        mse_train = ((y_train - pred_train)**2).mean()
        mse_test = ((y_test - pred_test)**2).mean()
        mse_test_m = ((y_test - pred_test_m)**2).mean()
        mse_test_s = ((ys - pred_test_s)**2).mean()
        mse_val = ((y_val - pred_val)**2).mean()

        var_train = ((y_train - y_train.mean())**2).mean()
        var_test = ((y_test - y_test.mean())**2).mean()
        var_test_s = ((ys - ys.mean())**2).mean()
        var_val = ((y_val - y_val.mean())**2).mean()

        r2_train = 1 - mse_train/var_train
        r2_test = 1 - mse_test/var_test
        r2_test_m = 1 - mse_test_m/var_test
        r2_test_s = 1 - mse_test_s/var_test_s
        r2_val = 1 - mse_val/var_val

        res = {'iter': it, 'n': n, 
               'R2_train': r2_train, 'R2_val': r2_val, 
               'R2_test': r2_test, 'R2_test_m': r2_test_m, 
               'R2_test_s': r2_test_s, 
               'mse_train': mse_train, 'mse_val': mse_val, 
               'mse_test': mse_test, 'mse_test_m': mse_test_m, 
               'mse_test_s': mse_test_s}

        results.append(res)

    return results
