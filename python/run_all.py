import numpy as np
import pandas as pd
from collections import namedtuple
from joblib import Memory, Parallel, delayed
from ground_truth import gen_params, gen_data,\
                         gen_params_selfmasking, gen_data_selfmasking
from NeuMiss_accelerated_with_init import Neumiss_mlp
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from estimators import ImputeMLPPytorch, OracleImputeMLPPytorch
from ground_truth import BayesPredictor_GSM_nonlinear,\
                         BayesPredictor_MCAR_MAR_nonlinear,\
                         ProbabilisticBayesPredictor
from tqdm import tqdm
import torch
location = './cachedir'
memory = Memory(location, verbose=0)


# Result item to create the DataFrame in a consistent way.
fields = ['iter', 'method', 'n', 'mse_train', 'mse_val', 'mse_test', 'mse_test_m', 'mse_test_s1', 'mse_test_s2',
          'R2_train', 'R2_val', 'R2_test', 'R2_test_m', 'R2_test_s1', 'R2_test_s2',
          'early_stopping', 'optimizer', 'residual_connection', 'depth',
          'n_epochs', 'learning_rate', 'lr', 'weight_decay', 'batch_size',
          'type_width', 'width',
          'mode',  'verbose', 'mlp_depth', 'init_type', 'max_iter', 'order0',
          'n_iter_no_change', 'add_mask', 'imputation_type', 'mdm',
          'n_features', 'missing_rate', 'prop_latent', 'snr', 'masking',
          'prop_for_masking', 'link', 'curvature', 'sm_type', 'sm_param',
          'perm', 'width_factor', 'max_leaf_nodes', 'min_samples_leaf']

ResultItem = namedtuple('ResultItem', fields)
ResultItem.__new__.__defaults__ = (np.nan, )*len(ResultItem._fields)


def run(n_iter, n_sizes, n_test, n_val, mdm, data_descs, methods_params,
        filename, n_jobs=1):

    params = []
    for method_params in methods_params:
        method = method_params.pop('method')
        for data_desc in data_descs.itertuples(index=False):
            data_desc = dict(data_desc._asdict())
            for it in range(n_iter):
                params.append([data_desc, method, method_params, it])

    results = Parallel(n_jobs=n_jobs)(
        delayed(run_one)(data_desc, method, method_params, it, n_sizes,
                         n_test, n_val, mdm)
        for data_desc, method, method_params, it in tqdm(params)
    )

    # combined_results is a list of all result items that combine the obtained
    # performances and the corresponding data and method parameters.
    # Note that results has the same size as store_params (correspondance)
    combined_results = []
    for i in range(len(params)):
        data_desc, method, method_params, _ = params[i]
        result = results[i]
        for result_n in result:
            result_item = ResultItem(
                method=method, **result_n, **data_desc, **method_params)
            combined_results.append(result_item)

    combined_results = pd.DataFrame(combined_results)
    combined_results.to_csv('../results/' + filename + '.csv')


def update_method_params(method, method_params, data_params=None):
    '''
    data_params: tuple
        Only useful in simulations to recover the data parameters for
        analytical NeuMiss or oracle imputation.
    '''

    params = method_params.copy()

    # Recover ground truth parameters for oracles
    if 'oracle' in method:
        if params['mdm'] in ['MCAR', 'MAR']:
            (_, mean, cov, beta, _, _, _, _, _, _) = data_params
        elif params['mdm'] == 'gaussian_sm':
            (_, _, sm_params, mean, cov, beta, _, _, _, _) = data_params
            params['tsigma2'] = sm_params['sigma2_tilde']
            k = sm_params['k']
            params['tmu'] = mean + k*np.sqrt(np.diag(cov))
        params['Sigma'] = cov
        params['mu'] = mean

    if method in ['BayesPredictor', 'BayesPredictor_order0', 'ProbabilisticBayesPredictor']:
        params['data_params'] = data_params

    return params


@memory.cache
def run_one(data_desc, method, method_params, it, n_sizes, n_test, n_val, mdm):

    n_tot = [n_train + n_test + n_val for n_train in n_sizes]

    # Generate the data
    if mdm == 'gaussian_sm':
        generate_params = gen_params_selfmasking
        generate_data = gen_data_selfmasking
    elif mdm in ['MCAR', 'MAR']:
        generate_params = gen_params
        generate_data = gen_data
    data_params = generate_params(**data_desc, random_state=it)
    gen = generate_data(n_tot, data_params, random_state=it, sample_vars=False)
    gen_shift1 = generate_data(n_tot, data_params, random_state=it*42)
    gen_shift2 = generate_data(n_tot, data_params, random_state=it*4242, sample_vars=False)


    updated_method_params = update_method_params(
        method, method_params, data_params)

    # Get method name and initialize estimator
    if method == "ProbabilisticBayesPredictor":
        est = ProbabilisticBayesPredictor
    elif 'BayesPredictor' in method:
        if mdm == 'gaussian_sm':
            est = BayesPredictor_GSM_nonlinear
        elif mdm in ['MCAR', 'MAR']:
            est = BayesPredictor_MCAR_MAR_nonlinear
    elif 'NeuMiss' in method:
        est = Neumiss_mlp
    elif 'oracleMLPPytorch' in method:
        est = OracleImputeMLPPytorch
    elif ('meanMLPPytorch' in method) or ('MICEMLPPytorch' in method)  or ('MultiMICEMLPPytorch' in method):
        est = ImputeMLPPytorch
    elif 'GBRT' in method:
        est = HistGradientBoostingRegressor
    else:
        raise ValueError('{} is not a known method'.format(method))

    # A list of dictionaries that give the MSE an R2 for each n and train, test
    # and validation sets.
    results = []

    # Loop over the different dataset sizes
    for (X, Xm, y), (_, Xs1, ys1), (_, Xs2, ys2) in zip(gen, gen_shift1, gen_shift2):
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

        if ('NeuMiss' in method and
           'custom_normal' in updated_method_params['init_type']):
            mask = 1 - np.isnan(Xm_train)
            mu_hat = np.nanmean(Xm_train, axis=0)
            X_train_centered = np.nan_to_num(Xm_train-mu_hat)
            Sigma_hat = X_train_centered.T.dot(X_train_centered)
            den = mask.T.dot(mask)
            Sigma_hat = Sigma_hat/(den-1)
            L_hat = np.linalg.norm(Sigma_hat, ord=2)
            updated_method_params['Sigma'] = Sigma_hat
            updated_method_params['mu'] = mu_hat
            updated_method_params['L'] = L_hat

        # Set the torch seed
        torch.manual_seed(0)
        if method in ['GBRT']:
            # For these methods the validation data for early stopping should
            # be given as a fraction of the training set.
            updated_method_params['validation_fraction'] = (
                n_val_half/(n + n_val_half))
            X_train_val_es = X[(n_test + n_val_half):]
            y_train_val_es = y[(n_test + n_val_half):]
            reg = est(**updated_method_params)
            reg.fit(X_train_val_es, y_train_val_es)
        elif method ==  'ProbabilisticBayesPredictor':
            reg = est(mdm=mdm, **updated_method_params)
            reg.fit(Xm_train, y_train)
        elif method in ['BayesPredictor', 'BayesPredictor_order0']:
            reg = est(**updated_method_params)
            reg.fit(Xm_train, y_train)
        else:
            # For these methods the validatin data for early stopping should be
            # given as standalone data.
            reg = est(**updated_method_params)
            reg.fit(Xm_train, y_train, X_val=Xm_val_es, y_val=y_val_es)

        pred_test = reg.predict(X_test)
        pred_test_m = reg.predict(Xm_test)
        pred_test_s1 = reg.predict(Xs1)
        pred_test_s2 = reg.predict(Xs2)
        pred_train = reg.predict(Xm_train)
        pred_val = reg.predict(Xm_val)

        mse_train = ((y_train - pred_train)**2).mean()
        mse_test = ((y_test - pred_test)**2).mean()
        mse_test_m = ((y_test - pred_test_m)**2).mean()
        mse_test_s1 = ((ys1 - pred_test_s1)**2).mean()
        mse_test_s2 = ((ys2 - pred_test_s2)**2).mean()
        mse_val = ((y_val - pred_val)**2).mean()

        var_train = ((y_train - y_train.mean())**2).mean()
        var_test = ((y_test - y_test.mean())**2).mean()
        var_test_s1 = ((ys1 - ys1.mean())**2).mean()
        var_test_s2 = ((ys2 - ys2.mean())**2).mean()
        var_val = ((y_val - y_val.mean())**2).mean()

        r2_train = 1 - mse_train/var_train
        r2_test = 1 - mse_test/var_test
        r2_test_m = 1 - mse_test_m/var_test
        r2_test_s1 = 1 - mse_test_s1/var_test_s1
        r2_test_s2 = 1 - mse_test_s2/var_test_s2
        r2_val = 1 - mse_val/var_val

        res = {'iter': it, 'n': n, 
               'R2_train': r2_train, 'R2_val': r2_val, 
               'R2_test': r2_test, 'R2_test_m': r2_test_m, 
               'R2_test_s1': r2_test_s1, 'R2_test_s2': r2_test_s2,
               'mse_train': mse_train, 'mse_val': mse_val, 
               'mse_test': mse_test, 'mse_test_m': mse_test_m, 
               'mse_test_s1': mse_test_s1, 'mse_test_s2': mse_test_s2,}

        results.append(res)

    return results
