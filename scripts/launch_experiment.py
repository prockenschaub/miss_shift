'''
Defines:
 - the paramaters of data simulations, 
 - the list of methods to compare and their hyperparameters,
And launches all experiments.
'''
from collections import namedtuple
import os
import yaml
import argparse
from joblib import Parallel, delayed
from datetime import datetime

import numpy as np
import pandas as pd

from miss_shift.run import run_one


def configure_runs(method, method_args):
    method_params = pd.DataFrame([method_args])
    for v in method_params.columns:
        method_params = method_params.explode(v)
    return method_params.to_dict(orient='records')

# Result item to create the DataFrame in a consistent way.
fields = ['iter', 'method', 'n', 'mse_train', 'mse_val', 'mse_test', 'mse_test_m', 'mse_test_s',
          'R2_train', 'R2_val', 'R2_test', 'R2_test_m', 'R2_test_s', 
          'early_stopping', 'optimizer', 'depth',
          'n_epochs', 'learning_rate', 'lr', 'weight_decay', 'batch_size',
          'type_width', 'width', 'n_draws', 'n_iter_no_change',
          'verbose', 'mlp_depth', 'init_type', 'max_iter', 'order0',
          'n_trials_no_change', 'validation_fraction', 'add_mask', 'imputation_type', 
          'n_features', 'prop_latent', 'snr', 'miss_orig', 'miss_shift',
          'link', 'curvature', 'width_factor', 'max_leaf_nodes', 'min_samples_leaf',
          'use_y_for_impute',
          'mode', 'input_size', 'latent_size', 'encoder_width', 'K'
   ]

ResultItem = namedtuple('ResultItem', fields)
ResultItem.__new__.__defaults__ = (np.nan, )*len(ResultItem._fields)


def launch(args):
    # Load the experiment definition
    file_path = f"experiments/{args.experiment}.yaml"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist. Please provide an experiment definition.")
    
    with open(file_path, 'r') as f:
        experiment = yaml.load(f, Loader=yaml.FullLoader)

    # Extract the data generation parameters from the experiment definition
    data_spec = experiment['data']
    
    if args.link is not None:
        data_spec['link'] = args.link
    
    if data_spec['link'] == 'square':
        data_spec['curvature'] = 1
    elif data_spec['link'] == 'stairs':
        data_spec['curvature'] = 20
    else:
        data_spec['curvature'] = None

    varying_data_params = [k for k, v in data_spec.items() if isinstance(v, list)]

    # Choose the missingness scenario
    if args.scenario not in experiment['missingness'].keys():
        raise ValueError(f"The missingness scenario '{args.scenario}' is not defined in the experiment '{args.experiment}'.")

    missingness = experiment['missingness'][args.scenario]
    miss_orig = missingness['orig']
    miss_shift = missingness['shift']

    default_values = {**data_spec, 'miss_orig': miss_orig, 'miss_shift': miss_shift}

    # Then vary parameters one by one while the other parameters remain constant,
    # and equal to their default values.
    data_descs = pd.DataFrame([default_values])
    for var in varying_data_params:
        data_descs = data_descs.explode(var)


    # Define the methods that will be compared
    methods_params = {}
    
    if args.estimator == "all":
        estimators = list(experiment['estimators'].keys())
    elif args.estimator.isdigit():
        estimators = [list(experiment['estimators'].keys())[int(args.estimator)]]
    else:
        estimators = [args.estimator]

    for estimator_name in estimators:
        estimator_params = experiment['estimators'][estimator_name]
        methods_params[estimator_name] = configure_runs(estimator_name, estimator_params)


    # Create output directory
    out_dir = os.path.join(args.out_dir, experiment['name'], args.link, args.scenario)
    os.makedirs(out_dir, exist_ok=True)

    # Run all trials for all hyperparam configurations of all models and store results
    for nm, scope in methods_params.items():
        print(f'Start running trials for model {nm}: {datetime.now()}')
        
        runs = []
        for params in scope:
            for data_desc in data_descs.itertuples(index=False):
                data_desc = dict(data_desc._asdict())
                for it in range(args.n_trials):
                    runs.append([data_desc, nm, params, it])

        results = Parallel(n_jobs=args.n_jobs, verbose=11)(
             delayed(run_one)(data_desc, method, method_params, it, args.n_train,
                             args.n_test, args.n_val, miss_orig['mdm'], args.tmp_dir)
             for data_desc, method, method_params, it in runs
         )
        
        print(f'Combining results for model {nm}: {datetime.now()}')

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Define experiment settings
    parser.add_argument('experiment', help='YAML specifying the experiment conditions (data gen, missingness, hyperparams)', type=str)
    parser.add_argument('scenario', help='missing data scenario (defined in experiment)', type=str)
    parser.add_argument('estimator', help='estimator to use (can be "all", a name, or a number corresponding to the order in the experiment YAML)', type=str)
    parser.add_argument('--link', help='type of link function for the outcome',
                        choices=['linear', 'square', 'stairs',
                                'discontinuous_linear'], required=False)
    
    # Define experiment scope
    parser.add_argument('--n_trials', help='number of trials per hyperparameter', type=int, default=1)
    parser.add_argument('--n_train', help='list of train set size(s)', nargs='+', type=int, default=20000)
    parser.add_argument('--n_val', help='size of the validation set', type=int, default=10000)
    parser.add_argument('--n_test', help='size of the test set', type=int, default=10000)

    # Define computational resources, paths, etc.
    parser.add_argument('--n_jobs', help='number of jobs to run in parallel', type=int, default=1)
    parser.add_argument('--out_dir', help='directory where to store the results', type=str, default='results')
    parser.add_argument('--tmp_dir', help='directory where to store any other files (e.g., pre-trained imputation models)', type=str, default='models')

    args = parser.parse_known_args()[0]
    launch(args)