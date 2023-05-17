'''
Defines:
 - the paramaters of data simulations, 
 - the list of methods to compare and their hyperparameters,
And launches all experiments.
'''
import os
import yaml
import pandas as pd
import argparse
from run_all import run


def configure_runs(method, method_args):
    method_params = pd.DataFrame([method_args])
    for v in method_params.columns:
        method_params = method_params.explode(v)
    return method_params.to_dict(orient='records')


def launch(args):
    
    # Load the experiment definition
    file_path = f"../experiments/{args.experiment}.yaml"
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
    methods_params['bayes'] = [{'order0': False}]

    if args.estimator == "all":
        estimators = list(experiment['estimators'].keys())
    if args.estimator.isdigit():
        estimators = [list(experiment['estimators'].keys())[int(args.estimator)]]
    else:
        estimators = [args.estimator]

    for estimator_name in estimators:
        estimator_params = experiment['estimators'][estimator_name]
        methods_params[estimator_name] = configure_runs(estimator_name, estimator_params)


    # Create output directory
    out_dir = os.path.join(args.out_dir, experiment['name'], args.scenario)
    os.makedirs(out_dir, exist_ok=True)

    run_params = {
            'n_trials': args.n_trials,
            'n_train': args.n_train,
            'n_val': args.n_val,
            'n_test': args.n_test,
            'mdm': miss_orig['mdm'],
            'data_descs': data_descs,
            'methods_params': methods_params,
            'out_dir': out_dir,
            'n_jobs': args.n_jobs}

    run(**run_params)


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
    parser.add_argument('--out_dir', help='directory where to store the results', type=str, default='../results')


    args = parser.parse_known_args()[0]
    launch(args)