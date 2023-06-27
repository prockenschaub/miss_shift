import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

NAMES = {
    'bayes_order0': 'Chaining oracles (cond.)',
    'prob_bayes': 'Chaining oracles (prob.)',
    'oracle_impute': 'Oracle impute + MLP',
    'oracle_impute_mask': 'Oracle impute & mask + MLP',
    'neumice': 'NeuMICE + MLP',
    'neumiss': 'NeuMiss + MLP',
    'mice_impute': 'MICE + MLP',
    'mice_impute_mask': 'MICE & mask + MLP',
    'multimice_impute': 'MultiMICE + MLP',
    'multimice_impute_mask': 'MultiMICE & mask + MLP',
    'mean_impute': 'mean impute + MLP',
    'mean_impute_mask': 'mean impute & mask + MLP',
    'gbrt': 'Gradient-boosted trees',
}

# Preprocess the scores

def load_scores(experiment, link, scenario, dir='results'):
    folder = os.path.join(dir, experiment, link, scenario)
    files = os.listdir(folder)
    
    scores = [pd.read_csv(os.path.join(folder, f)) for f in files]
    scores = pd.concat(scores, axis=0)
    
    scores['method'] = scores['method'].mask(scores['order0'] == True, scores['method'] + '_order0')
    scores['method'] = scores['method'].mask(scores['add_mask'] == True, scores['method'] + '_mask')
    return scores


def perf_by_params(scores): 
    scores_no_na = scores.copy()
    scores_no_na['depth'] = scores_no_na['depth'].fillna(value=0)
    scores_no_na['mlp_depth'] = scores_no_na['mlp_depth'].fillna(value=0)
    scores_no_na['lr'] = scores_no_na['lr'].fillna(value=0)
    scores_no_na['weight_decay'] = scores_no_na['weight_decay'].fillna(
        value=0)
    scores_no_na['width_factor'] = scores_no_na['width_factor'].fillna(
        value=0)
    scores_no_na['max_leaf_nodes'] = scores_no_na['max_leaf_nodes'].fillna(
        value=0)
    scores_no_na['min_samples_leaf'] = scores_no_na[
        'min_samples_leaf'].fillna(value=0)
    scores_no_na['max_iter'] = scores_no_na['max_iter'].fillna(value=0)
    # Averaging over iterations
    mean_score = scores_no_na.groupby(
        ['method', 'n', 'prop_latent', 'depth', 'mlp_depth', 'lr',
            'weight_decay', 'width_factor', 'max_leaf_nodes',
            'min_samples_leaf', 'max_iter'])['R2_val'].mean()
    mean_score = mean_score.reset_index()
    mean_score = mean_score.sort_values(
        by=['method', 'n', 'prop_latent', 'R2_val'])
    return mean_score


def find_best_params(scores): 
    scores_no_na = scores.copy()
    scores_no_na['depth'] = scores_no_na['depth'].fillna(value=0)
    scores_no_na['mlp_depth'] = scores_no_na['mlp_depth'].fillna(value=0)
    scores_no_na['lr'] = scores_no_na['lr'].fillna(value=0)
    scores_no_na['weight_decay'] = scores_no_na['weight_decay'].fillna(
        value=0)
    scores_no_na['width_factor'] = scores_no_na['width_factor'].fillna(
        value=0)
    scores_no_na['max_leaf_nodes'] = scores_no_na['max_leaf_nodes'].fillna(
        value=0)
    scores_no_na['min_samples_leaf'] = scores_no_na[
        'min_samples_leaf'].fillna(value=0)
    scores_no_na['max_iter'] = scores_no_na['max_iter'].fillna(value=0)
    # Averaging over iterations
    mean_score = scores_no_na.groupby(
        ['method', 'n', 'prop_latent', 'depth', 'mlp_depth', 'lr',
            'weight_decay', 'width_factor', 'max_leaf_nodes',
            'min_samples_leaf', 'max_iter'])['R2_val'].mean()
    mean_score = mean_score.reset_index()
    mean_score = mean_score.sort_values(
        by=['method', 'n', 'prop_latent', 'R2_val'])
    best_depth = mean_score.groupby(
        ['method', 'n', 'prop_latent']).last()[
            ['depth', 'mlp_depth', 'lr', 'weight_decay', 'width_factor',
                'max_leaf_nodes', 'min_samples_leaf', 'max_iter']]
    best_depth = best_depth.rename(
        columns={'depth': 'best_depth', 'mlp_depth': 'best_mlp_depth',
                    'lr': 'best_lr', 'weight_decay': 'best_weight_decay',
                    'width_factor': 'best_width_factor',
                    'max_leaf_nodes': 'best_max_leaf_nodes',
                    'min_samples_leaf': 'best_min_samples_leaf',
                    'max_iter': 'best_max_iter'})
    scores_no_na = scores_no_na.set_index(
        ['method', 'n', 'prop_latent']).join(best_depth)
    scores_no_depth = scores_no_na.reset_index()
    tmp = ('depth == best_depth and mlp_depth == best_mlp_depth' +
            ' and lr == best_lr and weight_decay == best_weight_decay' +
            ' and width_factor == best_width_factor' +
            ' and max_leaf_nodes == best_max_leaf_nodes' +
            ' and min_samples_leaf == best_min_samples_leaf' +
            ' and max_iter == best_max_iter')
    scores_no_depth = scores_no_depth.query(tmp)
    return scores_no_depth

def diff_to_bayes(scores, var):
    data_relative = scores.copy().set_index('method')
    data_relative[var] = data_relative.groupby(
        ['iter', 'n', 'prop_latent'])[var].transform(
            lambda df: df - df["bayes"])
    data_relative = data_relative.reset_index()
    data_relative['method'] = data_relative['method'].map(NAMES)
    data_relative = data_relative.query('~method.isna()').copy()
    data_relative['method'] = pd.Categorical(data_relative['method'], list(NAMES.values()))
    return data_relative


# Plotting

def plot_one(data, var, ax=None, type='violin', setup=False, callback=None):
    if ax is None:
        fig = plt.figure(figsize=(6, 4))   
        ax = fig.add_subplot(1, 1, 1) 

    sns.set_palette('bright')

    data = data.copy()
    data.loc[:, 'method'] = data['method'].cat.remove_unused_categories()
    methods = data['method'].unique()

    if type == 'violin':
        sns.violinplot(
            data=data, x=var, saturation=1, y='method', ax=ax, scale="width", palette='bright')
    elif type == 'box':
        sns.boxplot(
            data=data, x=var, saturation=1, y='method', ax=ax, palette='bright')
    elif type == 'scatter':
        sns.stripplot(
            data=data, x=var, y='method', hue='method', ax=ax, jitter=.2, alpha=.5, palette='bright', legend=False)

    for i in range(len(methods)):
        if i % 2:
            ax.axhspan(i - .5, i + .5, color='.9', zorder=0)

    # Set axes
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.axvline(0, color='.1', linewidth=1)
    ax.spines['right'].set_edgecolor('.6')
    ax.spines['left'].set_edgecolor('.6')
    ax.spines['top'].set_edgecolor('.6')
    ax.set_ylim(len(methods) - .5, -.5) 

    if callback:
        callback(ax)

def plot_latents(data, var, axes=None, i=0, j=0, n=2e4, type='violin', callback=None):
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex='col', sharey=True)
    
    for k, prop_latent in enumerate([0.3, 0.7]):
            to_plot = data.query('n == @n and prop_latent == @prop_latent')
            if len(axes.shape) == 1:
                ax = axes[2*j+k]
            else:
                ax = axes[i, 2*j+k]
            ax.grid(axis='x')
            ax.set_axisbelow(True)
            plot_one(to_plot, var, ax, type, callback=callback)


def plot_all(lst, var, n=2e4, num_experiments=2, type='violin', callback=None):
    fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharex='col', sharey=True)
    
    i, j = 0, 0
    for data in lst:
        plot_latents(data, var, axes, i, j, n, type, callback)

        if j == num_experiments - 1:
            j = 0
        else:
            j += 1
        
        if j == 0:
            i += 1
