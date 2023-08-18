import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


NAMES = {
    'bayes': 'Bayes predictor',
    'bayes_order0': 'Oracle (cond.)',
    'prob_bayes': 'Oracle (prob.)',
    'mice_impute_mask': 'MICE',
    'mice_impute_using_y': 'MICE+Y',
    'multimice_impute': 'MultiMICE',
    'multimice_impute_using_y': 'MultiMICE+Y',
    'neumiss': 'NeuMiss',
    'neumice': 'NeuMISE'
}

COLORS = {
    'bayes': '#949494',
    'bayes_order0': '#fbafe4',
    'prob_bayes': '#cc78bc',
    'mice_impute_mask': '#56b4e9',
    'mice_impute_using_y': '#306582',
    'multimice_impute': '#0173b2',
    'multimice_impute_using_y': '#013f61',
    'neumiss': '#de8f05',
    'neumice': '#d55e00'
}


# Preprocess the scores

def load_scores(experiment, link, scenario, dir='results'):
    folder = os.path.join(dir, experiment, link, scenario)
    files = os.listdir(folder)
    
    scores = [pd.read_csv(os.path.join(folder, f)) for f in files if '.csv' in f]
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


def find_best_params(scores, var): 
    val = re.sub('train|test(_m|_s)?', 'val', var)
    ascending = True if "R2" in val else False
    
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
            'min_samples_leaf', 'max_iter'])[val].mean()
    mean_score = mean_score.reset_index()
    mean_score = mean_score.sort_values(
        by=['method', 'n', 'prop_latent', val],
        ascending=ascending)
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
    def calc_diff(df):
        return df - df["bayes"]
    
    data_relative = scores.copy().set_index('method')
    data_relative[var] = data_relative.groupby(
        ['iter', 'n', 'prop_latent'], axis=0)[var].transform(calc_diff)
    data_relative = data_relative.reset_index()
    return data_relative

def diff_to(est, ref, var_est, var_ref):
    ids = ['iter', 'n', 'prop_latent']
    res = est.merge(ref[ids + [var_ref]], on=ids, suffixes=('', '_ref'))
    res[var_est] = res[var_est] - res[f'{var_ref}_ref']
    return res.drop(columns=f'{var_ref}_ref')

def rename_methods(scores):
    scores = scores.copy()
    scores['method'] = scores['method'].map(NAMES)
    scores = scores.query('~method.isna()').copy()
    scores['method'] = pd.Categorical(scores['method'], list(NAMES.values()))
    return scores


# Plotting

def plot_one(data, var, ax=None, type='violin', setup=False, limit=None):
    if ax is None:
        fig = plt.figure(figsize=(6, 4))   
        ax = fig.add_subplot(1, 1, 1) 

    data = data.copy()
    data['method'] = pd.Categorical(data['method'], list(NAMES.keys()))
    data.loc[:, 'method'] = data['method'].cat.remove_unused_categories()
    methods = data['method'].cat.categories.to_list()

    sns.set_palette(sns.color_palette([COLORS[m] for m in methods]))

    data['method'] = data['method'].cat.rename_categories(NAMES)

    if type == 'violin':
        sns.violinplot(
            data=data, x=var, saturation=1, y='method', ax=ax, scale="width", cut=0, linewidth=1)
    elif type == 'box':
        sns.boxplot(
            data=data, x=var, saturation=1, y='method', ax=ax)
    elif type == 'scatter':
        sns.stripplot(
            data=data, x=var, y='method', hue='method', ax=ax, jitter=.2, alpha=.5, legend=False)

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

    if limit is not None:
        default_max = 0.6

        if isinstance(limit, tuple):
            default_max = limit[1]
            limit = limit[0]

        if callable(limit):
            limit(ax)
        elif isinstance(limit, float):
            ax.set_xlim(right=limit)
        elif limit == 'clip' and data[var].max() > default_max:
            ax.set_xlim(left=-0.03, right=default_max)



def plot_latents(data, var, fig=None, axes=None, i=0, j=0, n=2e4, type='violin', limit=None):
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
            plot_one(to_plot, var, ax, type, limit=limit)

    return fig, axes


def plot_all(lst, var, n=2e4, num_comparisons=2, num_scenarios=2, type='violin', limit=None, n_ticks=3, **kwargs):
    fig, axes = plt.subplots(num_comparisons, 2 * num_scenarios, sharex=kwargs.pop('sharex', 'col'), sharey=kwargs.pop('sharey', True), **kwargs)
    
    i, j = 0, 0
    for data in lst:
        plot_latents(data, var, fig, axes, i, j, n, type, limit)

        if i == num_comparisons - 1:
            i = 0
        else:
            i += 1
        
        if i == 0:
            j += 1

    for j in range(num_scenarios * 2):
        if num_comparisons > 1:
            ax = axes[num_comparisons-1, j]
        else: 
            ax = axes[j]
        _, right_lim = ax.set_xlim()
        ax.set_xlim(left=-right_lim*0.05)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(n_ticks))

    return fig, axes