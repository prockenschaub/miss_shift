from plot_utils import *
import seaborn as sns

experiment = ['less_miss']
link = ['stairs']
scenario = ['mcar']
var = 'mse_test_m'
scores = [load_scores(e, l, s) for e, l, s in zip(experiment, link, scenario)]
scores = [perf_by_params(s) for s in scores]

data = scores[0]


def plot_hyperparams(method, n, param):
    sns.scatterplot(
        data[(data.method == method) & (data.n == n)],
        x=param, 
        y='mse_val',
        hue='prop_latent'
    )

plot_hyperparams('mice_impute_mask', 1e5, 'mlp_depth')
plot_hyperparams('mice_impute_mask', 1e5, 'width_factor')
plot_hyperparams('mice_impute_mask', 1e5, 'lr')
plot_hyperparams('mice_impute_mask', 1e5, 'weight_decay')

plot_hyperparams('multimice_impute', 1e5, 'mlp_depth')
plot_hyperparams('multimice_impute', 1e5, 'width_factor')
plot_hyperparams('multimice_impute', 1e5, 'lr')
plot_hyperparams('multimice_impute', 1e5, 'weight_decay')

plot_hyperparams('neumiss', 1e5, 'mlp_depth')
plot_hyperparams('neumiss', 1e5, 'width_factor')
plot_hyperparams('neumiss', 1e5, 'lr')
plot_hyperparams('neumiss', 1e5, 'weight_decay')

plot_hyperparams('neumice', 1e5, 'mlp_depth')
plot_hyperparams('neumice', 1e5, 'width_factor')
plot_hyperparams('neumice', 1e5, 'lr')
plot_hyperparams('neumice', 1e5, 'weight_decay')