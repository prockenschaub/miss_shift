from plot_utils import *
import seaborn as sns

experiment = ['main_experiment']
scenario = ['mcar']
var = 'R2_test_m'
scores = [load_scores(e, s) for e, s in zip(experiment, scenario)]
scores = [perf_by_params(s) for s in scores]

data = scores[0]


def plot_hyperparams(method, n, param):
    sns.scatterplot(
        data[(data.method == method) & (data.n == n)],
        x=param, 
        y='R2_val',
        hue='prop_latent'
    )

plot_hyperparams('neumiss', 2e4, 'mlp_depth')
plot_hyperparams('neumiss', 2e4, 'width_factor')
plot_hyperparams('neumiss', 2e4, 'lr')
plot_hyperparams('neumiss', 2e4, 'weight_decay')

plot_hyperparams('neumice', 2e4, 'mlp_depth')
plot_hyperparams('neumice', 2e4, 'width_factor')
plot_hyperparams('neumice', 2e4, 'lr')
plot_hyperparams('neumice', 2e4, 'weight_decay')