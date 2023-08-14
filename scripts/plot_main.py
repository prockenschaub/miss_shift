from plot_utils import *

def get_scores(experiment, link, scenario, var, n):
    scores = load_scores(experiment, link, scenario)
    scores = find_best_params(scores)
    scores = scores[scores.n == n]
    scores = diff_to_bayes(scores, var)
    scores = rename_methods(scores)
    return scores

def limits(a, b):
    def set_limits(ax):
        ax.set_xlim([a, b])
    return set_limits

experiment = "more_miss"
link = "stairs"
n=1e5

scenario = "monotone_mar"
var = 'mse_test_m'
scores = get_scores(experiment, link, scenario, var, n)

plot_latents(scores, var, type='violin', n=n)

var = 'mse_test_s'
scores = get_scores(experiment, link, scenario, var, n)

plot_latents(scores, var, type='box', n=n)

var = 'mse_test'
scores = get_scores(experiment, link, scenario, var, n)

plot_latents(scores, var, type='box', n=n)


print('*********************')
print('Gaussian self-masking')
scenario = "gaussian_sm"
var = 'mse_test_m'
scores = get_scores(experiment, link, scenario, var)

plot_latents(scores, var, type='box', n=n)

var = 'mse_test_s'
scores = get_scores(experiment, link, scenario, var)

plot_latents(scores, var, type='box', n=n, callback=limits(0, 0.4))

var = 'mse_test'
scores = get_scores(experiment, link, scenario, var)

plot_latents(scores, var, type='box', n=n, callback=limits(0, 0.5))
