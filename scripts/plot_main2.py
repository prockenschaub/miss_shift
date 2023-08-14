from plot_utils import *

def get_scores(experiment, link, scenario, var):
    scores = load_scores(experiment, link, scenario)
    scores = find_best_params(scores)
    #scores = diff_to_bayes(scores, var)
    #scores = rename_methods(scores)
    return scores

def diff_in_diff(ns, s, var):
    ids = ['method', 'n', 'prop_latent', 'iter']
    diff = ns[ids + [var]].merge(s[ids + [f'{var}_s']], on=ids)
    diff[f'{var}_diff'] = diff[f'{var}_s'] - diff[var]
    return diff


def limits(a, b):
    def set_limits(ax):
        ax.set_xlim([a, b])
    return set_limits

ns_exp = "more_miss"
s_exp = "less_miss"
link = "stairs"
var = 'mse_test'
n=1e5

scenario = "gaussian_sm"

ns_bayes = get_scores(ns_exp, link, scenario, f'{var}_m')
ns_bayes = ns_bayes[ns_bayes.method == "bayes"]


s_bayes  = get_scores(s_exp, link, scenario, f'{var}_s')
s_bayes = s_bayes[s_bayes.method == "bayes"]

import seaborn as sns

sns.boxplot(ns_bayes[(ns_bayes.n == 1e5) & (ns_bayes.prop_latent == 0.3)], x="method", y='mse_test_m')
sns.boxplot(s_bayes[(s_bayes.n == 1e5) & (s_bayes.prop_latent == 0.3)], x="method", y='mse_test_s')
sns.boxplot(s_bayes[(s_bayes.n == 1e5) & (s_bayes.prop_latent == 0.3)], x="method", y='mse_test_m')



def diff_to(est, ref, var_est, var_ref):
    ids = ['iter', 'n', 'prop_latent']
    res = est.merge(ref[ids + [var_ref]], on=ids, suffixes=('', '_ref'))
    res[var_est] = res[var_est] - res[f'{var_ref}_ref']
    return res.drop(columns=f'{var_ref}_ref')


ns_scores = get_scores(ns_exp, link, scenario, f'{var}_m')
ns_diff = diff_to(ns_scores, ns_scores[ns_scores.method == 'bayes'], f'{var}_m', f'{var}_m')
#ns_diff = ns_diff[ns_diff.method != 'bayes']

s_scores = get_scores(s_exp, link, scenario, f'{var}_s')
s_diff = diff_to(s_scores, ns_scores[ns_scores.method == 'bayes'], f'{var}_s', f'{var}_m')

plot_latents(rename_methods(ns_scores), f'{var}_m', type='box', n=n)
plot_latents(rename_methods(s_scores), f'{var}_s', type='box', n=n)

fig, axes = plot_latents(rename_methods(ns_diff), f'{var}_m', type='box', n=n)
fig.savefig(f'results/figures/{ns_exp}_{s_exp}_{link}_{scenario}_noshift.png')
fig, axes = plot_latents(rename_methods(s_diff), f'{var}_s', type='box', n=n)
fig.savefig(f'results/figures/{ns_exp}_{s_exp}_{link}_{scenario}_shift.png')
fig, axes = plot_latents(rename_methods(s_diff), f'{var}', type='box', n=n, callback=limits(0, 0.6))
fig.savefig(f'results/figures/{ns_exp}_{s_exp}_{link}_{scenario}_complete.png')
