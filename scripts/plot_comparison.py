from plot_utils import *

def get_scores(experiment, data, link, scenario, n, var):
    scores = load_scores(experiment, data, link, scenario)
    scores = scores[scores.n == n]
    scores = find_best_params(scores, var)
    return scores

ns_exp = "more_miss"
s_exp = "less_miss"
link = "stairs"
var = 'mse_test'
n = 1e5

save = False

# MAR and MNAR results for the main text (Figure 1) ----------------------

to_plot = []

data = "simulated"
scenario = "monotone_mar"

ns_scores = get_scores(ns_exp, data, link, scenario, n, f'{var}_m')
ns_diff = diff_to(ns_scores, ns_scores[ns_scores.method == 'bayes'], f'{var}_m', f'{var}_m')

s_scores = get_scores(s_exp, data, link, scenario, n, f'{var}_s')
s_diff = diff_to(s_scores, ns_scores[ns_scores.method == 'bayes'], f'{var}_s', f'{var}_m')

to_plot += [
    ns_diff.rename(columns={f'{var}_m': 'result'}),
    s_diff. rename(columns={f'{var}_s': 'result'})
]


data = "simulated"
scenario = "gaussian_sm"

ns_scores = get_scores(ns_exp, data, link, scenario, n, f'{var}_m')
ns_diff = diff_to(ns_scores, ns_scores[ns_scores.method == 'bayes'], f'{var}_m', f'{var}_m')

s_scores = get_scores(s_exp, data, link, scenario, n, f'{var}_s')
s_diff = diff_to(s_scores, ns_scores[ns_scores.method == 'bayes'], f'{var}_s', f'{var}_m')

to_plot += [
    ns_diff.rename(columns={f'{var}_m': 'result'}),
    s_diff. rename(columns={f'{var}_s': 'result'})
]

plt.rcParams.update({'font.size': 8})

fig, ax = plot_all(to_plot, 'result', n, num_scenarios=2, type='violin', limit='clip', figsize=(5.5, 4.5))
fig.supxlabel('Increase in MSE compared to Bayes predictor')
fig.text(0.31, 0.94, 'Monotone MAR', ha='center', fontweight='bold')
fig.text(0.72, 0.94, 'MNAR (gaussian self-masking)', ha='center', fontweight='bold')
ax[0, 0].set_ylabel('No shift (25%)', labelpad=20., fontweight='bold')
ax[1, 0].set_ylabel('Shift (50% to 25%)', labelpad=20., fontweight='bold')

ax[0, 0].set_xlabel('High correlation')
ax[0, 0].xaxis.set_label_position('top') 
ax[0, 1].set_xlabel('Low correlation')
ax[0, 1].xaxis.set_label_position('top') 
ax[0, 2].set_xlabel('High correlation')
ax[0, 2].xaxis.set_label_position('top') 
ax[0, 3].set_xlabel('Low correlation')
ax[0, 3].xaxis.set_label_position('top') 

ax[0, 0].annotate('', xy=(0, 1.12), xycoords='axes fraction', xytext=(2.2, 1.12),
arrowprops=dict(arrowstyle="-"))
ax[0, 2].annotate('', xy=(0, 1.12), xycoords='axes fraction', xytext=(2.2, 1.12),
arrowprops=dict(arrowstyle="-"))

if save:
    fig.savefig('results/figures/shift_performance_mar_mnar.png', dpi=150, bbox_inches = "tight")



# MAR and MNAR results for complete data (Figure 2) ----------------------

to_plot = []

data = "simulated"
scenario = "monotone_mar"

s_scores = get_scores(s_exp, data, link, scenario, n, f'{var}')
s_diff = diff_to(s_scores, s_scores[s_scores.method == 'bayes'], f'{var}', f'{var}')
s_diff = s_diff.loc[~s_diff.method.isin(['bayes', 'bayes_order0', 'prob_bayes']), :]

to_plot += [
    s_diff.rename(columns={f'{var}': 'result'})
]


data = "simulated"
scenario = "gaussian_sm"

s_scores = get_scores(s_exp, data, link, scenario, n, f'{var}')
s_diff = diff_to(s_scores, s_scores[s_scores.method == 'bayes'], f'{var}', f'{var}')
s_diff = s_diff.loc[~s_diff.method.isin(['bayes', 'bayes_order0', 'prob_bayes']), :]

to_plot += [
    s_diff. rename(columns={f'{var}': 'result'})
]

plt.rcParams.update({'font.size': 8})

fig, ax = plot_all(to_plot, 'result', n, num_comparisons=1, num_scenarios=2, type='violin', limit=('clip', 1.2), figsize=(5.5, 1.25))
fig.supxlabel('Increase in MSE compared to Bayes predictor', y = -0.25)
fig.text(0.31, 1.1, 'Monotone MAR', ha='center', fontweight='bold')
fig.text(0.72, 1.1, 'MNAR (gaussian self-masking)', ha='center', fontweight='bold')
ax[0].set_ylabel('Shift (50% to 0%)', labelpad=20., fontweight='bold')

ax[0].set_xlabel('High correlation')
ax[0].xaxis.set_label_position('top') 
ax[1].set_xlabel('Low correlation')
ax[1].xaxis.set_label_position('top') 
ax[2].set_xlabel('High correlation')
ax[2].xaxis.set_label_position('top') 
ax[3].set_xlabel('Low correlation')
ax[3].xaxis.set_label_position('top') 

ax[0].annotate('', xy=(0, 1.2), xycoords='axes fraction', xytext=(2.2, 1.2),
arrowprops=dict(arrowstyle="-"))
ax[2].annotate('', xy=(0, 1.2), xycoords='axes fraction', xytext=(2.2, 1.2),
arrowprops=dict(arrowstyle="-"))


if save:
    fig.savefig('results/figures/complete_performance_mar_mnar.png', dpi=150, bbox_inches = "tight")



# MCAR results for the appendix ----------------------------------

to_plot = []

data = "simulated"
scenario = "mcar"

ns_scores = get_scores(ns_exp, data, link, scenario, n, f'{var}_m')
ns_diff = diff_to(ns_scores, ns_scores[ns_scores.method == 'bayes'], f'{var}_m', f'{var}_m')

s_scores = get_scores(s_exp, data, link, scenario, n, f'{var}_s')
s_diff = diff_to(s_scores, ns_scores[ns_scores.method == 'bayes'], f'{var}_s', f'{var}_m')

to_plot += [
    ns_diff.rename(columns={f'{var}_m': 'result'}),
    s_diff. rename(columns={f'{var}_s': 'result'})
]

fig, ax = plot_all(to_plot, 'result', n, num_scenarios=1, type='violin', limit='clip', figsize=(2.75, 4.5))
fig.supxlabel('Increase in MSE compared to Bayes predictor')
fig.text(0.50, 0.94, 'MCAR', ha='center', fontweight='bold')
ax[0, 0].set_ylabel('No shift (25%)', labelpad=20., fontweight='bold')
ax[1, 0].set_ylabel('Shift (50% to 25%)', labelpad=20., fontweight='bold')

ax[0, 0].set_xlabel('High correlation')
ax[0, 0].xaxis.set_label_position('top') 
ax[0, 1].set_xlabel('Low correlation')
ax[0, 1].xaxis.set_label_position('top') 

ax[0, 0].annotate('', xy=(0, 1.12), xycoords='axes fraction', xytext=(2.2, 1.12),
arrowprops=dict(arrowstyle="-"))


if save:
    fig.savefig('results/figures/shift_performance_mcar.png', dpi=150, bbox_inches = "tight")



# LBIDD results ------------------------------------------------

to_plot = []

data = "lbidd"
scenario = "monotone_mar"

ns_scores = get_scores(ns_exp, data, link, scenario, n, f'{var}_m')
s_scores = get_scores(s_exp, data, link, scenario, n, f'{var}_s')

to_plot += [
    ns_scores.rename(columns={f'{var}_m': 'result'}),
    s_scores. rename(columns={f'{var}_s': 'result'})
]


fig, axes = plt.subplots(2, 1, figsize=(2, 3), sharex='col', sharey=True)

plot_one(to_plot[0], 'result', ax=axes[0])
plot_one(to_plot[1], 'result', ax=axes[1])

fig.supxlabel('MSE', x=0.535, y = -0.05)
fig.text(0.525, 0.97, 'Monotone MAR', ha='center', fontweight='bold')
axes[0].set_ylabel('No shift (25%)', labelpad=20., fontweight='bold')
axes[0].grid(axis='x')
axes[0].set_axisbelow(True)
axes[1].set_ylabel('Shift (50% to 25%)', labelpad=20., fontweight='bold')
axes[1].grid(axis='x')
axes[1].set_axisbelow(True)

axes[0].set_xlabel('LBIDD')
axes[0].xaxis.set_label_position('top') 

axes[0].annotate('', xy=(0, 1.2), xycoords='axes fraction', xytext=(1, 1.2),
arrowprops=dict(arrowstyle="-"))


if save:
    fig.savefig('results/figures/lbidd_performance_mar.png', dpi=150, bbox_inches = "tight")
