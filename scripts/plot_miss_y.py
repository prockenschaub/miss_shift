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
n=1e5

save = True

# MAR-Y results for the main text (Figure 3) ----------------------

data = "simulated"

# Get complete data performance for comparison
comparison = get_scores(ns_exp, data, link, "mcar", n, f'{var}')
comparison = comparison[comparison.method == "bayes"]

to_plot = []

scenario = "mar_y"

ns_scores = get_scores(ns_exp, data, link, scenario, n, f'{var}_m')
ns_diff = diff_to(ns_scores, comparison, f'{var}_m', f'{var}')

s_scores = get_scores(s_exp, data, link, scenario, n, f'{var}_s')
s_diff = diff_to(s_scores, comparison, f'{var}_s', f'{var}')

to_plot += [
    ns_diff.rename(columns={f'{var}_m': 'result'}),
    s_diff. rename(columns={f'{var}_s': 'result'})
]

plt.rcParams.update({'font.size': 8})

fig, ax = plot_all(to_plot, 'result', n, num_scenarios=1, type='violin', figsize=(2, 3), sharex="row")
fig.supxlabel('Change in MSE', y=-0.05)
fig.text(0.51, 0.97, 'MAR-Y', ha='center', fontweight='bold')
ax[0, 0].set_ylabel('No shift (25%)', labelpad=10., fontweight='bold')
ax[1, 0].set_ylabel('Shift (50% to 25%)', labelpad=10., fontweight='bold')

ax[0, 0].set_xlabel('High corr.')
ax[0, 0].xaxis.set_label_position('top') 
ax[0, 1].set_xlabel('Low corr.')
ax[0, 1].xaxis.set_label_position('top') 

ax[0, 0].annotate('', xy=(0, 1.2), xycoords='axes fraction', xytext=(2.2, 1.2),
arrowprops=dict(arrowstyle="-"))

ax[0, 0].set_xlim(left=-0.125, right=0.435)
ax[0, 0].set_xticks([0, 0.2, 0.4])
ax[1, 0].set_xlim(left=-1.25, right=4.35)
ax[1, 0].set_xticks([0, 2, 4])


if save:
    fig.savefig('results/figures/shift_performance_y.png', dpi=150, bbox_inches = "tight")

