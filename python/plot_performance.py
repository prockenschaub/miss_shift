from plot_utils import *

experiments = [
    'MCAR_square', 'MCAR_stairs', 
    'gaussian_sm_square', 'gaussian_sm_stairs'
]
var = 'R2_test_m'
scores = [load_scores(e) for e in experiments]
scores = [find_best_params(s) for s in scores]
scores = [diff_to_bayes(s, var) for s in scores]

plot_all(scores, var, type='violin', n=2e4)
plot_all(scores, var, type='box', n=2e4)

