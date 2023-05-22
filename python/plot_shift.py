from plot_utils import *

# Experiment where X_obs may differ at test time
experiments = ['MAR_stairs_shift']
var = 'R2_test_s1'   # Random X_obs in test
scores = [load_scores(e) for e in experiments]
scores = [find_best_params(s) for s in scores]
scores = [diff_to_bayes(s, var) for s in scores]

plot_one(scores[0], var, type='box')


# Experiment where X_obs stays the same at test time
experiments = ['MAR_stairs_shift']
var = 'R2_test_s2'   # Leave X_obs unchanged
scores = [load_scores(e) for e in experiments]
scores = [find_best_params(s) for s in scores]
scores = [diff_to_bayes(s, var) for s in scores]

plot_one(scores[0], var, type='box')