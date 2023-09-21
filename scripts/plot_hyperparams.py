from plot_utils import *
import seaborn as sns

experiment = ["less_miss"]
data = ["simulated"]
link = ["stairs"]
scenario = ["mcar"]
var = "mse_test_m"
scores = [load_scores(e, l, s) for e, l, s in zip(experiment, data, link, scenario)]
scores = [perf_by_params(s) for s in scores]

data = scores[0]


def plot_hyperparams(method, n, param):
    sns.scatterplot(
        data[(data.method == method) & (data.n == n)],
        x=param,
        y="mse_val",
        hue="prop_latent",
    )


plot_hyperparams("ice_impute_mask", 1e5, "mlp_depth")
plot_hyperparams("ice_impute_mask", 1e5, "width_factor")
plot_hyperparams("ice_impute_mask", 1e5, "lr")
plot_hyperparams("ice_impute_mask", 1e5, "weight_decay")

plot_hyperparams("mice_impute", 1e5, "mlp_depth")
plot_hyperparams("mice_impute", 1e5, "width_factor")
plot_hyperparams("mice_impute", 1e5, "lr")
plot_hyperparams("mice_impute", 1e5, "weight_decay")

plot_hyperparams("neumiss", 1e5, "mlp_depth")
plot_hyperparams("neumiss", 1e5, "width_factor")
plot_hyperparams("neumiss", 1e5, "lr")
plot_hyperparams("neumiss", 1e5, "weight_decay")

plot_hyperparams("neumise", 1e5, "mlp_depth")
plot_hyperparams("neumise", 1e5, "width_factor")
plot_hyperparams("neumise", 1e5, "lr")
plot_hyperparams("neumise", 1e5, "weight_decay")


from plot_utils import *
import seaborn as sns

experiments = ["more_miss", "less_miss"]
data = "simulated"
link = "stairs"
scenarios = ["mcar", "monotone_mar", "gaussian_sm", "mar_y"]
var = "mse_test"
n = 1e5


def get_scores(experiment, data, link, scenario, n, var):
    scores = load_scores(experiment, data, link, scenario)
    scores = scores[scores.n == n]
    scores = find_best_params(scores, var)
    scores = scores[~scores.method.str.contains("bayes")]
    return scores


def summarise_hyperparams(scores):
    scores = scores[
        [
            "method",
            "prop_latent",
            "best_lr",
            "best_weight_decay",
            "best_width_factor",
            "best_mlp_depth",
        ]
    ]
    scores = scores.drop_duplicates()
    scores["method"] = pd.Categorical(scores["method"], list(NAMES.keys()))
    return scores.sort_values(["prop_latent", "method"])


hparams = []

for s in scenarios:
    for e in experiments:
        scores = get_scores(e, data, link, s, n, var)
        hparam = summarise_hyperparams(scores)
        hparam["scenario"] = s
        hparam["experiment"] = e
        hparams.append(hparam)


pd.concat(hparams, axis=0)

scores = get_scores(experiment, data, link, scenario, n, f"{var}_m")
