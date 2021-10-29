This repository contains the code to reproduce the experiments of the paper:

__What's a good imputation to predict with missing values?.__

The file **Impute_then_Regress.yml** indicates the packages required as well as
the versions used in our experiments.

The methods used are implemented in the following files:
 * **NeuMiss_accelerated_with_init**: the NeuMiss + MLP network.
 * **mlp_new**: the feedforward neural network.
 * **estimators**: the other methods used.

 The files **ground_truth** and **amputation** contain the code for data
 simulation and the code for the Bayes predictors.

 To reproduce the experiments, use:
  * `python launch_all.py MCAR square` (bowl)
  * `python launch_all.py MCAR stairs` (wave)
  * `python launch_all.py MCAR discontinuous_linear` (break)
  * `python launch_all.py gaussian_sm square` (bowl)
  * `python launch_all.py gaussian_sm stairs` (wave)
  * `python launch_all.py gaussian_sm discontinuous_linear` (break)

Modifications in the parameters of the data simulations can be made in
**launch_all**.

These scripts save their results as csv files in the **results** foder. The
plots can be obtained from these **csv** files by running the **plots_xxx**
files.