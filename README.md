# Robust prediction under missingness shifts

## Paper

If you use this code in your research, please cite the following publication:

```
@misc{rockenschaub2024robust,
      title={Robust prediction under missingness shifts}, 
      author={Patrick Rockenschaub and Zhicong Xian and Alireza Zamanian and Marta Piperno and Octavia-Andreea Ciora and Elisabeth Pachl and Narges Ahmidi},
      year={2024},
      eprint={2406.16484},
      archivePrefix={arXiv}
}
```

This paper can be found on arxiv: [https://arxiv.org/abs/2406.16484](https://arxiv.org/abs/2406.16484)

## Acknowledgements

This implementation builds on top of the excellent work by Le Morvan et al. (2021). [What's a good imputation to predict with missing values?](https://papers.nips.cc/paper/2021/file/5fe8fdc79ce292c39c5f209d734b7206-Paper.pdf). The original code can be found in the paper's [GitHub repo](https://github.com/marineLM/Impute_then_Regress) (from commit [37e70a3](https://github.com/marineLM/Impute_then_Regress/commit/37e70a33fb60330b6d4f95173d202a58135d0dae)). 


## Repo structure

The repo has been extensively refactored from its original structure to bring it into a more modular form and turn it into a valid python package. The main parts are:

1. `src/`: source code containing all major functionality including both the data generation/amputation as well as all the oracles and estimators.
2. `experiments/`: experiment configurations as `.yaml` files that specify the experiment name, the data generation mechanism, the missing data scenarios, and the estimators including their hyperparameter ranges. 
3. `scripts/`: helper scripts that launch experiments and plot the results

## Setup

Run the following commands in a terminal to clone this repo and create the Conda environment:

```bash
git clone https://github.com/prockenschaub/miss_shift.git
cd miss_shift/

conda env create -f environment.yml
conda activate miss_shift
pip install -e .
```

All experiments were run using Python 3.9.12 on an Apple M1 Max with Ventura 13.2.1 and on a Linux HPC cluster. 


## Running experiments

Running the experiments is just as easy. Again in a terminal, run the following command to replicate the results for conditional oracles on MAR data reported in the paper:  

```bash
python scripts/launch_experiment.py more_miss mcar bayes --data simulated --link stairs
python scripts/launch_experiment.py less_miss mcar bayes --data simulated --link stairs
```

This runs two experiments: one where missingness is 25% in the source environment and 50% in the target environment (`more_miss`) and one where missingness is 50% in the source environment and 25% in the target environment (`less_miss`). Together, these two experiments allow to create the result displayed in Figure 2, with the no shift performance taken from `more_miss` and the shift performance taken from `less_miss`.

To run all estimators reported in the paper, just use `all`:

```bash
python scripts/launch_experiment.py more_miss mcar all --data simulated --link stairs
python scripts/launch_experiment.py less_miss mcar all --data simulated --link stairs
```

All experiment results will be saved in `results/[EXPERIMENT_NAME]/[DATA]/[LINK]`.


## License

This source code is released under a BSD 3-Clause license, included [here](LICENSE.txt).
