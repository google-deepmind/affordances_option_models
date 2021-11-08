# Code for Temporally Abstract Partial Models

Accompanies the code for the experimental section of the paper:
[Temporally Abstract Partial Models, Khetarpal, Ahmed, Comanici and Precup, 2021](https://arxiv.org/abs/2108.03213)
that is to be published at NeurIPS 2021.

## Installation

1. Clone the `deepmind-research` repository and `cd` into this directory:
```
git clone https://github.com/deepmind/deepmind-research.git
```

2. Now install the requirements to your system
`pip install -r ./requirements.txt`. It is recommended
to use a `virtualenv` to isolate dependencies.

For example:
```
git clone https://github.com/deepmind/affordances_option_models.git
cd deepmind-research

python3 -m virtualenv affordances
source affordances/bin/activate

pip install -r ./requirements.txt
```

## Usage

1. The first step of the experiment is to build, train and save the low level
options: `python3 -m affordances_option_models.lp_learn_options --save_path ./options`
which will save the option policies into `./options/args/...`. The low level
options are trained by creating a reward matrix for the 75 options (see
`option_utils.check_option_termination`) and then running value iteration.
2. The next step is to learn the option models, policy over options and
affordance models all online:
`python3 -m affordances_option_models.lp_learn_model_from_options --path_to_options=./options/gamma0.99/max_iterations1000/options/`.
See _Arguments_ below to see how to select `--affordances_name`.


## Arguments

1. The default arguments for `lp_learn_options.py` will produce a reasonable set
of option policies.
2. For `lp_learn_model_from_options.py` use the argument `--affordances_name` to
switch between the affordance that will be used for model learning. For the
heuristic affordances (`everything`, `only_pickup_drop` and
`only_relevant_pickup_drop`) the model learned will be evaluated via value
iteration (i.e. planning) with every other affordance type. For the `learned`
affordances, only `learned` affordances will be used in value iteration.


### Experiments in Section 5.1

To reproduce the experiments with heuristics use the command
```
python3 -m affordances_option_models.lp_learn_model_from_options  \
--num_rollout_nodes=1 --total_steps=50000000 \
--seed=0 --affordances_name=everything
```
and run this command for every combination of the arguments:
- `--seed=`: 0, 1, 2, 3
- `--affordances_name=`: `everything`, `only_pickup_drop`, `only_relevant_pickup_drop`.

### Experiments in Section 5.2

To reproduce the experiments with learned affordances use the command
```
python3 -m affordances_option_models.lp_learn_model_from_options  \
--num_rollout_nodes=1 --total_steps=50000000 --affordances_name=learned \
--seed=0 --affordances_threshold=0.0
```
and run this command for every combination of the arguments:
- `--seed=`: 0, 1, 2, 3
- `--affordances_threshold=`: 0.0, 0.1, 0.25, 0.5, 0.75.


## Citation

If you use this codebase in your research, please cite the paper:

```bibtex
@misc{khetarpal2021temporally,
      title={Temporally Abstract Partial Models},
      author={Khimya Khetarpal and Zafarali Ahmed and Gheorghe Comanici and Doina Precup},
      year={2021},
      eprint={2108.03213},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Disclaimer

This is not an official Google product.
