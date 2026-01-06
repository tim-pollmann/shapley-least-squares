## Overview

This repository contains the code accompanying the paper **"On least squares approximations for Shapley values and applications to interpretable machine learning"**.

All experiments described in the paper can be reproduced using the commands described in the [Usage](#usage) section.

## Project structure

| Path | Content |
|----------|----------|
| [```src/shapley_least_squares/approx_algorithms```](https://github.com/tim-pollmann/shapley-least-squares/tree/main/src/shapley_least_squares/approx_algorithms) | The Shapley value approximation algorithms considered in the paper.  |
| [```src/shapley_least_squares/games```](https://github.com/tim-pollmann/shapley-least-squares/tree/main/src/shapley_least_squares/games) | The cooperative games used for the experiments. |
| [```src/shapley_least_squares/scripts```](https://github.com/tim-pollmann/shapley-least-squares/tree/main/src/shapley_least_squares/scripts) | The scripts starting the experiments. |
| [```data```](https://github.com/tim-pollmann/shapley-least-squares/tree/main/data) | The results of the experiments saved as CSV files. |
| [```figures```](https://github.com/tim-pollmann/shapley-least-squares/tree/main/figures) | The figures from paper based on the experiment results in the [```data```](https://github.com/tim-pollmann/shapley-least-squares/tree/main/data) directory. |


## Installation

The installation and execution requires **Python ≥ 3.12**. The code was tested on **Ubuntu 24.04.3 LTS** only. Based on your needs, choose one of the following installation types:

### Standard (recommended)

```sh
pip install .
```

### Development

```sh
pip install -e .[development]
pre-commit install
```

## Usage

After installation, run one of the following commands to reproduce the figures from the paper:

| Command | Description |
|----------|----------|
| ```run-ag-mse-comparison``` | Compares the mean squared errors of all algorithms on an airport game with 100 players. |
| ```run-wvg-mse-comparison``` | Compares the mean squared errors of all algorithms on a weighted voting game with 50 players. |
| ```run-diabetes-mse-comparison``` | Compares the mean squared errors of all algorithms when approximating the feature importances of a [```GradientBoostingRegressor```](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) in the context of the [diabetes dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html). |
| ```run-housing-mse-comparison``` | Compares the mean squared errors of all algorithms when approximating the feature importances of an [```MLPRegressor```](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html) in the context of the [California housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html). |
| ```run-wine-mse-comparison``` | Compares the mean squared errors of all algorithms when approximating the feature importances of a [```RandomForestClassifier```](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) in the context of the [wine dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html) (probability of class $0$ only).  |
| ```run-variance-comparison``` | Compares the theoretical and empirical variances of *UKS*, *LSS*, and *S-LSS* on different weighted voting games. |

> [!IMPORTANT]
> The mean squared error comparisons do ```iters_per_tau``` runs per $\tau$ to average the mean squared error at any given $\tau$. When executing ```SRS-LSS```, it is not guaranteed that the algorithm runs successfully (compare **Proposition XXX** in the paper). Thus, for any $\tau$, we require at least ```iters_per_tau / 2``` successful executions for the average mean squared error to be shown in the final figure.

## Citation

*Will be added after final publication.*

<!-- If you use this code, please cite:

```sh
@article{...}
``` -->
