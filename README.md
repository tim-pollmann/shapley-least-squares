## Overview

This repository contains the code accompanying the paper **"On Least Squares Approximations for Shapley Values and Applications to Interpretable Machine Learning"**.

All experiments described in the paper can be reproduced using the commands described in the [Usage](#usage) section.

## Project structure

| Path | Content |
|----------|----------|
| [```src/shapley_least_squares/approx_algorithms```](https://github.com/tim-pollmann/shapley-least-squares/tree/main/src/shapley_least_squares/approx_algorithms) | The Shapley value approximation algorithms considered in the paper.  |
| [```src/shapley_least_squares/games```](https://github.com/tim-pollmann/shapley-least-squares/tree/main/src/shapley_least_squares/games) | The cooperative games used for the experiments. |
| [```src/shapley_least_squares/scripts```](https://github.com/tim-pollmann/shapley-least-squares/tree/main/src/shapley_least_squares/scripts) | The scripts starting the experiments. |
| [```data```](https://github.com/tim-pollmann/shapley-least-squares/tree/main/data) | The results of the experiments saved as CSV files. |
| [```figures```](https://github.com/tim-pollmann/shapley-least-squares/tree/main/figures) | The figures from the paper based on the experiment results in the [```data```](https://github.com/tim-pollmann/shapley-least-squares/tree/main/data) directory. |


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
| ```run-wvg-n-mse-comparison``` | Compares the mean squared errors of all algorithms on a weighted voting game with 50 players with the weights being normally distributed. |
| ```run-wvg-u-mse-comparison``` | Compares the mean squared errors of all algorithms on a weighted voting game with 150 players with the weights being uniformly distributed. |
| ```run-diabetes-mse-comparison``` | Compares the mean squared errors of all algorithms when approximating the feature importances of a [```GradientBoostingRegressor```](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) in the context of the [diabetes dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html). |
| ```run-housing-mse-comparison``` | Compares the mean squared errors of all algorithms when approximating the feature importances of an [```MLPRegressor```](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html) in the context of the [California housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html). |
| ```run-wine-mse-comparison``` | Compares the mean squared errors of all algorithms when approximating the feature importances of a [```RandomForestClassifier```](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) in the context of the [wine dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html) (probability of class $0$ only).  |
| ```run-variance-comparison``` | Compares the theoretical and empirical variances of *UKS*, *LSS*, and *S-LSS* on different weighted voting games. |
| ```show-lss-vs-uks-sampling-probs-comparison``` | Visualizes the sampling distributions of *LSS* and *UKS*. |
| ```create-raw-ag-data``` | Runs all algorithms 1000 times on an airport game with 100 players and saves the corresponding approximated Shapley values in raw format. |
| ```create-raw-wvg-data``` | Runs all algorithms 1000 times on a weighted voting game with 50 players and saves the corresponding approximated Shapley values in raw format. |
| ```extract-ag-stats``` | Extracts key statistics and validates the Hoeffding bounds based on the data generated via ```create-raw-ag-data```. |
| ```extract-wvg-stats``` | Extracts key statistics and validates the Hoeffding bounds based on the data generated via ```create-raw-wvg-data```. |

> [!IMPORTANT]
> The mean squared error comparisons do ```iters_per_T``` runs per $T$ to average the mean squared errors at any given $T$. When executing *SRS-LSS without WarmUp*, it is not guaranteed that the algorithm runs successfully (compare **Proposition 4** in the paper). Thus, for any $T$, we require at least ```iters_per_T / 2``` successful executions for the average mean squared error to be shown in the final figure.

## Citation

If you use this code, please cite:

```
@Article{foundations6020018,
    AUTHOR = {Pollmann, Tim and Staudacher, Jochen},
    TITLE = {On Least Squares Approximations of Shapley Values and Applications to Interpretable Machine Learning},
    JOURNAL = {Foundations},
    VOLUME = {6},
    YEAR = {2026},
    NUMBER = {2},
    ARTICLE-NUMBER = {18},
    URL = {https://www.mdpi.com/2673-9321/6/2/18},
    ISSN = {2673-9321},
    ABSTRACT = {The Shapley value is the predominant point-valued solution concept in cooperative game theory and has recently become a foundational method in interpretable machine learning. In this domain, a prevailing strategy for circumventing the computational intractability of exact Shapley values is to approximate them via a weighted least squares optimization framework. In this paper, we investigate an existing algorithmic framework for weighted least squares Shapley approximation, assessing its feasibility for feature attribution. Methodologically, we conduct a theoretical variance analysis within a Monte Carlo sampling framework, investigate an approach for sample reuse across strata, and establish a relation to Unbiased KernelSHAP. Our analysis reveals three main findings: (i) a structural equivalence between least squares sampling and Unbiased KernelSHAP; (ii) the non-zero covariance between sampled coalitions introduced by reusing samples across strata in one of the existing least squares-based approaches; and (iii) the absence of a universally optimal sampling strategy across tasks. We validate these results empirically on several cooperative games and practical machine learning problems.},
    DOI = {10.3390/foundations6020018}
}
```
