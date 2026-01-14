from math import comb

import matplotlib.pyplot as plt
import numpy as np

from shapley_least_squares.approx_algorithms.utils.shap_utils import (
    size_probs_based_on_shap_kernel,
)
from shapley_least_squares.scripts.utils.update_plt_params import update_plt_params

_N = 9


def default() -> None:
    experiment_name = "lss_vs_uks_sampling_probs_comparison"
    update_plt_params()

    size_probs_lss = [1 / (_N - 1)] * (_N - 1)
    assert np.sum(size_probs_lss) == 1.0
    p_S_lss = [
        size_prob / comb(_N, s) for s, size_prob in enumerate(size_probs_lss, start=1)
    ]
    plt.plot(range(1, _N), p_S_lss, marker="o", label="LSS")

    size_probs_uks = size_probs_based_on_shap_kernel(_N)[1:-1]
    assert np.sum(size_probs_uks) == 1.0
    p_S_uks = [
        size_prob / comb(_N, s) for s, size_prob in enumerate(size_probs_uks, start=1)
    ]
    plt.plot(range(1, _N), p_S_uks, marker="x", label="UKS")

    plt.ticklabel_format(style="sci", axis="y", scilimits=(-1, 1))
    plt.xlabel(r"subset size $s$")
    plt.ylabel(r"sampling probability $\mathbb{P}(\mathcal{S})$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"figures/{experiment_name}.png", dpi=600)
    plt.show()
