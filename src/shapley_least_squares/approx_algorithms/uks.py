from math import comb
from typing import override

import numpy as np

from shapley_least_squares.approx_algorithms.utils.check_number_of_samples_used import (
    check_number_of_samples_used,
)
from shapley_least_squares.approx_algorithms.utils.shap_utils import (
    harmonic_number,
    size_probs_based_on_shap_kernel,
    z,
)
from shapley_least_squares.exact_algorithms.utils.powerset_iterator import (
    powerset_iterator,
)
from shapley_least_squares.utils.interfaces import (
    ApproxAlgorithmInterface,
    GameInterface,
)


class UKS(ApproxAlgorithmInterface):
    @staticmethod
    @override
    def name() -> str:
        return "UKS"

    @staticmethod
    @override
    def run(game: GameInterface, T: int) -> np.ndarray:
        tau = T

        N = np.array(range(game.n))
        b = np.zeros(game.n)
        size_probs = size_probs_based_on_shap_kernel(game.n)

        n_samples_used = 0
        for _ in range(tau):
            sampled_size = np.random.choice(range(game.n + 1), p=size_probs)
            S = np.random.choice(N, size=sampled_size, replace=False)
            b += z(S, game.n) * game.v(S)
            n_samples_used += 1

        b /= tau
        A = UKS._compute_A(game.n)
        A_inv = np.linalg.inv(A)
        ones = np.ones(game.n)
        denominator = ones.T @ A_inv @ ones
        numerator = ones.T @ A_inv @ b - game.v(N)
        shapley_values = A_inv @ (b - ones * (numerator / denominator))

        check_number_of_samples_used(n_samples_used, T, UKS.name())

        return shapley_values

    @staticmethod
    @override
    def variance(game: GameInterface, T: int, true_values: np.ndarray) -> np.ndarray:
        tau = T

        n = game.n
        variances = np.zeros(n)
        H = harmonic_number(n - 1)

        for S in powerset_iterator(range(n)):
            s = len(S)

            if s == 0 or s == n:
                continue

            var_notin = 2 * s * H / (n * (n - s) * comb(n, s))
            var_in = (2 * H * n - 2 * s * H) / (comb(n, s) * s * n)

            v_S = game.v(S)

            for i in range(n):
                if i in S:
                    variances[i] += var_in * (v_S**2)
                elif i not in S:
                    variances[i] += var_notin * (v_S**2)

        N = np.array(range(game.n))
        return (variances - (true_values - game.v(N) / game.n) ** 2) / tau

    @staticmethod
    def _compute_A(n: int) -> np.ndarray:
        denominator = sum(1.0 / (k * (n - k)) for k in range(1, n))
        numerator = sum((k - 1) / float(n - k) for k in range(2, n))

        a_ij = numerator / (n * (n - 1) * denominator)
        a_ii = 0.5

        A = np.full((n, n), a_ij, dtype=float)
        np.fill_diagonal(A, a_ii)

        return A
