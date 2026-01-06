from typing import override

import numpy as np

from shapley_least_squares.approx_algorithms.utils.check_number_of_samples_used import (
    check_number_of_samples_used,
)
from shapley_least_squares.approx_algorithms.utils.shap_utils import (
    size_probs_based_on_shap_kernel,
    z,
)
from shapley_least_squares.utils.interfaces import (
    ApproxAlgorithmInterface,
    GameInterface,
)


class KS(ApproxAlgorithmInterface):
    @staticmethod
    @override
    def name() -> str:
        return "KS"

    @staticmethod
    @override
    def run(game: GameInterface, tau: int) -> np.ndarray:
        N = np.arange(game.n)
        A = np.zeros((game.n, game.n))
        b = np.zeros(game.n)
        size_probs = size_probs_based_on_shap_kernel(game.n)

        n_samples_used = 0
        for _ in range(tau - 1):
            sampled_size = np.random.choice(np.arange(game.n + 1), p=size_probs)
            S = np.random.choice(N, size=sampled_size, replace=False)
            z_vec = z(S, game.n)
            A += z_vec.reshape(-1, 1) @ z_vec.reshape(-1, 1).T
            b += z_vec * game.v(S)
            n_samples_used += 1

        A /= tau
        b /= tau

        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(
                "KernelSHAP linear system is singular: Matrix A could not be inverted. Try increasing m."
            ) from e

        ones = np.ones(game.n)
        denominator = ones.T @ A_inv @ ones
        numerator = ones.T @ A_inv @ b - game.v(N)
        n_samples_used += 1
        shapley_values = A_inv @ (b - ones * (numerator / denominator))

        check_number_of_samples_used(n_samples_used, tau, KS.name())

        return shapley_values

    @staticmethod
    @override
    def variance(game: GameInterface, tau: int, true_values: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
