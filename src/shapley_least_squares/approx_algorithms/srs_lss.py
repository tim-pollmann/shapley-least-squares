from __future__ import annotations

import random
from typing import override

import numpy as np

from shapley_least_squares.approx_algorithms.utils.check_number_of_samples_used import (
    check_number_of_samples_used,
)
from shapley_least_squares.utils.interfaces import (
    ApproxAlgorithmInterface,
    GameInterface,
)


class SRSLSS(ApproxAlgorithmInterface):
    @staticmethod
    @override
    def name() -> str:
        return "SRS-LSS"

    @staticmethod
    @override
    def run(game: GameInterface, T: int) -> np.ndarray:
        n = game.n
        N = np.array(range(n))
        v_N = game.v(N)
        a_i_map = {i: 0.0 for i in N}

        tau_s = int(np.ceil(T / (n - 1)))
        n_samples_used = 0
        for s in range(1, n):
            tau_map = {i: 0 for i in N}
            a_is_map = {i: 0.0 for i in N}

            for _ in range(tau_s):
                S = random.sample(N.tolist(), s)
                v_S = game.v(S)
                n_samples_used += 1
                for i in N:
                    if i in S:
                        a_is_map[i] += v_S
                        tau_map[i] += 1

            for i in N:
                if tau_map[i] == 0:
                    return np.full(n, np.nan)

                a_i_map[i] += a_is_map[i] / (tau_map[i] * (n - s))

        shapley_values = np.array(
            [
                v_N / n
                + ((n - 1) / n) * a_i_map[i]
                - (1 / n) * sum(a_i_map[j] for j in N if j != i)
                for i in N
            ]
        )

        check_number_of_samples_used(
            n_samples_used,
            T,
            SRSLSS.name(),
            max_deviation=n - 1,
        )
        return shapley_values

    @staticmethod
    @override
    def variance(game: GameInterface, T: int, true_values: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
