from math import comb
from typing import override

import numpy as np

from shapley_least_squares.approx_algorithms.utils.check_number_of_samples_used import (
    check_number_of_samples_used,
)
from shapley_least_squares.exact_algorithms.utils.powerset_iterator import (
    powerset_iterator,
)
from shapley_least_squares.utils.interfaces import (
    ApproxAlgorithmInterface,
    GameInterface,
)


class LSS(ApproxAlgorithmInterface):
    @staticmethod
    @override
    def name() -> str:
        return "LSS"

    @staticmethod
    @override
    def run(game: GameInterface, T: int) -> np.ndarray:
        tau = T

        N = np.array(range(game.n))
        shapley_values = np.array([game.v(N) / game.n] * game.n)
        tsc = 0

        for _ in range(tau):
            s = np.random.randint(1, game.n)
            S = np.random.choice(N, size=s, replace=False)
            v_S = game.v(S)
            tsc += 1
            w_in = (game.n - 1) / s / tau
            w_out = -(game.n - 1) / (game.n - s) / tau
            in_S = np.isin(N, S)
            shapley_values[in_S] += w_in * v_S
            shapley_values[~in_S] += w_out * v_S

        check_number_of_samples_used(tsc, T, LSS.name())
        return shapley_values

    @staticmethod
    @override
    def variance(game: GameInterface, T: int, true_values: np.ndarray) -> np.ndarray:
        tau = T

        n = game.n
        variances = np.zeros(n)

        for S in powerset_iterator(range(n)):
            s = len(S)

            if s == 0 or s == n:
                continue

            var_notin = s / (n * (n - s) * comb(n - 2, s - 1))
            var_in = (n - s) / (n * s * comb(n - 2, s - 1))

            v_S = game.v(S)

            for i in range(n):
                if i in S:
                    variances[i] += var_in * (v_S**2)
                elif i not in S:
                    variances[i] += var_notin * (v_S**2)

        N = np.array(range(game.n))
        return (variances - (true_values - game.v(N) / game.n) ** 2) / tau
