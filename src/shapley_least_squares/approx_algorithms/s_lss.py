from itertools import product
from math import comb
from typing import override

import numpy as np

from shapley_least_squares.approx_algorithms.utils.check_number_of_samples_used import (
    check_number_of_samples_used,
)
from shapley_least_squares.approx_algorithms.utils.sample_subset_including_player import (
    sample_subset_including_player,
)
from shapley_least_squares.exact_algorithms.utils.powerset_iterator import (
    powerset_iterator,
)
from shapley_least_squares.utils.interfaces import (
    ApproxAlgorithmInterface,
    GameInterface,
)


class SLSS(ApproxAlgorithmInterface):
    @staticmethod
    @override
    def name() -> str:
        return "S-LSS"

    @staticmethod
    @override
    def run(game: GameInterface, tau: int) -> np.ndarray:
        n = game.n
        N = np.array(range(n))
        v_N = game.v(N)
        tsc = 0
        a_i_m_map = {i: 0.0 for i in N}

        tau_per_stratum = {
            (j, s): int(np.ceil(2 * s * tau / (game.n * game.n * (game.n - 1))))
            for j, s in product(range(game.n), range(1, game.n))
        }
        for i in N:
            for s in range(1, n):
                for _ in range(tau_per_stratum[(i, s)]):
                    a_i_m_map[i] += (
                        1
                        / (tau_per_stratum[(i, s)] * (n - s))
                        * game.v(sample_subset_including_player(N, s, i))
                    )
                    tsc += 1

        shapley_values = np.array(
            [
                v_N / n
                + ((n - 1) / n) * a_i_m_map[i]
                - (1 / n) * sum(a_i_m_map[j] for j in N if j != i)
                for i in N
            ]
        )

        check_number_of_samples_used(
            tsc,
            tau,
            SLSS.name(),
            max_deviation=game.n * game.n * (game.n - 1),
        )
        return shapley_values

    @staticmethod
    @override
    def variance(
        game: GameInterface,
        tau: int,
        true_values: np.ndarray,
    ) -> np.ndarray:
        n = game.n
        variances = np.zeros(n)

        tau_per_stratum = {
            (j, s): int(np.ceil(2 * s * tau / (game.n * game.n * (game.n - 1))))
            for j, s in product(range(game.n), range(1, game.n))
        }

        def _var_a_is(
            game: GameInterface, i: int, s: int, tau_of_stratum: int
        ) -> float:
            def _m(s: int) -> float:
                return 1 / ((game.n - 1) * comb(game.n - 2, s - 1))

            def true_a_is(game: GameInterface, i: int, s: int) -> float:
                N = list(range(game.n))
                result = 0.0
                for subset in powerset_iterator(N):
                    if (
                        len(subset) == s
                        and i in subset
                        and len(subset) > 0
                        and len(subset) < game.n
                    ):
                        result += _m(s) * game.v(subset)
                return result

            N = list(range(game.n))
            result = 0.0
            for subset in powerset_iterator(N):
                if (
                    len(subset) == s
                    and i in subset
                    and len(subset) > 0
                    and len(subset) < game.n
                ):
                    result += (
                        game.v(subset) ** 2
                        / (game.n - s)
                        / (game.n - 1)
                        / comb(game.n - 2, s - 1)
                    )

            result -= true_a_is(game, i, s) ** 2
            return result / tau_of_stratum

        for i in range(game.n):
            variances[i] = ((n - 1) / game.n) ** 2 * sum(
                _var_a_is(game, i, s, tau_per_stratum[(i, s)]) for s in range(1, game.n)
            ) + 1 / (game.n**2) * sum(
                sum(
                    _var_a_is(game, j, s, tau_per_stratum[(j, s)])
                    for s in range(1, game.n)
                )
                for j in range(game.n)
                if j != i
            )

        return variances
