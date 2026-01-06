from math import factorial

import click
import numpy as np

from shapley_least_squares.exact_algorithms.utils.powerset_iterator import (
    powerset_iterator,
)
from shapley_least_squares.utils.interfaces import GameInterface


def brute_force_calculation_via_sum(game: GameInterface) -> np.ndarray[float]:
    N = np.array(range(game.n))

    with click.progressbar(
        range(game.n), label="Computing Shapley values"
    ) as progressbar:
        return np.array(
            [
                sum(
                    factorial(len(S))
                    * factorial(game.n - len(S) - 1)
                    / factorial(game.n)
                    * (game.v(np.append(S, i)) - game.v(S))
                    for S in powerset_iterator(N[i != N])
                )
                for i in progressbar
            ]
        )
