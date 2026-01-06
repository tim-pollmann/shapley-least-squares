from typing import override

import numpy as np

from shapley_least_squares.games.utils.base_game import BaseGame


class AirportGame(BaseGame):
    def __init__(self, costs: list[int] | np.ndarray) -> None:
        if not all(cost > 0 for cost in costs):
            raise ValueError("All costs must be positive.")

        super().__init__(len(costs))
        self._costs = np.array(costs)

    @override
    def v(self, S: list[int] | np.ndarray) -> float:
        if len(S) == 0:
            return 0.0

        return float(np.max(self._costs[S]))

    @property
    def shapley_values(self) -> np.ndarray:
        return self._calculate_shapley_values()

    def _calculate_shapley_values(self) -> np.ndarray:
        costs = np.array(self._costs, dtype=float)

        order = np.argsort(costs)
        sorted_lengths = costs[order]

        extended = np.concatenate(([0], sorted_lengths))

        deltas = np.diff(extended)

        shapley_values_sorted = np.zeros(self._n)
        for i in range(self._n):
            for k in range(i + 1):
                shapley_values_sorted[i] += deltas[k] / (self._n - k)

        shapley_values = np.zeros(self._n)
        shapley_values[order] = shapley_values_sorted

        return shapley_values


class CustomAirportGameLarge(AirportGame):
    def __init__(self) -> None:
        costs = (
            [1] * 8
            + [2] * 12
            + [3] * 6
            + [4] * 14
            + [5] * 8
            + [6] * 9
            + [7] * 13
            + [8] * 10
            + [9] * 10
            + [10] * 10
        )
        super().__init__(costs)


class CustomAirportGameSmall(AirportGame):
    def __init__(self) -> None:
        costs = [1, 2, 3, 7, 2, 1, 2, 3, 4, 4, 4, 2, 1, 2, 3, 1, 1, 1, 8, 2, 2, 3, 4]
        super().__init__(costs)
