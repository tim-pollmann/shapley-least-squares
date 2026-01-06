from typing import override

import numpy as np

from shapley_least_squares.games.utils.base_game import BaseGame


class GloveGame(BaseGame):
    def __init__(self, L: set[int], R: set[int]) -> None:
        if not L.isdisjoint(R):
            raise ValueError("L and R must be disjoint sets.")

        if set(range(len(L) + len(R))) != set(L) | set(R):
            raise ValueError(
                "Player indices must form a complete, consecutive set starting at 0."
            )

        super().__init__(len(L) + len(R))
        self._L = np.array(list(L))
        self._R = np.array(list(R))

    @override
    def v(self, S: list[int] | np.ndarray) -> float:
        count_L = np.isin(S, self._L).sum()
        count_R = np.isin(S, self._R).sum()
        return float(min(count_L, count_R))
