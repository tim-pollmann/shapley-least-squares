from typing import override

import numpy as np

from shapley_least_squares.games.utils.base_game import BaseGame


class BankruptcyGame(BaseGame):
    def __init__(self, claims: list[int] | np.ndarray, E: float) -> None:
        if not all(claim > 0 for claim in claims):
            raise ValueError("All claims must be positive.")

        if sum(claims) <= E:
            raise ValueError("The sum of all claims must be > E.")

        super().__init__(len(claims))
        self._N = np.arange(len(claims))
        self._claims = np.array(claims)
        self._E = E

    @override
    def v(self, S: list[int] | np.ndarray) -> float:
        indices = np.setdiff1d(self._N, S)
        return float(max(0, self._E - sum(self._claims[indices])))
