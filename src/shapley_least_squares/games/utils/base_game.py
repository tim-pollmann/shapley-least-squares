from typing import override

from shapley_least_squares.utils.interfaces import GameInterface


class BaseGame(GameInterface):
    def __init__(self, n: int) -> None:
        if n < 2:
            raise ValueError("n must be >= 2.")

        self._n = n

    @property
    @override
    def n(self) -> int:
        return self._n
