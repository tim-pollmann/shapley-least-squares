from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class GameInterface(ABC):
    @property
    @abstractmethod
    def n(self) -> int:
        pass

    @abstractmethod
    def v(self, S: list[int] | np.ndarray) -> float:
        pass


class ApproxAlgorithmInterface:
    @staticmethod
    @abstractmethod
    def name() -> str:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def run(game: GameInterface, T: int, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def variance(game: GameInterface, T: int, true_values: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
