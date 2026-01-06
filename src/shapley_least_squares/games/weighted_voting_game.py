from typing import override

import numpy as np

from shapley_least_squares.games.utils.base_game import BaseGame


class WeightedVotingGame(BaseGame):
    def __init__(self, weights: list[int] | np.ndarray, quota: int) -> None:
        if not all(weight > 0 for weight in weights):
            raise ValueError("All weights must be positive.")

        if sum(weights) < quota:
            raise ValueError("The sum of all weights must be >= quota.")

        super().__init__(len(weights))
        self._weights = np.array(weights)
        self._quota = quota

    @override
    def v(self, S: list[int] | np.ndarray) -> float:
        return float(np.sum(self._weights[S]) >= self._quota)


class CustomWeightedVotingGame(WeightedVotingGame):
    def __init__(self) -> None:
        self.shapley_values = np.array(
            [
                0.0195633,
                0.00758229,
                0.0315659,
                0.0183738,
                0.0172237,
                0.0203783,
                0.0126878,
                0.0280294,
                0.0146914,
                0.0164787,
                0.0285206,
                0.0247204,
                0.0265837,
                0.0176027,
                0.0229297,
                0.0251793,
                0.0120538,
                0.0207923,
                0.0331522,
                0.0164787,
                0.0111343,
                0.0229297,
                0.0176027,
                0.0187659,
                0.026111,
                0.00758229,
                0.0130112,
                0.00834009,
                0.0126878,
                0.0187659,
                0.0150403,
                0.0233705,
                0.0256429,
                0.0242659,
                0.0275429,
                0.00886581,
                0.0295169,
                0.0191624,
                0.0146914,
                0.016849,
                0.023816,
                0.0285206,
                0.0105422,
                0.0399336,
                0.0300221,
                0.00808335,
                0.0179861,
                0.0176027,
                0.0358923,
                0.0111343,
            ]
        )

        super().__init__(
            weights=[
                9801,
                3844,
                15625,
                9216,
                8649,
                10201,
                6400,
                13924,
                7396,
                8281,
                14161,
                12321,
                13225,
                8836,
                11449,
                12544,
                6084,
                10404,
                16384,
                8281,
                5625,
                11449,
                8836,
                9409,
                12996,
                3844,
                6561,
                4225,
                6400,
                9409,
                7569,
                11664,
                12769,
                12100,
                13689,
                4489,
                14641,
                9604,
                7396,
                8464,
                11881,
                14161,
                5329,
                19600,
                14884,
                4096,
                9025,
                8836,
                17689,
                5625,
            ],
            quota=249646,
        )
