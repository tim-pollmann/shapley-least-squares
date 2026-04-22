from shapley_least_squares.approx_algorithms.ks import KS
from shapley_least_squares.approx_algorithms.lss import LSS
from shapley_least_squares.approx_algorithms.s_lss import SLSS
from shapley_least_squares.approx_algorithms.srs_lss import SRSLSS_WarmUp
from shapley_least_squares.approx_algorithms.uks import UKS
from shapley_least_squares.games.airport_game import CustomAirportGameLarge
from shapley_least_squares.games.weighted_voting_game import (
    CustomWeightedVotingGameNormalSqrd,
)
from shapley_least_squares.scripts.utils.run_algorithms import (
    run_algorithms,
)
from shapley_least_squares.utils.interfaces import ApproxAlgorithmInterface

_ALGORITHMS: list[ApproxAlgorithmInterface] = [LSS, SLSS, SRSLSS_WarmUp, KS, UKS]
_T: int = 50000
_N_ITERS: int = 1000


def ag() -> None:
    experiment_name = "ag"
    game = CustomAirportGameLarge()

    run_algorithms(game, _ALGORITHMS, experiment_name, T=_T, n_iters=_N_ITERS)


def wvg() -> None:
    experiment_name = "wvg"
    game = CustomWeightedVotingGameNormalSqrd()
    run_algorithms(game, _ALGORITHMS, experiment_name, T=_T, n_iters=_N_ITERS)
