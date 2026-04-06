from shapley_least_squares.approx_algorithms.lss import LSS
from shapley_least_squares.approx_algorithms.s_lss import SLSS
from shapley_least_squares.approx_algorithms.uks import UKS
from shapley_least_squares.games.airport_game import CustomAirportGameLarge
from shapley_least_squares.games.weighted_voting_game import (
    CustomWeightedVotingGameNormalSqrd,
)
from shapley_least_squares.scripts.utils.hoeffding_utils import (
    extract_stats,
    run_algorithms,
)
from shapley_least_squares.utils.interfaces import ApproxAlgorithmInterface

_ALGORITHMS: list[ApproxAlgorithmInterface] = [
    LSS,
    UKS,
    SLSS,
]
_T: int = 50000
_N_ITERS: int = 1000
_PLAYER: int = 0


def ag() -> None:
    experiment_name = "ag_hoeffding"
    game = CustomAirportGameLarge()

    run_algorithms(game, _ALGORITHMS, experiment_name, T=_T, n_iters=_N_ITERS)
    extract_stats(experiment_name, game.shapley_values, _PLAYER, save_result_as="csv")
    extract_stats(experiment_name, game.shapley_values, _PLAYER, save_result_as="latex")


def wvg() -> None:
    experiment_name = "wvg_hoeffding"
    game = CustomWeightedVotingGameNormalSqrd()

    run_algorithms(game, _ALGORITHMS, experiment_name, T=_T, n_iters=_N_ITERS)
    extract_stats(experiment_name, game.shapley_values, _PLAYER, save_result_as="csv")
    extract_stats(experiment_name, game.shapley_values, _PLAYER, save_result_as="latex")
