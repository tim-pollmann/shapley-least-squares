from shapley_least_squares.approx_algorithms.lss import LSS
from shapley_least_squares.approx_algorithms.s_lss import SLSS
from shapley_least_squares.approx_algorithms.srs_lss import SRSLSS
from shapley_least_squares.approx_algorithms.uks import UKS
from shapley_least_squares.exact_algorithms.brute_force_calculation_via_sum import (
    brute_force_calculation_via_sum,
)
from shapley_least_squares.games.airport_game import CustomAirportGameLarge
from shapley_least_squares.games.explainability_game import (
    DiabetesGBRGame,
    HousingMLPGame,
    WineRFGame,
)
from shapley_least_squares.games.weighted_voting_game import CustomWeightedVotingGame
from shapley_least_squares.scripts.utils.plot_mse_comparison import plot_mse_comparison
from shapley_least_squares.scripts.utils.run_mse_comparison import run_mse_comparison
from shapley_least_squares.utils.interfaces import ApproxAlgorithmInterface

_ALGORITHMS: list[ApproxAlgorithmInterface] = [LSS, SLSS, SRSLSS, UKS]
_TAUS_SYNTHETIC_GAMES = [30000, 40000, 50000, 60000, 70000, 80000, 100000]
_TAUS_EXPLAINABILITY_GAMES = [10000, 15000, 20000, 25000, 30000, 40000, 50000]
_ITERS_PER_TAU = 250


def ag() -> None:
    experiment_name = "ag"
    game = CustomAirportGameLarge()
    ground_truth_shapley_values = game.shapley_values

    run_mse_comparison(
        game,
        ground_truth_shapley_values,
        _ALGORITHMS,
        experiment_name,
        _TAUS_SYNTHETIC_GAMES,
        _ITERS_PER_TAU,
    )

    plot_mse_comparison(experiment_name)


def wvg() -> None:
    experiment_name = "wvg"
    game = CustomWeightedVotingGame()
    ground_truth_shapley_values = game.shapley_values

    run_mse_comparison(
        game,
        ground_truth_shapley_values,
        _ALGORITHMS,
        experiment_name,
        _TAUS_SYNTHETIC_GAMES,
        _ITERS_PER_TAU,
    )

    plot_mse_comparison(experiment_name)


def diabetes() -> None:
    experiment_name = "diabetes"
    game = DiabetesGBRGame()
    ground_truth_shapley_values = brute_force_calculation_via_sum(game)

    run_mse_comparison(
        game,
        ground_truth_shapley_values,
        _ALGORITHMS,
        experiment_name,
        _TAUS_EXPLAINABILITY_GAMES,
        _ITERS_PER_TAU,
    )

    plot_mse_comparison(experiment_name)


def housing() -> None:
    experiment_name = "housing"
    game = HousingMLPGame()
    ground_truth_shapley_values = brute_force_calculation_via_sum(game)

    run_mse_comparison(
        game,
        ground_truth_shapley_values,
        _ALGORITHMS,
        experiment_name,
        _TAUS_EXPLAINABILITY_GAMES,
        _ITERS_PER_TAU,
    )

    plot_mse_comparison(experiment_name)


def wine() -> None:
    experiment_name = "wine"
    game = WineRFGame()
    ground_truth_shapley_values = brute_force_calculation_via_sum(game)

    run_mse_comparison(
        game,
        ground_truth_shapley_values,
        _ALGORITHMS,
        experiment_name,
        _TAUS_EXPLAINABILITY_GAMES,
        _ITERS_PER_TAU,
    )

    plot_mse_comparison(experiment_name)
