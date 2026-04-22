from shapley_least_squares.games.airport_game import CustomAirportGameLarge
from shapley_least_squares.games.weighted_voting_game import (
    CustomWeightedVotingGameNormalSqrd,
)
from shapley_least_squares.scripts.utils.extract_stats import extract_stats

_PLAYER: int = 0


def ag() -> None:
    extract_stats(
        raw_file="ag_raw",
        stats_file="ag_stats",
        ground_truth_shapley_values=CustomAirportGameLarge().shapley_values,
        player=_PLAYER,
    )


def wvg() -> None:
    extract_stats(
        raw_file="wvg_raw",
        stats_file="wvg_stats",
        ground_truth_shapley_values=CustomWeightedVotingGameNormalSqrd().shapley_values,
        player=_PLAYER,
    )
