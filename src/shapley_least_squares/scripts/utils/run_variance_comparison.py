import click
import numpy as np
import pandas as pd

from shapley_least_squares.exact_algorithms.brute_force_calculation_via_sum import (
    brute_force_calculation_via_sum,
)
from shapley_least_squares.games.weighted_voting_game import WeightedVotingGame
from shapley_least_squares.utils.interfaces import ApproxAlgorithmInterface

_WEIGHTS = [1, 2, 3, 1, 1, 1, 1]
_GAMES = {
    quota + 1: WeightedVotingGame(_WEIGHTS, quota + 1) for quota in range(sum(_WEIGHTS))
}


def run_variance_comparison(
    algorithms: list[ApproxAlgorithmInterface],
    experiment_name: str,
    tau: int,
    player: int,
    n_iters: int,
) -> None:
    click.echo(f'Starting experiment "{experiment_name}"...')

    records: list[dict] = []

    for quota, game in _GAMES.items():
        click.echo(f"{quota=}")

        ground_truth_shapley_values = brute_force_calculation_via_sum(game)

        for algorithm in algorithms:
            theoretical_variance = algorithm.variance(
                game, tau, ground_truth_shapley_values
            )

            records.append(
                {
                    "quota": quota,
                    "algorithm": algorithm.name(),
                    "variance_type": "theoretical",
                    "variance": float(theoretical_variance[player]),
                }
            )

            empirical_variances = []

            for _ in range(n_iters):
                approximated_shapley_values = algorithm.run(game, tau)

                if not np.isnan(approximated_shapley_values).any():
                    empirical_variances.append(
                        (
                            approximated_shapley_values[player]
                            - ground_truth_shapley_values[player]
                        )
                        ** 2
                    )

            records.append(
                {
                    "quota": quota,
                    "algorithm": algorithm.name(),
                    "variance_type": "empirical",
                    "variance": float(np.mean(np.array(empirical_variances))),
                }
            )

    df = pd.DataFrame.from_records(records)
    df.to_csv(f"data/{experiment_name}.csv", index=False)

    click.echo(f'Finished experiment "{experiment_name}".')
