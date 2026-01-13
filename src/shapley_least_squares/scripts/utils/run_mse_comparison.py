import click
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from shapley_least_squares.utils.interfaces import (
    ApproxAlgorithmInterface,
    GameInterface,
)


def run_mse_comparison(
    game: GameInterface,
    ground_truth_shapley_values: np.ndarray,
    algorithms: list[ApproxAlgorithmInterface],
    experiment_name: str,
    Ts: list[int],
    iters_per_T: int,
) -> None:
    click.echo(f'Starting experiment "{experiment_name}"...')

    mses_per_T = {algorithm.name(): [] for algorithm in algorithms}
    for T in Ts:
        click.echo(f"{T=}")

        mses_for_given_T = {algorithm.name(): [] for algorithm in algorithms}

        for _ in range(iters_per_T):
            for algorithm in algorithms:
                approximated_shapley_values = algorithm.run(game, T)

                if not np.isnan(approximated_shapley_values).any():
                    mses_for_given_T[algorithm.name()].append(
                        mean_squared_error(
                            ground_truth_shapley_values, approximated_shapley_values
                        )
                    )

        for algorithm in algorithms:
            # make sure that algorithms like S-MCIS have at least half the amount of executions in comparison to all
            # other algorithms
            if len(mses_for_given_T[algorithm.name()]) >= iters_per_T / 2:
                avg_mse_for_given_T = float(np.mean(mses_for_given_T[algorithm.name()]))
                mses_per_T[algorithm.name()].append(avg_mse_for_given_T)
            else:
                mses_per_T[algorithm.name()].append(np.nan)

    df = pd.DataFrame(mses_per_T, index=Ts)
    df.index.name = "T"
    df.to_csv(f"data/{experiment_name}.csv")

    click.echo(f'Finished experiment "{experiment_name}".')
