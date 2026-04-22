import click
import numpy as np
import pandas as pd

from shapley_least_squares.utils.interfaces import (
    ApproxAlgorithmInterface,
    GameInterface,
)


def run_algorithms(
    game: GameInterface,
    algorithms: list[ApproxAlgorithmInterface],
    experiment_name: str,
    T: int,
    n_iters: int,
) -> None:
    click.echo(f'Starting experiment "{experiment_name}"...')

    results = {algorithm.name(): [] for algorithm in algorithms}
    results["iteration"] = []

    for k in range(n_iters):
        click.echo(f"Iteration: {k + 1}/{n_iters}")
        results["iteration"].append(k)
        for algorithm in algorithms:
            approximated_shapley_values = algorithm.run(game, T)
            results[algorithm.name()].append(approximated_shapley_values)

    df = pd.DataFrame(results)

    algo_names = [alg.name() for alg in algorithms]
    df = df.explode(algo_names)

    df["player_id"] = np.tile(np.arange(game.n), n_iters)

    df.to_csv(f"data/{experiment_name}_raw.csv", index=False)
    click.echo(f"Results saved to data/{experiment_name}_raw.csv")
