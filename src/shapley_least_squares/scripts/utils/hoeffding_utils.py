from typing import Literal

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


def extract_stats(
    experiment_name: str,
    ground_truth_shapley_values: np.ndarray,
    player: int,
    save_result_as: Literal["csv", "latex"],
) -> None:
    df = pd.read_csv(f"data/{experiment_name}_raw.csv")

    algo_cols = [c for c in df.columns if c not in ["iteration", "player_id"]]
    stats_list = []

    for col in algo_cols:
        df_pivoted = df.pivot(index="iteration", columns="player_id", values=col)

        error_matrix = df_pivoted.values - ground_truth_shapley_values
        abs_error_matrix = np.abs(error_matrix)
        squared_error_matrix = error_matrix**2

        player_errors = abs_error_matrix[:, player]

        min_abs_errors = np.min(abs_error_matrix, axis=1)
        mean_abs_errors = np.mean(abs_error_matrix, axis=1)
        max_abs_errors = np.max(abs_error_matrix, axis=1)

        mean_squared_errors = np.mean(squared_error_matrix, axis=1)

        stats_list.append(
            {
                "Algorithm": col,
                "MinAbsPlayerError": player_errors.min(),
                "MeanAbsPlayerError": player_errors.mean(),
                "MaxAbsPlayerError": player_errors.max(),
                "MeanMinAbsError": min_abs_errors.mean(),
                "MeanMeanAbsError": mean_abs_errors.mean(),
                "MeanMaxAbsError": max_abs_errors.mean(),
                "MinMeanSquaredError": mean_squared_errors.min(),
                "MeanMeanSquaredError": mean_squared_errors.mean(),
                "MaxMeanSquaredError": mean_squared_errors.max(),
                "GlobalMinAbsError": abs_error_matrix.min(),
                "GlobalMaxAbsError": abs_error_matrix.max(),
            }
        )

    df = pd.DataFrame(stats_list).set_index("Algorithm")

    if save_result_as == "csv":
        df.to_csv(f"data/{experiment_name}_stats.csv")
        click.echo(f"Stats saved to data/{experiment_name}_stats.csv")
    elif save_result_as == "latex":
        latex_table = _create_latex_table(df, player)
        with open(f"tables/{experiment_name}_stats.tex", "w") as f:
            f.write(latex_table)
        click.echo(f"Stats saved to tables/{experiment_name}_stats.tex")
    else:
        raise ValueError(
            f"Invalid value for save_result_as: {save_result_as}. Expected 'csv' or 'latex'."
        )


def _create_latex_table(df: pd.DataFrame, player: int) -> str:
    df = df.drop(
        columns=[
            "GlobalMinAbsError",
            "GlobalMaxAbsError",
        ]
    )
    df = df.rename(
        columns={
            "MinAbsPlayerError": f"Min. Absolute Error of $i={player+1}$",
            "MeanAbsPlayerError": f"Mean Absolute Error of $i={player+1}$",
            "MaxAbsPlayerError": f"Max. Absolute Error of $i={player+1}$",
            "MeanMinAbsError": "Avg. Min. Absolute Error",
            "MeanMeanAbsError": "Avg. Mean Absolute Error",
            "MeanMaxAbsError": "Avg. Max. Absolute Error",
            "MinMeanSquaredError": "Min. Mean Squared Error",
            "MeanMeanSquaredError": "Avg. Mean Squared Error",
            "MaxMeanSquaredError": "Max. Mean Squared Error",
        }
    )
    df = df.T

    latex_string = df.to_latex(float_format="{:.2e}".format)

    lines = latex_string.split("\n")
    lines.insert(7, "\\midrule")
    lines.insert(11, "\\midrule")
    final_latex = "\n".join(lines)

    return final_latex
