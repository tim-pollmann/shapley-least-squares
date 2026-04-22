import click
import numpy as np
import pandas as pd


def _calculate_distributions(array: np.ndarray, prefix: str) -> dict[str, float]:
    mean = np.mean(array)
    std = np.std(array, ddof=1)
    q1, q3 = np.percentile(array, [25, 75])
    low95, up95 = np.percentile(array, [2.5, 97.5])

    return {
        f"{prefix}_Mean": mean,
        f"{prefix}_Min": np.min(array),
        f"{prefix}_Max": np.max(array),
        f"{prefix}_Std": std,
        f"{prefix}_CV": std / mean if mean != 0 else np.nan,
        f"{prefix}_IQR": q3 - q1,
        f"{prefix}_Lower95": low95,
        f"{prefix}_Upper95": up95,
    }


def extract_stats(
    raw_file: str,
    stats_file: str,
    ground_truth_shapley_values: np.ndarray,
    player: int,
) -> None:
    df_raw = pd.read_csv(f"data/{raw_file}.csv")
    algo_cols = [c for c in df_raw.columns if c not in ["iteration", "player_id"]]
    results = []

    for col in algo_cols:
        df_pivoted = df_raw.pivot(index="iteration", columns="player_id", values=col)
        error_matrix = df_pivoted.values - ground_truth_shapley_values
        abs_error_matrix = np.abs(error_matrix)
        squared_error_matrix = error_matrix**2

        player_abs_errors = abs_error_matrix[:, player]
        min_abs_errors = np.min(abs_error_matrix, axis=1)
        mean_abs_errors = np.mean(abs_error_matrix, axis=1)
        max_abs_errors = np.max(abs_error_matrix, axis=1)
        mean_squared_errors = np.mean(squared_error_matrix, axis=1)

        results.append(
            {
                "Algorithm": col,
                "PlayerAbs_Min": player_abs_errors.min(),
                "PlayerAbs_Mean": player_abs_errors.mean(),
                "PlayerAbs_Max": player_abs_errors.max(),
                "MinAbs_Mean": min_abs_errors.mean(),
                "MeanAbs_Mean": mean_abs_errors.mean(),
                **_calculate_distributions(max_abs_errors, prefix="MaxAbs"),
                **_calculate_distributions(mean_squared_errors, prefix="MSE"),
            }
        )

    df_stats = pd.DataFrame(results).set_index("Algorithm")
    df_stats.to_csv(f"data/{stats_file}.csv")
    click.echo(f"Stats successfully saved to data/{stats_file}.csv")
