import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from shapley_least_squares.scripts.utils.update_plt_params import update_plt_params


def plot_variance_comparison(experiment_name: str, player: int) -> None:
    update_plt_params()

    df = pd.read_csv(f"data/{experiment_name}.csv")

    algorithm_colors: dict[str, str] = {}

    for algorithm, df_algorithms in df.groupby("algorithm"):
        emp = df_algorithms[df_algorithms["variance_type"] == "empirical"].sort_values(
            "quota"
        )

        (line_emp,) = plt.plot(
            emp["quota"],
            emp["variance"],
            label=f"{algorithm} (empirical)",
            linestyle="None",
            marker=".",
            markersize=10,
        )

        algorithm_colors[algorithm] = line_emp.get_color()

    signature_to_algorithms: dict[tuple, list[str]] = {}

    for algorithm, df_algorithms in df.groupby("algorithm"):
        theor = df_algorithms[
            df_algorithms["variance_type"] == "theoretical"
        ].sort_values("quota")
        signature = tuple(np.round(theor["variance"].to_numpy(), decimals=12))

        if signature not in signature_to_algorithms:
            signature_to_algorithms[signature] = []

        signature_to_algorithms[signature].append(algorithm)

    plotted_signatures = set()
    shared_color = "black"

    for signature, algorithms in signature_to_algorithms.items():
        if signature in plotted_signatures:
            continue
        plotted_signatures.add(signature)

        quota_values = df[
            (df["algorithm"] == algorithms[0]) & (df["variance_type"] == "theoretical")
        ].sort_values("quota")["quota"]
        variance_values = np.array(signature)

        if len(algorithms) > 1:
            color = shared_color
            label = f"{', '.join(algorithms)} (theoretical)"
        else:
            algorithm = algorithms[0]
            color = algorithm_colors.get(algorithm, shared_color)
            label = f"{algorithm} (theoretical)"

        plt.plot(
            quota_values,
            variance_values,
            linestyle="None",
            marker="x",
            markersize=10,
            color=color,
            label=label,
        )

    plt.ticklabel_format(style="sci", axis="y", scilimits=(-1, 1))
    plt.xlabel(r"quota $C$")
    plt.ylabel(r"variance of $\hat{\phi}_" + f"{player+1}$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"figures/{experiment_name}.png", dpi=600)
    plt.show()
