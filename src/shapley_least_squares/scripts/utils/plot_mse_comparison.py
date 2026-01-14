import matplotlib.pyplot as plt
import pandas as pd

from shapley_least_squares.scripts.utils.update_plt_params import update_plt_params


def plot_mse_comparison(experiment_name: str) -> None:
    update_plt_params()

    df = pd.read_csv(f"data/{experiment_name}.csv", index_col="T")

    for col in df.columns:
        plt.plot(df.index, df[col], label=col)

    plt.ticklabel_format(style="sci", axis="y", scilimits=(-1, 1))
    plt.xlabel(r"overall sample budget $T$")
    plt.ylabel(r"mean squared error of $\hat{\phi}$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"figures/{experiment_name}.png", dpi=600)
    plt.show()
