import matplotlib.pyplot as plt


def update_plt_params() -> None:
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times"],
            "axes.labelsize": 14,
            "font.size": 12,
            "legend.fontsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "text.latex.preamble": r"\usepackage{amsfonts} \usepackage{amsmath}",
        }
    )
