from shapley_least_squares.approx_algorithms.lss import LSS
from shapley_least_squares.approx_algorithms.s_lss import SLSS
from shapley_least_squares.approx_algorithms.uks import UKS
from shapley_least_squares.scripts.utils.plot_variance_comparison import (
    plot_variance_comparison,
)
from shapley_least_squares.scripts.utils.run_variance_comparison import (
    run_variance_comparison,
)
from shapley_least_squares.utils.interfaces import ApproxAlgorithmInterface

_ALGORITHMS: list[ApproxAlgorithmInterface] = [UKS, LSS, SLSS]
_T = 20000
_PLAYER = 0
_N_ITERS = 5000


def default() -> None:
    experiment_name = "variance_comparison"

    run_variance_comparison(_ALGORITHMS, experiment_name, _T, _PLAYER, _N_ITERS)

    plot_variance_comparison(experiment_name, _PLAYER)
