from __future__ import annotations

import random
from typing import override

import numpy as np

from shapley_least_squares.approx_algorithms.utils.check_number_of_samples_used import (
    check_number_of_samples_used,
)
from shapley_least_squares.utils.interfaces import (
    ApproxAlgorithmInterface,
    GameInterface,
)


class SRSLSS(ApproxAlgorithmInterface):
    @staticmethod
    @override
    def name() -> str:
        return "SRS-LSS"

    @staticmethod
    @override
    def run(game: GameInterface, T: int) -> np.ndarray:
        return _srs_lss(game, T, with_warmup=False)

    @staticmethod
    @override
    def variance(game: GameInterface, T: int, true_values: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class SRSLSS_Warmup(ApproxAlgorithmInterface):
    @staticmethod
    @override
    def name() -> str:
        return "SRS-LSS (Warmup)"

    @staticmethod
    @override
    def run(game: GameInterface, T: int) -> np.ndarray:
        return _srs_lss(game, T, with_warmup=True)

    @staticmethod
    @override
    def variance(game: GameInterface, T: int, true_values: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


def _srs_lss(game: GameInterface, T: int, with_warmup: bool) -> np.ndarray:
    n = game.n
    N = np.array(range(n))
    v_N = game.v(N)

    if with_warmup:
        a_si_map, n_samples_used = _srs_lss_warmup(game)
    else:
        a_si_map, n_samples_used = {s: np.zeros(n) for s in range(1, n)}, 0

    assert n_samples_used < T

    a_i_arr = np.zeros(n)
    tau_s = int(np.ceil((T - n_samples_used) / (n - 1)))

    for s in range(1, n):
        tau_arr = np.ones(n) if with_warmup else np.zeros(n)
        a_is_arr = a_si_map[s]

        for _ in range(tau_s):
            S = random.sample(N.tolist(), s)
            v_S = game.v(S)
            n_samples_used += 1

            for i in S:
                a_is_arr[i] += v_S
                tau_arr[i] += 1

        if (tau_arr == 0).any():
            return np.full(n, np.nan)

        for i in N:
            a_i_arr[i] += a_is_arr[i] / (tau_arr[i] * (n - s))

    a_i_sum = np.sum(a_i_arr)
    shapley_values = a_i_arr + (v_N - a_i_sum) / n

    check_number_of_samples_used(
        n_samples_used,
        T,
        SRSLSS_Warmup.name() if with_warmup else SRSLSS.name(),
        max_deviation=n - 1,
    )

    return shapley_values


def _srs_lss_warmup(game: GameInterface) -> tuple[dict[int, np.ndarray], int]:
    n = game.n
    N = np.array(range(n))

    a_si_map = {s: np.zeros(n) for s in range(1, n)}

    n_samples_used = 0

    for s in range(1, n):
        rand_perm = np.random.permutation(N)
        K = np.ceil(n / s).astype(int)

        for k in range(K):
            if n % s == 0 or k < K - 1:
                S = rand_perm[k * s : (k + 1) * s]
                v_S = game.v(S)
            else:
                S = rand_perm[k * s : n]
                S_filler = np.random.choice(
                    rand_perm[: k * s], s - (n % s), replace=False
                )
                v_S = game.v(np.concatenate((S, S_filler)))

            n_samples_used += 1
            for i in S:
                a_si_map[s][i] = v_S

    return a_si_map, n_samples_used
