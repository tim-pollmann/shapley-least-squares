import random

import numpy as np


def sample_subset_including_player(
    N: list[int] | np.ndarray, s: int, i: int
) -> list[int]:
    if i not in N:
        raise ValueError("required_value must be in the list")

    if s < 1:
        raise ValueError("Subset size must be at least 1")

    if s > len(N):
        raise ValueError("Subset size cannot be larger than list size")

    remaining = [j for j in N if j != i]
    sampled = random.sample(remaining, s - 1)
    sampled.append(i)
    return sampled
