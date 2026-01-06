from collections.abc import Iterator
from itertools import chain, combinations

import numpy as np


def powerset_iterator(input_set: list[int] | np.ndarray) -> Iterator[np.ndarray]:
    return chain.from_iterable(
        (np.array(c, dtype=int) for c in combinations(input_set, r))
        for r in range(len(input_set) + 1)
    )
