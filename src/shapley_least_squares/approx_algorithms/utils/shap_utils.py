import numpy as np


def size_probs_based_on_shap_kernel(n: int) -> np.ndarray:
    size_probs = np.array(
        [1.0 / (s * (n - s)) if s != 0 and s != n else 0.0 for s in range(n + 1)]
    )
    size_probs /= np.sum(size_probs)
    return size_probs


def z(S: np.ndarray, n: int) -> np.ndarray:
    result = np.zeros(n, dtype=int)
    result[S] = 1
    return result


def harmonic_number(n: int) -> float:
    return sum(1.0 / k for k in range(1, n + 1))
