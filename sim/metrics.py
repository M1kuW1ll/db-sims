import numpy as np

def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    total = np.sum(x)
    if total == 0 or n == 0:
        return 0.0
    sorted_x = np.sort(x)
    raw = (2 * np.sum(np.arange(1, n + 1) * sorted_x)) / (n * total) - (n + 1) / n
    return float(np.clip(raw, 0.0, 1.0))


def entropy(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n <= 1:
        return 0.0
    total = np.sum(x)
    if total == 0:
        return 0.0
    probs = x / total
    probs = probs[probs > 0]
    raw = -np.sum(probs * np.log(probs)) / np.log(n)
    return float(np.clip(raw, 0.0, 1.0))


def hhi(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    total = np.sum(x)
    if total == 0:
        return 0.0
    shares = x / total
    return float(np.clip(np.sum(shares ** 2), 0.0, 1.0))
