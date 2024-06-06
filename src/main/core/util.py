import numpy as np


def clip_grads(grads: list[np.ndarray], max_norm: float):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(np.square(grad))

    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-7)
    if rate < 1:
        for grad in grads:
            grad *= rate
