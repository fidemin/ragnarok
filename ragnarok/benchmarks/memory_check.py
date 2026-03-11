import numpy as np
from memory_profiler import profile

from ragnarok.core.function.math import Square
from ragnarok.core.tensor import Tensor


@profile
def function_graph_with_large_data():
    for _ in range(1000):
        x = Tensor(np.random.randn(10000))
        y = Square()(Square()(x))


if __name__ == "__main__":
    function_graph_with_large_data()
