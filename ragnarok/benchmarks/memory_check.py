import numpy as np
from memory_profiler import profile

from ragnarok.core.function import Square
from ragnarok.core.variable import Variable


@profile
def function_graph_with_large_data():
    for _ in range(10):
        x = Variable(np.random.randn(100000))
        y = Square()(Square()(x))


if __name__ == '__main__':
    function_graph_with_large_data()
