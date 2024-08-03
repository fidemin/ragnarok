import numpy as np

from src.main.ragnarok.core.variable import Variable
from src.main.ragnarok.graph.graph import DotGraph
from src.main.ragnarok.graph.plot import plot_graph


def goldstein_price(x, y):
    return (
        1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)
    ) * (
        30
        + (2 * x - 3 * y) ** 2
        * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2)
    )


x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = goldstein_price(x, y)

if __name__ == "__main__":
    z.backward()
    graph = DotGraph(z)
    plot_graph(
        graph, verbose=True, output_file="temp/goldstein_price.png", temp_dir="temp"
    )
