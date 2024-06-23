import numpy as np

from src.main.ragnarok.core.function import Tanh
from src.main.ragnarok.core.variable import Variable
from src.main.ragnarok.graph.graph import DotGraph
from src.main.ragnarok.graph.plot import plot_graph

if __name__ == "__main__":
    x = Variable(np.array([1.0, 2.0]), name="x")
    y: Variable = Tanh()(x)
    y.name = "y"
    y.backward(enable_double_backprop=True)

    for i in range(3):
        gx: Variable = x.grad
        x.clear_grad()
        gx.backward(enable_double_backprop=True)

        graph = DotGraph(gx)
        plot_graph(
            graph, verbose=False, output_file=f"temp/tanh_{i+1}th.png", temp_dir="temp"
        )
