import numpy as np

from src.main.ragnarok.core.tensor import Tensor
from src.main.ragnarok.graph.graph import DotGraph
from src.main.ragnarok.graph.plot import plot_graph
from src.main.ragnarok.nn.function.activation import Tanh

if __name__ == "__main__":
    x = Tensor(np.array([1.0, 2.0]), name="x")
    y: Tensor = Tanh()(x)
    y.name = "y"
    y.backward(enable_double_backprop=True)

    for i in range(3):
        gx: Tensor = x.grad
        x.clear_grad()
        gx.backward(enable_double_backprop=True)

        graph = DotGraph(gx)
        plot_graph(
            graph, verbose=False, output_file=f"temp/tanh_{i+1}th.png", temp_dir="temp"
        )
