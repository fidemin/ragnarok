import numpy as np

from ragnarok.core.tensor import Tensor
from ragnarok.graph.plot import plot_tensor_graph
from ragnarok.nn.function.activation import Tanh

if __name__ == "__main__":
    x = Tensor(np.array([1.0, 2.0]), name="x")
    y: Tensor = Tanh()(x)
    y.name = "y"
    y.backward(enable_double_backprop=True)

    for i in range(3):
        gx: Tensor = x.grad
        x.clear_grad()
        gx.backward(enable_double_backprop=True)

        plot_tensor_graph(
            gx, verbose=False, output_file=f"temp/tanh_{i+1}th.png", temp_dir="temp"
        )
