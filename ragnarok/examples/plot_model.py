import numpy as np
from keras.api.datasets import mnist

from ragnarok.core.tensor import Tensor
from ragnarok.graph.plot import plot_tensor_graph
from ragnarok.nn.model.model import MLP

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    y_train = np.eye(10)[y_train].astype("float32")
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0
    y_test = np.eye(10)[y_test].astype("float32")

    model = MLP(out_sizes=[1000, 10], activation="sigmoid")
    y = model(Tensor(x_train[:1]))

    plot_tensor_graph(y, output_file="temp/mlp_model_graph.png", temp_dir="temp")
