import numpy as np

from src.main.ragnarok.core.variable import Variable
from src.main.ragnarok.graph.graph import DotGraph
from src.main.ragnarok.graph.plot import plot_graph
from src.main.ragnarok.nn.function.loss import MeanSquaredError
from src.main.ragnarok.nn.layer.activation import ReLU
from src.main.ragnarok.nn.layer.linear import Linear
from src.main.ragnarok.nn.model.model import Sequential
from src.main.ragnarok.nn.optimizer.optimizer import SGD

if __name__ == "__main__":
    layer1 = Linear(8)
    layer2 = ReLU()
    layer3 = Linear(4, 8, use_bias=False)

    model = Sequential([layer1, layer2, layer3])

    loss_func = MeanSquaredError()
    optimizer = SGD(lr=0.1)

    x = Variable(np.random.randn(10, 8))
    t = Variable(np.random.randn(10, 4))

    epoch = 10000
    for i in range(epoch):
        for param in model.params.values():
            param.clear_grad()

        y = model.forward(x)
        loss = loss_func(y, t)

        loss.backward()

        optimizer.update(model.params.values())

    graph = DotGraph(loss)
    plot_graph(
        graph, verbose=True, output_file="temp/two_layer_net.png", temp_dir="temp"
    )
    print(f"Loss: {loss.data}")
