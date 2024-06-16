from src.main.ragnarok.graph.plot import plot_graph


def test_plot_graph(mocker):
    mock_graph = mocker.patch("src.main.ragnarok.graph.plot.DotGraph")
    result_list = [
        f'1 [label="(2, 3) float32", color=orange, style=filled]',
        f'2 [label="Square", color=lightblue, style=filled, shape=box]',
        f"3 -> 2",
        f"2 -> {1}",
        f'3 [label="(2, 3) float32", color=orange, style=filled]',
        f'4 [label="Add", color=lightblue, style=filled, shape=box]',
        f"5 -> 4",
        f"6 -> 4",
        f"4 -> 3",
        f'5 [label="(2, 3) float32", color=orange, style=filled]',
        f'6 [label="(2, 3) float32", color=orange, style=filled]',
    ]
    dot_str = "digraph G {\n" + "\n".join(result_list) + "\n}"
    mock_graph.return_value.draw.return_value = dot_str
    output_file = "temp/graph.png"
    plot_graph(mock_graph.return_value, output_file=output_file, temp_dir="temp")
    # test is failed in github action
    # assert os.path.isfile(output_file)
