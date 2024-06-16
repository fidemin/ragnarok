import os
import subprocess

from src.main.ragnarok.graph.graph import DotGraph


def plot_graph(
    graph: DotGraph, verbose: bool = True, output_file="graph.png", temp_dir=None
):
    graph_str = graph.draw(verbose=verbose)

    graph_file_path = "temp_graph.dot"

    if not temp_dir:
        temp_dir = os.path.join(os.path.expanduser("~"), "temp")

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    graph_file_path = os.path.join(temp_dir, graph_file_path)

    with open(graph_file_path, "w") as f:
        f.write(graph_str)

    extension = output_file.split(".")[-1]
    cmd = f"dot {graph_file_path} -T {extension} -o {output_file}"
    subprocess.run(cmd, shell=True)
