# utility functions


import torch


def graph_element_counts(dataset):
    return torch.Tensor([graph.num_nodes + graph.num_edges / 2 for graph in dataset])
