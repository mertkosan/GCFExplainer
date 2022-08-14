import copy

import scipy.sparse as sp
import numpy as np
import networkx as nx
import dgl
import grakel
import torch
# import igraph as ig
import torch_geometric.utils as torch_utils
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx, to_networkx
from data import Mutagenicity_v2


def adjacency_matrix(graph):
    """
    Return the adjacency matrix for the given graph.

    :param graph: Input graph.
    :return: Sparse adjacency matrix in scipy.sparse.csr_matrix format.
    """
    A = sp.csr_matrix(
        (graph.edge_weight.view(-1) if 'edge_weight' in graph else [1] * graph.num_edges, (graph.edge_index[0], graph.edge_index[1])),
        shape=(graph.num_nodes, graph.num_nodes)
    )
    return A


def pyg_to_networkx(graph):
    """
    Convert a pyg graph to a networkx graph with edge labels as of sorted tuple of node labels.

    :param graph: Input pyg graph.
    :return: Networkx graph.
    """
    # Convert from adjacency matrix.
    A = adjacency_matrix(graph)
    G = nx.from_scipy_sparse_matrix(A)

    # Add graph label.
    G.y = graph.y.item()

    # Add node labels.
    for i, label in enumerate(graph.node_labels):
        G.nodes[i]['label'] = label.item()

    # Add edge labels. Currently all edge labels set to 1.
    nx.set_edge_attributes(G, 1, 'label')

    return G


def pyg_to_dgl(graph):
    """
    Convert a pyg graph to a dgl graph.

    :param graph: Input pyg graph.
    :return: DGL graph and its label.
    """
    g = dgl.graph(tuple(graph.edge_index), num_nodes=graph.num_nodes)
    g.ndata['label'] = graph.node_labels

    return g, graph.y


def pyg_to_grakel(graph):
    """
    Convert a pyg graph to a grakel graph.

    :param graph: Input pyg graph.
    :return: grakel graph and its label.
    """
    G = pyg_to_networkx(graph)
    gras = grakel.utils.graph_from_networkx([G, ], node_labels_tag='label', edge_labels_tag='label', as_Graph=True)

    return next(gras), graph.y


def graph_to_grakel(graph):
    """
    Convert Mert's graph to a grakel graph.

    :param graph: Input Mert's graph.
    :return: grakel graph.
    """
    adj = graph['adj'].numpy()
    # node_labels = torch.argmax(graph['node_labels'], dim=1).numpy()
    node_labels = {index: label for index, label in enumerate(graph['node_labels'].numpy())}
    gra = grakel.Graph(adj, node_labels=node_labels)

    return gra


def dense_graph_to_sparse_graph(graph):
    """
    Convert Mert's dense graph to Mert's sparse graph.

    :param graph: Input Mert's dense graph.
    :return: Mert's sparse graph.
    """
    graph_ = deepcopy_graph(graph)
    graph_['adj'] = graph_['adj'].to_sparse()
    graph_['x'] = graph_['x'].to_sparse()
    return graph_


def sparse_graph_to_dense_graph(graph):
    """
    Convert Mert's sparse graph to Mert's dense graph.

    :param graph: Input Mert's sparse graph.
    :return: Mert's dense graph.
    """
    graph_ = deepcopy_graph(graph)
    graph_['adj'] = graph_['adj'].to_dense()
    graph_['x'] = graph_['x'].to_dense()
    return graph_


def pyg_to_graph(graph):
    """
    Convert a pyg graph to a Mert's graph.

    :param graph: Input pyg graph.
    :return: Mert's graph.
    """
    return {'adj': torch_utils.to_dense_adj(graph.edge_index, max_num_nodes=graph.num_nodes)[0], 'node_labels': graph.node_labels}


def deepcopy_graph(graph):
    """
    deepcopy by using clone
    :param graph:
    :return: deepcopied graph
    """
    return {
        'adj': graph['adj'].clone(),
        'x': graph['x'].clone()
    }


def graph_to_pyg(graph):
    """
    Convert a Mert's graph to a pyg graph with node features field x (one hot coded)

    :param graph: Input Mert's graph.
    :return: pyg graph.
    """
    edge_index, _ = torch_utils.dense_to_sparse(graph['adj'])
    # x = Mutagenicity_v2.one_hot_from_label(graph['node_labels'])


def graphs_to_pygs_x(dataset):
    """
    Convert a Mert's graphs to a pyg graphs with node features field x (one hot coded)

    :param dataset: Input Mert's graphs.
    :return: pyg graphs.
    """
    num_nodes = dataset[0]['adj'].shape[0]
    all_node_labels = Mutagenicity_v2.one_hot_from_label(torch.cat([graph['node_labels'] for graph in dataset]))
    xs = all_node_labels.view(-1, num_nodes, Mutagenicity_v2.num_classes())

    gras = []
    for i, graph in enumerate(dataset):
        edge_index, _ = torch_utils.dense_to_sparse(graph['adj'])
        gras.append(Data(edge_index=edge_index, x=xs[i]))

    return gras


# def graph_to_igraph(graph):
#     """
#     Convert a Mert's graph to a igraph graph.
#
#     :param graph: Input Mert's graph.
#     :return: Igraph.
#     """
#     adj = graph['adj'].numpy()
#     # node_labels = torch.argmax(graph['node_labels'], dim=1).numpy()
#     g = ig.Graph.Adjacency(adj)
#     g.vs['label'] = graph['node_labels'].tolist()
#
#     return g


def graph_to_networkx(graph):
    """
    Convert Mert's graph to a networkx graph.

    :param graph: Mert's graph.
    :return: networkx graph.
    """

    A = graph['adj'].numpy()
    G = nx.from_numpy_array(A)

    # Add node labels.
    for i, label in enumerate(graph['node_labels']):
        G.nodes[i]['label'] = label.item()

    return G


def graph_to_gg(graph):
    """
    Convert Mert's graph to v2 (adj and one-hot node labels).

    :param graph: Mert's graph
    :return: Mert's graph v2.
    """

    return {
        'adj': graph['adj'],
        'x': Mutagenicity_v2.one_hot_from_label(graph['node_labels'])
    }


def pyg_to_gg(graph):
    """
    Convert a pyg graph to a Mert's graph v2.

    :param graph: Input pyg graph.
    :return: Mert's graph v2.
    """
    return {
        'adj': torch_utils.to_dense_adj(graph.edge_index, max_num_nodes=graph.num_nodes)[0],
        'x': Mutagenicity_v2.one_hot_from_label(graph.node_labels)
    }


def gg_to_pyg(gg):
    """
    Convert  Mert's graph v2 to a pyg graph.

    :param gg: Mert's graph v2.
    :return: a pyg graph
    """
    edge_index, _ = torch_utils.dense_to_sparse(gg['adj'])
    return Data(edge_index=edge_index, x=gg['x'])


def gg_to_networkx(gg):
    """
    Convert Mert's graph v2 to a networkx graph.

    :param gg: Mert's graph v2.
    :return: networkx graph.
    """

    A = gg['adj'].numpy()
    G = nx.from_numpy_array(A)

    # Add node labels.
    for i, label in enumerate(Mutagenicity_v2.label_from_one_hot(Mutagenicity_v2.label_from_one_hot(gg['node_labels']))):
        G.nodes[i]['label'] = label.item()

    return G


def largest_components_from_pyg(graph):
    G_nx = to_networkx(graph, to_undirected=True)
    subgraph = sorted(nx.connected_components(G_nx), key=len, reverse=True)[0]
    G_nx = nx.subgraph(G_nx, subgraph)
    new_graph = from_networkx(G_nx)
    return Data(edge_index=new_graph.edge_index, node_labels=graph.node_labels[sorted(subgraph)], pred=graph.pred)


def is_pyg_connected(graph):
    G_nx = to_networkx(graph, to_undirected=True)
    return nx.is_connected(G_nx)


def graph_element_counts(dataset):
    return torch.Tensor([graph.num_nodes + graph.num_edges / 2 for graph in dataset])
