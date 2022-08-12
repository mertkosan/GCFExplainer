import util
import os

import torch
from neurosed import models


def graph_neurosed_distance_all(graphs, model: models.NormGEDModel):
    # graphs_ = []
    # for i, graph in enumerate(graphs):
    # graphs_.append(Data(edge_index=graph.edge_index, x=Mutagenicity_v2.one_hot_from_label(graph.node_labels)))
    batch_size = len(graphs) * len(graphs)
    while True:
        try:
            res = model.predict_outer(graphs, graphs, batch_size=batch_size)
            break
        except RuntimeError as re:
            batch_size = batch_size // 2
    return res


def load_neurosed_and_distance_matrix(graphs, original_graphs, neurosed_model_path, distance_matrix_path, device, normalize=True):
    """
    Returns model and distance matrix if it is already learned, or evaluates with graphs.

    :param graphs: PyG datasets
    :param original_graphs: PyG datasets
    :param neurosed_model_path: loading path of model
    :param distance_matrix_path: saving or loading path of distance matrix
    :param device: cuda device to load, or 'cpu'
    :param normalize: normalize GED to n_GED
    :return:
    """
    if not os.path.exists(neurosed_model_path):
        raise FileNotFoundError(f'The neurosed model: {neurosed_model_path} is not found!')

    model = models.NormGEDModel(8, original_graphs[0].x.shape[1], 64, 64, device=device)
    model.load_state_dict(torch.load(neurosed_model_path, map_location=device))
    model.eval()
    model.embed_targets(original_graphs)

    if os.path.exists(distance_matrix_path):
        S = torch.load(distance_matrix_path, map_location=device)
    else:
        S = graph_neurosed_distance_all(graphs, model)
        torch.save(S.cpu(), distance_matrix_path)

    if normalize:
        gras_element_counts = util.graph_element_counts(graphs)
        s = torch.cartesian_prod(gras_element_counts, gras_element_counts).sum(dim=1).view(len(gras_element_counts), len(gras_element_counts)).to(device)
        S = S / s

    return S, model
