# distance calculation functions between graphs

import os

import torch
from neurosed import models


def graph_neurosed_distance_all(graphs, model: models.NormGEDModel):
    batch_size = len(graphs) * len(graphs)
    while True:
        try:
            res = model.predict_outer(graphs, graphs, batch_size=batch_size)
            break
        except RuntimeError as re:
            batch_size = batch_size // 2
    return res


def load_neurosed(original_graphs, neurosed_model_path, device):
    """
    Returns model and embed original graphs.

    :param original_graphs: PyG datasets
    :param neurosed_model_path: loading path of model
    :param device: cuda device to load, or 'cpu'
    :return:
    """
    if not os.path.exists(neurosed_model_path):
        raise FileNotFoundError(f'The neurosed model: {neurosed_model_path} is not found!')

    model = models.NormGEDModel(8, original_graphs[0].x.shape[1], 64, 64, device=device)
    model.load_state_dict(torch.load(neurosed_model_path, map_location=device))
    model.eval()
    model.embed_targets(original_graphs)

    return model
