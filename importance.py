import os.path
import torch
import util
import numpy as np
import distance
from torch_geometric.data import Data, Batch, DataLoader


def prepare_and_get(graphs, gnn_model, original_graph_indices, theta, importance_function_name, device1, device2, dataset_name):
    original_graphs = graphs[original_graph_indices.tolist()]
    neurosed_folder = f'data/{dataset_name}/neurosed'
    if not os.path.exists(neurosed_folder):
        os.makedirs(neurosed_folder)

    neurosed_model_path = os.path.join(neurosed_folder, 'best_model.pt')
    neurosed_model = distance.load_neurosed(original_graphs, neurosed_model_path=neurosed_model_path, device=device2)

    original_graphs_elements_counts = util.graph_element_counts(original_graphs)

    return {
        'gnn_model': gnn_model,
        'neurosed_model': neurosed_model,
        'original_graphs': original_graphs,
        'original_graphs_element_counts': original_graphs_elements_counts,
        'importance_function': importance_function_name,
        'distance_threshold': theta,
        'gnn_device': device1,
        'neurosed_device': device2
    }


def call(graphs, wargs):
    try:
        preds, graph_embeddings = prediction(wargs['gnn_model'], Batch.from_data_list(graphs).to(wargs['gnn_device']))
        preds = preds.cpu().numpy()
        graph_embeddings = graph_embeddings.cpu().numpy()
    except RuntimeError as re:
        loader = DataLoader(graphs, batch_size=128)
        preds, graph_embeddings = [], []
        for batch in loader:
            pred, graph_embedding = prediction(wargs['gnn_model'], batch.to(wargs['gnn_device']))
            preds.append(pred)
            graph_embeddings.append(graph_embedding)
        preds = torch.cat(preds).cpu().numpy()
        graph_embeddings = torch.cat(graph_embeddings).cpu().numpy()

    torch.cuda.set_device(wargs['gnn_device'])
    torch.cuda.empty_cache()

    coverage = np.ones(shape=preds.shape)  # .to(preds.device)

    coverage_matrix = neurosed_threshold_coverage_estimation(wargs['neurosed_model'], graphs, wargs['original_graphs_element_counts'], wargs['distance_threshold'])
    coverage_matrix = coverage_matrix.cpu()

    torch.cuda.set_device(wargs['neurosed_model'].device)
    torch.cuda.empty_cache()

    return np.stack([preds, coverage]).T, graph_embeddings, coverage_matrix


@torch.no_grad()
def prediction(model, graphs):
    node_embeddings, graph_embeddings, preds = model(graphs)  # .to(model.device))
    preds = torch.exp(preds)
    return preds[:, [1]].sum(axis=1), graph_embeddings


@torch.no_grad()
def neurosed_threshold_coverage_estimation(neurosed_model, dataset, original_graphs_element_counts, threshold):
    gras_element_counts = util.graph_element_counts(dataset)
    batch_size = len(dataset)
    while True:
        try:
            d = neurosed_model.predict_outer_with_queries(dataset, batch_size=batch_size).cpu()
            break
        except RuntimeError as e:
            batch_size = batch_size // 2

    s = torch.cartesian_prod(gras_element_counts, original_graphs_element_counts).sum(dim=1).view(len(dataset), len(original_graphs_element_counts))
    d = d / s
    selected = d <= threshold

    return selected.float()
