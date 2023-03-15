# generate summary counterfactuals from candidates using greedy algorithm, and report coverage and cost performance for different size of summary set


from tqdm import tqdm
import torch

import util
from data import load_dataset
from gnn import load_trained_gnn, load_trained_prediction

import distance

import argparse


def parse_args():
    parser = argparse.ArgumentParser(prog='Global Graph Counterfactuals')

    # Dataset
    parser.add_argument('--dataset', help="Dataset flag", type=str, choices=['mutagenicity', 'aids', 'nci1', 'proteins'])
    parser.add_argument('--theta', type=float, default=0.1, help='distance threshold value during evaluation.')
    parser.add_argument('--device', type=str, help='Cuda device or cpu for gnn', default='0', choices=['0', '1', '2', '3', 'cpu'])

    return parser.parse_args()


def greedy_counterfactual_summary_from_covering_sets(counterfactual_covering, graphs_covered_by, k):
    """
    :param counterfactual_covering: Counterfactual -> Original graphs covered.
    :param graphs_covered_by: Original graphs -> counterfactuals that cover it.
    :param k: Number of counterfactuals in the summary.

    :return: List of indices of selected counterfactuals as summary, and the set of indices of the covered graphs.
    """

    # Greedily add the counterfactuals with maximum coverage in the remaining graphs.
    coverings = {}
    covered = set()

    # while len(indices) < k:
    for i in tqdm(range(1, k + 1)):
        counterfactual_index, covered_indices = max(counterfactual_covering.items(), key=lambda pair: len(pair[1]))
        covered.update(covered_indices)
        counterfactual_covering.pop(counterfactual_index)
        for covered_index in covered_indices:  # Update the mapping.
            for other_counterfactual_index in graphs_covered_by[covered_index] - {counterfactual_index}:
                if other_counterfactual_index in counterfactual_covering:
                    counterfactual_covering[other_counterfactual_index].remove(covered_index)

        coverings[i] = (counterfactual_index, len(covered))

    return coverings


def minimum_distance_cost_summary(distance_matrix, k):
    """
    cost reaching counterfactuals from original graphs with different number of counterfactuals.

    :param distance_matrix: n x c distance matrix where n is the number of original graphs, c is the number of counterfactual candidates
    :param k: Number of counterfactuals in the summary.
    :return selected counterfactuals and cost to reach them for different number of counterfactuals
    """
    costs = {}

    # distance_matrix = distance_matrix.detach().cpu().numpy()
    # min_state = np.array([distance_matrix.max() for _ in range(distance_matrix.shape[0])])
    min_state = torch.repeat_interleave(distance_matrix.max(), distance_matrix.shape[0])
    cost = min_state.sum()
    for i in tqdm(range(1, k + 1)):
        temp = min_state - distance_matrix[:, i - 1]
        gain = (temp * (temp > 0)).sum()
        # min_state = np.vstack([min_state, distance_matrix[:, i - 1]]).min(axis=0)
        min_state = torch.stack([min_state, distance_matrix[:, i - 1]]).min(dim=0).values
        cost = cost - gain
        costs[i] = (cost.item(), None, i - 1, min_state.clone())

    return costs


if __name__ == '__main__':
    args = parse_args()
    dataset_name = args.dataset
    device = 'cuda:' + args.device if torch.cuda.is_available() and args.device in ['0', '1', '2', '3'] else 'cpu'
    threshold_theta = args.theta

    # Load dataset
    graphs = load_dataset(dataset_name)

    # Load GNN model for dataset
    gnn_model = load_trained_gnn(dataset_name, device=device)
    gnn_model.eval()

    # Load prediction based on model
    preds = load_trained_prediction(dataset_name, device=device)
    preds = preds.cpu()

    pred_main_idx = torch.where(preds == 0)[0]
    pred_reverse_idx = torch.where(preds != 0)[0]

    original_graphs = graphs[pred_main_idx.tolist()]

    run_path = f'./results/{dataset_name}/runs/counterfactuals.pt'
    counterfactual_rw = torch.load(run_path)
    counterfactuals = []
    candidates = counterfactual_rw['counterfactual_candidates']

    i = 0
    while len(counterfactuals) < len(original_graphs) and len(candidates) > i:
        candidate = candidates[i]
        prediction_importance_value = candidate['importance_parts'][0]
        graph_hash = candidate['graph_hash']
        if prediction_importance_value >= 0.5:
            graph_can = counterfactual_rw['graph_map'][graph_hash]
            counterfactuals.append(graph_can)
        i += 1

    neurosed_model = distance.load_neurosed(original_graphs, neurosed_model_path=f'data/{dataset_name}/neurosed/best_model.pt', device=device)
    S = neurosed_model.predict_outer_with_queries(counterfactuals, batch_size=1000).cpu()
    original_graphs_element_counts = util.graph_element_counts(original_graphs)
    rw_graphs_element_counts = util.graph_element_counts(counterfactuals)
    s = torch.cartesian_prod(rw_graphs_element_counts, original_graphs_element_counts).sum(dim=1).view(len(rw_graphs_element_counts), len(original_graphs_element_counts))
    S = S / s
    S = S.T

    # Coverage
    print(f'Evaluation Coverage,  Threshold: {threshold_theta} running...')
    close_graphs = []
    for i in range(S.shape[0]):
        s = set(torch.where(S[i] <= threshold_theta)[0].tolist())  # select closer graphs based on threshold
        close_graphs.append(s)

    counterfactuals = set()
    for i in range(S.shape[0]):
        for close_graph in close_graphs[i]:
            counterfactuals.add(close_graph)

    # Create counterfactual mapping.
    counterfactual_covering = {i: set() for i in range(S.shape[1])}  # Counterfactual -> Original graphs covered.
    graphs_covered_by = {j: set() for j in range(S.shape[0])}  # Original graphs -> counterfactuals that cover it.
    idxs = torch.where(S <= threshold_theta)
    idxs = [(idxs[0][i].item(), idxs[1][i].item()) for i in range(idxs[0].shape[0])]
    for graph_idx, counterfactual_idx in idxs:
        counterfactual_covering[counterfactual_idx].add(graph_idx)
        graphs_covered_by[graph_idx].add(counterfactual_idx)

    coverings = greedy_counterfactual_summary_from_covering_sets(counterfactual_covering=counterfactual_covering,
                                                                 graphs_covered_by=graphs_covered_by,
                                                                 k=len(counterfactual_covering))

    x = []
    y = []

    for i in coverings:
        x.append(i)
        y.append(coverings[i][1] / S.shape[0])

    x_axis = range(1, S.shape[0] + 1)
    summary_from_coverage = [coverings[idx][0] for idx in coverings]

    for i in x_axis:
        if i in [1, 5, 10, 25, 50, 100]:
            print(f'Top {i}: {y[i - 1]}')

    # Cost based on Coverage Summary
    print('Calculating cost...')
    costs_from_summary = minimum_distance_cost_summary(S[:, summary_from_coverage], k=S.shape[1])

    y = []
    for c in costs_from_summary:
        state = costs_from_summary[c][3]
        y.append(torch.median(state).item())
    results = y

    for i in x_axis:
        if i in [1, 5, 10, 25, 50, 100]:
            print(f'Top {i}: {y[i - 1]}')
