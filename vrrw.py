import os

import torch
import util
import torch_geometric.utils as torch_utils
import networkx as nx
import random

from tqdm import tqdm
import argparse
import importance

import numpy as np
from data import load_dataset
from gnn import load_trained_gnn, load_trained_prediction

graph_map = {}  # graph_hash -> {edge_index, x}
graph_index_map = {}  # graph hash -> index in counterfactual_graphs
counterfactual_candidates = []  # [{frequency: int, graph_hash: str, importance_parts: tuple, input_graphs_covering_indexes: set}]
input_graphs_covered = []  # [int] with of number of input graphs
covering_graphs = set()  # dictionary graph hash which is in first #number input graph counterfactual list (i.e., contributing input_graph_covered)
transitions = {}  # graph_hash -> {transitions ([hashes], [actions], [importance_parts], tensor(input_graph_covering_for_all_neighbours))}

MAX_COUNTERFACTUAL_SIZE = 0
starting_step = 1

traversed_hashes = []  # list of traversed graph hashes


def get_args():
    parser = argparse.ArgumentParser(description='Graph Global Counterfactual Summary')
    parser.add_argument('--dataset', type=str, default='mutagenicity', choices=['mutagenicity', 'aids', 'nci1', 'proteins'])
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha value to balance individual and cumulative coverage')
    parser.add_argument('--theta', type=float, default=0.05, help='distance threshold value during training.')
    parser.add_argument('--teleport', type=float, default=0.1, help='teleport probability to input graphs')
    parser.add_argument('--max_steps', type=int, default=50000, help='random walk step size')
    parser.add_argument('--k', type=int, default=100000, help='number of graphs will be selected from counterfactuals')
    parser.add_argument('--device1', type=str, help='Cuda device or cpu for gnn model', default='0')
    parser.add_argument('--device2', type=str, help='Cuda device or cpu for neurosed model', default='0')
    parser.add_argument('--sample_size', type=int, help='Sample count for neighbour graphs', default=10000)
    parser.add_argument('--sample', action='store_true')
    return parser.parse_args()


def calculate_hash(graph_embedding):
    if isinstance(graph_embedding, (np.ndarray,)):
        return hash(graph_embedding.tobytes())
    else:
        raise Exception('graph_embedding should be ndarray')


def node_label_change(graph):
    neighbor_graphs_actions = []
    neighbor_graphs = []
    for i in range(graph.x.shape[0]):
        for j in range(graph.x.shape[1]):
            # if graph['node_labels'][i] != j:
            if graph.x[i, j] != 1:
                neighbor_graph_action = ('NLC', i, j)
                neighbor_graphs_actions.append(neighbor_graph_action)
                neighbor_graphs.append(neighbor_graph_access(graph, neighbor_graph_action))
    return neighbor_graphs_actions, neighbor_graphs


def node_addition(graph):
    neighbor_graphs_actions = []
    neighbor_graphs = []
    for i in range(graph.x.shape[0]):
        for j in range(graph.x.shape[1]):  # Add a new node with label j connected with node i.
            neighbor_graph_action = ('NA', i, j)
            neighbor_graphs_actions.append(neighbor_graph_action)
            neighbor_graphs.append(neighbor_graph_access(graph, neighbor_graph_action))
    return neighbor_graphs_actions, neighbor_graphs


def isolated_node_addition(graph):
    neighbor_graphs_actions = []
    neighbor_graphs = []
    for j in range(graph.x.shape[1]):  # Add a new isolated node with label j
        neighbor_graph_action = ('INA', j, j)
        neighbor_graphs_actions.append(neighbor_graph_action)
        neighbor_graphs.append(neighbor_graph_access(graph, neighbor_graph_action))
    return neighbor_graphs_actions, neighbor_graphs


def node_removal(graph):
    degree = torch_utils.degree(graph.edge_index[0], num_nodes=graph.num_nodes)
    neighbor_graphs_actions = []
    neighbor_graphs = []
    for i in range(len(degree)):
        if degree[i] == 1:  # Remove nodes with exactly one edge only.
            neighbor_graph_action = ('NR', i, i)
            neighbor_graphs_actions.append(neighbor_graph_action)
            neighbor_graphs.append(neighbor_graph_access(graph, neighbor_graph_action))
    return neighbor_graphs_actions, neighbor_graphs


def isolated_node_removal(graph):
    degree = torch_utils.degree(graph.edge_index[0], num_nodes=graph.num_nodes)
    neighbor_graphs_actions = []
    neighbor_graphs = []
    for i in range(len(degree)):
        if degree[i] == 0:  # Remove isolated nodes only.
            neighbor_graph_action = ('INR', i, i)
            neighbor_graphs_actions.append(neighbor_graph_action)
            neighbor_graphs.append(neighbor_graph_access(graph, neighbor_graph_action))
    return neighbor_graphs_actions, neighbor_graphs


def edge_change(graph, keep_bridge=True, only_removal=False):
    nxg = torch_utils.to_networkx(graph, to_undirected=True)  # 157 µs ± 71.9 µs per loop
    bridges = set(nx.bridges(nxg)) if keep_bridge else set()  # 556 µs ± 31.2 µs per loop
    num_nodes = graph.x.shape[0]
    neighbor_graphs_actions = []
    neighbor_graphs = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if nxg.has_edge(i, j):
                if keep_bridge and (i, j) not in bridges:  # edge exist and its removal does not disconnect the graph
                    neighbor_graph_action = ('ER', i, j)
                else:  # remove edge regardlessly
                    neighbor_graph_action = ('ERR', i, j)
                neighbor_graphs_actions.append(neighbor_graph_action)
                neighbor_graphs.append(neighbor_graph_access(graph, neighbor_graph_action))
            elif not nxg.has_edge(i, j) and not only_removal:  # add edges
                neighbor_graph_action = ('EA', i, j)
                neighbor_graphs_actions.append(neighbor_graph_action)
                neighbor_graphs.append(neighbor_graph_access(graph, neighbor_graph_action))
    return neighbor_graphs_actions, neighbor_graphs


def neighbor_graph_access(graph, neighbor_graph_action):
    neighbor_graph = graph.clone()
    action = neighbor_graph_action[0]
    if action == 'NOTHING':
        neighbor_graph = neighbor_graph
    elif action == 'NLC':  # node label change
        _, i, j = neighbor_graph_action
        neighbor_graph.x[i] = 0  # 6.93 µs ± 301 ns per loop
        neighbor_graph.x[i][j] = 1  # 7.9 µs ± 420 ns per loop
    elif action == 'NA':  # node addition
        _, i, j = neighbor_graph_action
        neighbor_graph.num_nodes += 1
        neighbor_graph.edge_index = torch.hstack([graph.edge_index, torch.tensor([[i, graph.num_nodes], [graph.num_nodes, i]])])  # 14.1 µs ± 57.3 ns per loop, 3 times faster than padder.
        neighbor_graph.x = torch.vstack([graph.x, torch.nn.functional.one_hot(torch.tensor(j), graph.x.shape[1])])  # 36.8 µs ± 340 ns per loop, similar to padder.
    elif action == 'INA':  # isolated node addition.
        _, i, j = neighbor_graph_action
        neighbor_graph.num_nodes += 1
        neighbor_graph.x = torch.vstack([graph.x, torch.nn.functional.one_hot(torch.tensor(j), graph.x.shape[1])])  # 36.8 µs ± 340 ns per loop, similar to padder.
    elif action in ('NR', 'INR'):  # (isolated) node removal
        _, i, j = neighbor_graph_action
        indices = torch.LongTensor(list(range(i)) + list(range(i + 1, graph.num_nodes)))  # 4.93 µs ± 244 ns per loop
        neighbor_graph.num_nodes -= 1
        neighbor_graph.edge_index = torch_utils.subgraph(indices, graph.edge_index, relabel_nodes=True, num_nodes=graph.num_nodes)[0]  # 80.5 µs ± 992 ns per loop
        neighbor_graph.x = neighbor_graph.x[indices]  # 7.44 µs ± 149 ns per loop
    elif action in ('ER', 'ERR'):  # edge removal (regardlessly)
        _, i, j = neighbor_graph_action
        neighbor_graph.edge_index = graph.edge_index[:, ~((graph.edge_index[0] == i) & (graph.edge_index[1] == j) | (graph.edge_index[0] == j) & (graph.edge_index[1] == i))]  # 78.9 µs ± 1.87 µs per loop
    elif action == 'EA':  # edge addition
        _, i, j = neighbor_graph_action
        neighbor_graph.edge_index = torch.hstack([graph.edge_index, torch.tensor([[i, j], [j, i]])])  # 14 µs ± 262 ns per loop
    else:
        raise NotImplementedError(f'Neighbor edit action {action} not supported. ')
    return neighbor_graph


def is_counterfactual_array_full():
    return len(counterfactual_candidates) >= MAX_COUNTERFACTUAL_SIZE


def get_minimum_frequency():
    return counterfactual_candidates[-1]['frequency']


def is_graph_counterfactual(graph_hash):
    return counterfactual_candidates[graph_index_map[graph_hash]]['importance_parts'][0] >= 0.5


def reorder_counterfactual_candidates(start_idx):
    swap_idx = start_idx - 1
    while swap_idx >= 0 and counterfactual_candidates[start_idx]['frequency'] > counterfactual_candidates[swap_idx]['frequency']:
        swap_idx -= 1
    swap_idx += 1
    if swap_idx < start_idx:
        graph_index_map[counterfactual_candidates[start_idx]['graph_hash']] = swap_idx
        graph_index_map[counterfactual_candidates[swap_idx]['graph_hash']] = start_idx
        counterfactual_candidates[start_idx], counterfactual_candidates[swap_idx] = counterfactual_candidates[swap_idx], counterfactual_candidates[start_idx]
    return swap_idx


def update_input_graphs_covered(add_graph_covering_list=None, remove_graph_covering_list=None):
    global input_graphs_covered
    if add_graph_covering_list is not None:
        input_graphs_covered += add_graph_covering_list
    if remove_graph_covering_list is not None:
        input_graphs_covered -= remove_graph_covering_list


def check_reinforcement_condition(graph_hash):
    return is_graph_counterfactual(graph_hash)


def populate_counterfactual_candidates(graph_hash, importance_parts, input_graphs_covering_list):
    is_new_graph = False
    if graph_hash in graph_index_map:
        graph_idx = graph_index_map[graph_hash]
        condition = check_reinforcement_condition(graph_hash)
        if condition:
            counterfactual_candidates[graph_idx]['frequency'] += 1
            swap_idx = reorder_counterfactual_candidates(graph_idx)
        else:
            swap_idx = graph_idx
    else:
        is_new_graph = True
        if is_counterfactual_array_full():
            deleting_graph_hash = counterfactual_candidates[-1]['graph_hash']
            del graph_index_map[deleting_graph_hash]
            del graph_map[deleting_graph_hash]
            if deleting_graph_hash in transitions:
                del transitions[deleting_graph_hash]
            counterfactual_candidates[-1] = {
                "frequency": get_minimum_frequency() + 1,
                "graph_hash": graph_hash,
                "importance_parts": importance_parts,
                "input_graphs_covering_list": input_graphs_covering_list
            }
        else:
            counterfactual_candidates.append({
                'frequency': 2,
                'graph_hash': graph_hash,
                "importance_parts": importance_parts,
                "input_graphs_covering_list": input_graphs_covering_list
            })
        graph_idx = len(counterfactual_candidates) - 1
        graph_index_map[graph_hash] = graph_idx
        swap_idx = reorder_counterfactual_candidates(graph_idx)

    # updating input_graphs_covered entries
    if swap_idx == graph_idx:  # no swap
        if is_new_graph and graph_idx < len(input_graphs_covered) and is_graph_counterfactual(graph_hash):
            update_input_graphs_covered(add_graph_covering_list=input_graphs_covering_list)
            covering_graphs.add(graph_hash)
    else:  # swapped graph_idx position has swapped graph now
        swapped_graph = counterfactual_candidates[graph_idx]
        if is_graph_counterfactual(swapped_graph['graph_hash']) and graph_idx >= len(input_graphs_covered) > swap_idx:
            update_input_graphs_covered(remove_graph_covering_list=swapped_graph['input_graphs_covering_list'])
            covering_graphs.remove(swapped_graph['graph_hash'])
        if is_new_graph:
            if is_graph_counterfactual(graph_hash) and swap_idx < len(input_graphs_covered):
                update_input_graphs_covered(add_graph_covering_list=input_graphs_covering_list)
                covering_graphs.add(graph_hash)
        else:
            if is_graph_counterfactual(graph_hash) and swap_idx < len(input_graphs_covered) <= graph_idx:
                update_input_graphs_covered(add_graph_covering_list=input_graphs_covering_list)
                covering_graphs.add(graph_hash)


def calculate_importance(hashes, importances, coverage_matrices, importance_args):
    cum_coverage = None
    ind_coverage = None
    if importance_args['alpha'] < 1:
        # cumulative coverage
        belong = torch.Tensor([hash_ in covering_graphs for hash_ in hashes])
        support = coverage_matrices.to_dense() + (coverage_matrices.to_dense().T * belong).T - input_graphs_covered
        cum_selected = torch.maximum(torch.zeros(input_graphs_covered.shape), support)
        cum_coverage = cum_selected.sum(dim=1) / input_graphs_covered.shape[0]
        cum_coverage = cum_coverage.numpy()
        importances[:, 1] = cum_coverage
    if importance_args['alpha'] > 0:
        # individual coverage
        ind_coverage = coverage_matrices.to_dense().sum(dim=1) / input_graphs_covered.shape[0]
        ind_coverage = ind_coverage.numpy()
        importances[:, 1] = ind_coverage

    importances[:, 1] = alpha * ind_coverage + (1 - alpha) * cum_coverage
    if importances[:, 1].sum() == 0:  # all coverage values are zero, we will only use prediction as importance
        importance_values = importances[:, 0]
    else:
        importance_values = np.prod(importances, axis=1)
    return importance_values


def move_from_known_graph(hashes, importances, coverage_matrices, importance_args):
    probabilities = []
    importance_values = calculate_importance(hashes, np.array(importances), coverage_matrices, importance_args)
    for i, hash_i in enumerate(hashes):
        importance_value = importance_values[i]
        if hash_i in graph_index_map:  # and is_graph_counterfactual(hash_i):  # reinforcing only seen counterfactuals
            frequency = counterfactual_candidates[graph_index_map[hash_i]]['frequency']
        else:
            frequency = get_minimum_frequency() if is_counterfactual_array_full() else 1
        probabilities.append(importance_value * frequency)

    if sum(probabilities) == 0:  # if probability values are all 0, we assign equal probs to all transitions
        probabilities = np.ones(len(probabilities)) / len(probabilities)
    else:
        probabilities = np.array(probabilities) / sum(probabilities)
    selected_hash_idx = random.choices(range(len(hashes)), weights=probabilities)[0]
    return selected_hash_idx


def move_to_next_graph(graph_hash, importance_args, teleport_probability):
    graph = graph_map[graph_hash]
    not_teleport = False
    if random.uniform(0, 1) < teleport_probability:  # teleport to start
        return None, not not_teleport
    else:
        if graph_hash in transitions:
            target_graphs_hashes, target_graphs_actions, target_graphs_importance_parts, target_graphs_coverage_matrix = transitions[graph_hash]
            selected_hash_idx = move_from_known_graph(target_graphs_hashes, target_graphs_importance_parts, target_graphs_coverage_matrix, importance_args)
        else:  # uncalculated transitions for a graph
            neighbor_graphs_actions_edge_change, neighbor_graphs_edge_change = edge_change(graph, keep_bridge=True, only_removal=False)  # still n nodes
            neighbor_graphs_actions_node_label_change, neighbor_graphs_node_label_change = node_label_change(graph)  # still n nodes
            neighbor_graphs_actions_node_addition, neighbor_graphs_node_addition = node_addition(graph)  # n+1 nodes
            neighbor_graphs_actions_node_removal, neighbor_graphs_node_removal = node_removal(graph)  # n-1 nodes

            neighbor_graphs_actions = neighbor_graphs_actions_edge_change + neighbor_graphs_actions_node_label_change + neighbor_graphs_actions_node_addition + neighbor_graphs_actions_node_removal
            all_graph_set = neighbor_graphs_edge_change + neighbor_graphs_node_label_change + neighbor_graphs_node_addition + neighbor_graphs_node_removal

            if sample_size < len(neighbor_graphs_actions) and is_sample:
                samples = random.sample(range(len(neighbor_graphs_actions)), sample_size)
                neighbor_graphs_actions = [neighbor_graphs_actions[sample] for sample in samples]
                all_graph_set = [all_graph_set[sample] for sample in samples]

            neighbor_graphs_importance_parts, neighbor_graphs_embeddings, neighbor_graphs_coverage_matrix = importance.call(all_graph_set, importance_args)

            target_graphs_set = {graph_hash}
            target_graphs_hashes = [graph_hash]
            target_graphs_actions = [("NOTHING", None, None)]
            target_graphs_importance_parts = [counterfactual_candidates[graph_index_map[graph_hash]]['importance_parts']]
            needed_i = []
            for i in range(len(neighbor_graphs_embeddings)):
                graph_neighbour_hash = calculate_hash(neighbor_graphs_embeddings[i])
                if graph_neighbour_hash not in target_graphs_set:
                    needed_i.append(i)
                    target_graphs_importance_parts.append(neighbor_graphs_importance_parts[i])
                    target_graphs_hashes.append(graph_neighbour_hash)
                    target_graphs_set.add(graph_neighbour_hash)
                    target_graphs_actions.append(neighbor_graphs_actions[i])
            target_graphs_coverage_matrix = torch.cat([counterfactual_candidates[graph_index_map[graph_hash]]['input_graphs_covering_list'].unsqueeze(0), neighbor_graphs_coverage_matrix[needed_i].to_sparse()])

            selected_hash_idx = move_from_known_graph(target_graphs_hashes, target_graphs_importance_parts, target_graphs_coverage_matrix, importance_args)

            # update transition part of cur_graph
            transitions[graph_hash] = (target_graphs_hashes, target_graphs_actions, target_graphs_importance_parts, target_graphs_coverage_matrix)

        selected_hash = target_graphs_hashes[selected_hash_idx]
        selected_action = target_graphs_actions[selected_hash_idx]
        selected_importance_parts = target_graphs_importance_parts[selected_hash_idx]
        selected_graph = neighbor_graph_access(graph, selected_action)

        if selected_hash not in graph_map:
            selected_input_graphs_covering_list = target_graphs_coverage_matrix[selected_hash_idx]
            graph_map[selected_hash] = selected_graph  # next graph addition to memory
        else:
            selected_input_graphs_covering_list = counterfactual_candidates[graph_index_map[selected_hash]]['input_graphs_covering_list']
        populate_counterfactual_candidates(selected_hash, selected_importance_parts, selected_input_graphs_covering_list)

        return selected_hash, not_teleport


def dynamic_teleportation_probabilities():
    input_graphs_covered_exp = np.exp(input_graphs_covered)
    return (1 / input_graphs_covered_exp) / (1 / input_graphs_covered_exp).sum()


def restart_randomwalk(input_graphs):
    dynamic_probs = dynamic_teleportation_probabilities()
    idx = random.choices(range(dynamic_probs.shape[0]), weights=dynamic_probs)[0]
    graph = input_graphs[idx]
    importance_parts, graph_embeddings, coverage_matrix = importance.call([graph], importance_args)
    input_graphs_covering_list = coverage_matrix[0].to_sparse()
    graph_hash = calculate_hash(graph_embeddings[0])
    if graph_hash not in graph_index_map:
        graph_map[graph_hash] = graph
    populate_counterfactual_candidates(graph_hash, importance_parts[0], input_graphs_covering_list)
    return graph_hash


def counterfactual_summary_with_randomwalk(input_graphs, importance_args, teleport_probability, max_steps):
    cur_graph_hash = restart_randomwalk(input_graphs)
    for step in tqdm(range(starting_step, max_steps + 1)):
        traversed_hashes.append(cur_graph_hash)
        next_graph_hash, is_teleported = move_to_next_graph(graph_hash=cur_graph_hash,
                                                            importance_args=importance_args,
                                                            teleport_probability=teleport_probability)

        cur_graph_hash = restart_randomwalk(input_graphs) if is_teleported else next_graph_hash

        # checking if memory is handled well
        assert len(graph_map) == len(graph_index_map) == len(counterfactual_candidates)  # == len(transitions) - len(input_graphs)
        assert set(graph_index_map.keys()) == set(graph_map.keys())

    save_item = {
        'graph_map': graph_map,
        'graph_index_map': graph_index_map,
        'counterfactual_candidates': counterfactual_candidates,
        'MAX_COUNTERFACTUAL_SIZE': MAX_COUNTERFACTUAL_SIZE,
        'traversed_hashes': traversed_hashes,
        'input_graphs_covered': input_graphs_covered,
    }
    if not os.path.exists(f'results/{dataset_name}/runs/'):
        os.makedirs(f'results/{dataset_name}/runs/')
    torch.save(save_item, f'results/{dataset_name}/runs/counterfactuals.pt')


def prepare_devices(device1, device2):
    device1 = 'cuda:' + device1 if torch.cuda.is_available() and device1 in ['0', '1', '2', '3'] else 'cpu'
    device2 = 'cuda:' + device2 if torch.cuda.is_available() and device2 in ['0', '1', '2', '3'] else 'cpu'

    return device1, device2


if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    args = get_args()

    device1, device2 = prepare_devices(args.device1, args.device2)

    teleport_probability = args.teleport
    max_steps = args.max_steps
    dataset_name = args.dataset
    alpha = args.alpha
    if alpha > 1 or alpha < 0:
        raise Exception('Alpha cannot be bigger than 1, or smaller than 0!')
    sample_size = args.sample_size
    is_sample = args.sample

    # global MAX_COUNTERFACTUAL_SIZE
    MAX_COUNTERFACTUAL_SIZE = args.k

    # Load dataset
    graphs = load_dataset(dataset_name)

    # Load GNN model for dataset
    gnn_model = load_trained_gnn(dataset_name, device=device1)
    gnn_model.eval()

    # Load prediction based on model
    preds = load_trained_prediction(dataset_name, device=device1)
    preds = preds.cpu().numpy()
    input_graph_indices = np.array(range(len(preds)))[preds == 0]
    input_graphs = graphs[input_graph_indices.tolist()]

    # setting covered graph numbers to 0
    input_graphs_covered = torch.zeros(len(input_graphs), dtype=torch.float)

    importance_args = importance.prepare_and_get(graphs, gnn_model, input_graph_indices, args.alpha, args.theta, device1=device1, device2=device2, dataset_name=dataset_name)

    # graphs with adjacency matrix and feature matrix
    counterfactual_summary_with_randomwalk(input_graphs=input_graphs,
                                           importance_args=importance_args,
                                           teleport_probability=teleport_probability,
                                           max_steps=max_steps)
