# Inspired by https://github.com/deepfindr/gnn-project
# dataset classes and load function

import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.datasets import TUDataset
from torch.nn.functional import one_hot
import os
import shutil


class Mutagenicity(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.root = root
        self.graph_count = 4308

        super(Mutagenicity, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return []

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        return [f'data_{i}.pt' for i in range(self.graph_count)]

    def download(self):
        pass

    def process(self):
        graphs = TUDataset(root=os.path.join(self.root, 'tudataset'), name='Mutagenicity')

        # we remove graphs if they include node classes which has frequency less than or equal to 50.
        x_all = []
        for graph in graphs:
            x_all.append(graph.x.sum(dim=0))
        x_all = torch.stack(x_all).sum(dim=0)
        atoms_to_keep = torch.where(x_all > 50)[0]
        print(f'There are {len(atoms_to_keep)} valid number of labels!')

        count = 0
        for i, graph in enumerate(graphs):
            # Create data object
            if graph.x.sum() == graph.x[:, atoms_to_keep].sum():
                # Create data
                x = graph.x[:, atoms_to_keep]
                data = Data(edge_index=graph.edge_index.clone(),
                            y=torch.tensor(graph.y.item()),  # 0 is mutagenetic, which is undesired for drug discovery.
                            # node_labels=self.label_from_one_hot(x),
                            x=x.clone(),
                            num_nodes=graph.num_nodes
                            )

                torch.save(data, os.path.join(self.processed_dir, f'data_{count}.pt'))
                count += 1
        print(f"There are {count} graphs!")

        # Delete TUDataset.
        del i, graph, graphs
        shutil.rmtree(os.path.join(self.root, 'tudataset'))

    def len(self):
        return self.graph_count

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        data = torch.load(os.path.join(self.processed_dir,
                                       f'data_{idx}.pt'))
        return data

    @staticmethod
    def one_hot_from_label(labels):
        return one_hot(labels, num_classes=10)

    @staticmethod
    def label_from_one_hot(one_hot):
        return torch.argmax(one_hot, dim=1)

    @staticmethod
    def num_classes():
        return 10


class AIDS(Dataset):

    def __init__(self, root, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.root = root
        self.graph_count = 1837

        super(AIDS, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return []

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        return [f'data_{i}.pt' for i in range(self.graph_count)]

    def download(self):
        pass

    def process(self):

        graphs = TUDataset(root=os.path.join(self.root, 'tudataset'), name='AIDS')

        # we remove graphs if they include node classes which has frequency less than or equal to 50.
        x_all = []
        for graph in graphs:
            x_all.append(graph.x.sum(dim=0))
        x_all = torch.stack(x_all).sum(dim=0)
        atoms_to_keep = torch.where(x_all > 50)[0]
        print(f'There are {len(atoms_to_keep)} valid number of labels!')

        count = 0
        for i, graph in enumerate(graphs):
            if graph.x.sum() == graph.x[:, atoms_to_keep].sum():
                # Create data
                x = graph.x[:, atoms_to_keep]
                data = Data(edge_index=graph.edge_index.clone(),
                            edge_attr=torch.argmax(graph.edge_attr, dim=1).clone(),
                            y=torch.tensor(0) if graph.y.item() else torch.tensor(1),  # Note that original graph label 0 means active against aids, which is desired. So we swap the order.
                            # node_labels=self.label_from_one_hot(x),
                            x=x.clone(),
                            num_nodes=graph.num_nodes,
                            )
                torch.save(data, os.path.join(self.processed_dir, f'data_{count}.pt'))
                count += 1
        print(f"There are {count} graphs!")

        # Delete TUDataset.
        del i, graph, graphs
        shutil.rmtree(os.path.join(self.root, 'tudataset'))

    def len(self):
        return self.graph_count

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        data = torch.load(os.path.join(self.processed_dir,
                                       f'data_{idx}.pt'))
        return data

    @staticmethod
    def one_hot_from_label(labels):
        return one_hot(labels, num_classes=9)

    @staticmethod
    def label_from_one_hot(one_hot):
        return torch.argmax(one_hot, dim=1)

    @staticmethod
    def num_classes():
        # return 38
        return 9


class NCI1(Dataset):

    def __init__(self, root, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.root = root
        self.graph_count = 3978

        super(NCI1, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return []

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        return [f'data_{i}.pt' for i in range(self.graph_count)]

    def download(self):
        pass

    def process(self):
        graphs = TUDataset(root=os.path.join(self.root, 'tudataset'), name='NCI1')

        # we remove graphs if they include node classes which has frequency less than or equal to 50.
        x_all = []
        for graph in graphs:
            x_all.append(graph.x.sum(dim=0))
        x_all = torch.stack(x_all).sum(dim=0)
        atoms_to_keep = torch.where(x_all > 50)[0]
        print(f'There are {len(atoms_to_keep)} valid number of labels!')

        count = 0
        for i, graph in enumerate(graphs):
            # Create data object
            if graph.x.sum() == graph.x[:, atoms_to_keep].sum():
                x = graph.x[:, atoms_to_keep]
                data = Data(edge_index=graph.edge_index.clone(),
                            y=torch.tensor(0) if graph.y.item() else torch.tensor(1),  # Note that original graph label 0 means active against cancer, which is desired. So we swap the order.
                            # node_labels=self.label_from_one_hot(x),
                            x=x.clone(),
                            num_nodes=graph.num_nodes
                            )

                torch.save(data, os.path.join(self.processed_dir, f'data_{count}.pt'))
                count += 1
        print(f"There are {count} graphs!")

        # Delete TUDataset.
        del i, graph, graphs
        shutil.rmtree(os.path.join(self.root, 'tudataset'))

    def len(self):
        return self.graph_count

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        data = torch.load(os.path.join(self.processed_dir,
                                       f'data_{idx}.pt'))
        return data

    @staticmethod
    def one_hot_from_label(labels):
        return one_hot(labels, num_classes=10)

    @staticmethod
    def label_from_one_hot(one_hot):
        return torch.argmax(one_hot, dim=1)

    @staticmethod
    def num_classes():
        # return 37
        return 10


class PROTEINS(Dataset):

    def __init__(self, root, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.root = root
        self.graph_count = 1113

        super(PROTEINS, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return []

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        return [f'data_{i}.pt' for i in range(self.graph_count)]

    def download(self):
        pass

    def process(self):
        graphs = TUDataset(root=os.path.join(self.root, 'tudataset'), name='PROTEINS')
        for i, graph in enumerate(graphs):
            # Create data object
            # assert torch.abs(torch.sum(graph.x, axis=1) - torch.ones(size=torch.sum(graph.x, axis=1).size())).sum().item() == 0
            data = Data(edge_index=graph.edge_index.clone(),
                        y=torch.tensor(0) if graph.y.item() else torch.tensor(1),  # Note that original graph label 0 means enzyme, which is desired. So we swap the order.
                        # node_labels=self.label_from_one_hot(graph.x),
                        x=graph.x.clone(),
                        num_nodes=graph.num_nodes
                        )

            torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))

        # Delete TUDataset.
        del i, graph, graphs
        shutil.rmtree(os.path.join(self.root, 'tudataset'))

    def len(self):
        return self.graph_count

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        data = torch.load(os.path.join(self.processed_dir,
                                       f'data_{idx}.pt'))
        return data

    @staticmethod
    def one_hot_from_label(labels):
        return one_hot(labels, num_classes=3)

    @staticmethod
    def label_from_one_hot(one_hot):
        return torch.argmax(one_hot, dim=1)

    @staticmethod
    def num_classes():
        return 3


def load_dataset(dataset_name):
    if dataset_name == 'mutagenicity':
        dataset = Mutagenicity('data/mutagenicity')
    elif dataset_name == 'aids':
        dataset = AIDS('data/aids')
    elif dataset_name == 'nci1':
        dataset = NCI1('data/nci1')
    elif dataset_name == 'proteins':
        dataset = PROTEINS('data/proteins')
    else:
        raise ValueError(f'Dataset {dataset_name} not supported. ')

    return dataset
