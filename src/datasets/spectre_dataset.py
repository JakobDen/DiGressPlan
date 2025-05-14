import os
import pathlib
from scipy.io import loadmat
import torch
from torch_geometric.data import Dataset, Data
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
import numpy as np


ROOM_TYPE_COUNT = 17
EDGE_TYPE_COUNT = 10


class SpectreGraphDataset(Dataset):
    def __init__(self, split, root, transform=None, pre_transform=None, pre_filter=None):
        self.split = split
        self.root = root
        self.file_name = f"data_{split}.mat"
        self.data_path = os.path.join(self.root, self.file_name)

        assert os.path.exists(self.data_path), f"File not found: {self.data_path}"

        self.mat_data = loadmat(self.data_path, struct_as_record=False, squeeze_me=True)["data"]
        self._len = len(self.mat_data)

        super().__init__(root, transform, pre_transform, pre_filter)

    def len(self):
        return self._len

    def get(self, idx):
        entry = self.mat_data[idx]
        room_types = torch.tensor(entry.rType, dtype=torch.long)
        node_features = torch.nn.functional.one_hot(room_types, num_classes=ROOM_TYPE_COUNT).float()
        n = len(entry.order)
        edges = entry.rEdge #(N_edges, 3)
        edges = np.atleast_2d(edges)
        if edges.ndim == 1:
            edges = edges[np.newaxis, :]

        edge_index = torch.tensor(edges[:, :2].T, dtype=torch.long)
        edge_type = torch.tensor(edges[:, 2], dtype=torch.long)
        edge_attr = torch.nn.functional.one_hot(edge_type, num_classes=EDGE_TYPE_COUNT).float()

        y = torch.zeros([1, 0]).float()
        num_nodes = n * torch.ones(1, dtype=torch.long)

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=y, n_nodes=num_nodes)
        return data

class SpectreGraphDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, cfg["dataset"]["datadir"])
        datasets = {
            "train": SpectreGraphDataset(split='train', root=root_path),
            "val": SpectreGraphDataset(split='val', root=root_path),
            "test": SpectreGraphDataset(split='test', root=root_path),
        }
        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class SpectreDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = "nx_graphs"
        self.n_nodes = datamodule.node_counts()
        self.node_types = torch.arange(1, 18)
        self.edge_types = datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)