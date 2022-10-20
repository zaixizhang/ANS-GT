from torch_geometric.datasets import Planetoid, Amazon
from pygsp import graphs
from graph_coarsening.coarsening_utils import coarsen
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset, download_url, Data
import numpy as np
import time
import scipy.sparse as sp
import torch
import os.path as osp

class dataset_heterophily(InMemoryDataset):
    def __init__(self, root='data/', name=None,
                 p2raw=None,
                 train_percent=0.01,
                 transform=None, pre_transform=None):

        existing_dataset = ['chameleon', 'film', 'squirrel', 'cornell']
        if name not in existing_dataset:
            raise ValueError(
                f'name of hypergraph dataset must be one of: {existing_dataset}')
        else:
            self.name = name

        self._train_percent = train_percent

        if (p2raw is not None) and osp.isdir(p2raw):
            self.p2raw = p2raw
        elif p2raw is None:
            self.p2raw = None
        elif not osp.isdir(p2raw):
            raise ValueError(
                f'path to raw hypergraph dataset "{p2raw}" does not exist!')

        if not osp.isdir(root):
            os.makedirs(root)

        self.root = root

        super(dataset_heterophily, self).__init__(
            root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_percent = self.data.train_percent

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        p2f = osp.join(self.raw_dir, self.name)
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)

name = 'film'
#dataset = Planetoid(root='/apdcephfs/private_zaixizhang/data/', name=name, transform=T.NormalizeFeatures())
dataset = dataset_heterophily(root='/apdcephfs/private_zaixizhang/data/', name=name, transform=T.NormalizeFeatures())
data = dataset[0]
adj = sp.coo_matrix((np.ones(data.edge_index.shape[1]), (data.edge_index[0], data.edge_index[1])),
                            shape=(data.y.shape[0], data.y.shape[0]),
                            dtype=np.float32)
tic = time.time()
G = graphs.Graph(adj)
C, Gc, _, _ = coarsen(G, K=10, r=0.5, method='algebraic_JC')

toc = time.time()
print(toc-tic)