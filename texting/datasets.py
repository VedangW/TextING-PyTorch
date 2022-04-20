import sys
import numpy as np
import pickle as pkl

from torch.utils.data import Dataset
from tqdm import tqdm


class GraphDataset(Dataset):
    def __init__(self, dataset_str, part='train'):
        self.adjs, self.features, self.y = self._load_data(dataset_str, part)
        self.adjs, self.mask = self._preprocess_adj(self.adjs)
        self.features = self._preprocess_features(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.adjs[idx], self.mask[idx], self.y[idx]

    def __len__(self):
        return len(self.y)

    def _load_data(self, dataset_str, part):
        """
        Loads input data from gcn/data directory

        ind.dataset_str.x => the feature vectors and adjacency matrix of the training instances as list;
        ind.dataset_str.tx => the feature vectors and adjacency matrix of the test instances as list;
        ind.dataset_str.allx => the feature vectors and adjacency matrix of both labeled and unlabeled training instances
            (a superset of ind.dataset_str.x) as list;
        ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
        ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
        ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;

        All objects above must be saved using python pickle module.

        :param dataset_str: Dataset name
        :return: All data input files loaded (as well the training/test data).
        """
        names = {
            'train': ['x_adj', 'x_embed', 'y'],
            'val': ['allx_adj', 'allx_embed', 'ally'],
            'test': ['tx_adj', 'tx_embed', 'ty']
        }[part]
        
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x_adj, x_embed, y = tuple(objects)
        
        if part == 'val':
            with open(f'data/ind.{dataset_str}.y', 'rb') as f:
                if sys.version_info > (3, 0):
                    temp = pkl.load(f, encoding='latin1')
                else:
                    temp = pkl.load(f)

            start, end = len(temp), len(y)
            del temp
        else:
            start, end = 0, len(y)
            
        adjs = []
        embeds = []
        
        for i in range(start, end):
            adj = x_adj[i].toarray()
            embed = np.array(x_embed[i])
            adjs.append(adj)
            embeds.append(embed)

        adjs = np.array(adjs)
        embeds = np.array(embeds)
        y = np.array(y[start:end])
        return adjs, embeds, y

    def _normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        rowsum = np.array(adj.sum(1))
        with np.errstate(divide='ignore'):
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

    def _preprocess_adj(self, adj):
        """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
        max_length = max([a.shape[0] for a in adj])
        mask = np.zeros((adj.shape[0], max_length, 1)) # mask for padding

        for i in tqdm(range(adj.shape[0])):
            adj_normalized = self._normalize_adj(adj[i]) # no self-loop
            pad = max_length - adj_normalized.shape[0] # padding for each epoch
            adj_normalized = np.pad(adj_normalized, ((0,pad),(0,pad)), mode='constant')
            mask[i,:adj[i].shape[0],:] = 1.
            adj[i] = adj_normalized

        return np.array(list(adj)), mask

    def _preprocess_features(self, features):
        """Row-normalize feature matrix and convert to tuple representation"""
        max_length = max([len(f) for f in features])
        
        for i in tqdm(range(features.shape[0])):
            feature = np.array(features[i])
            pad = max_length - feature.shape[0] # padding for each epoch
            feature = np.pad(feature, ((0,pad),(0,0)), mode='constant')
            features[i] = feature
        
        return np.array(list(features))