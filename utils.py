import numpy as np
import torch
import random
import os
import sys
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import networkx as nx


def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_run(log_path, seed):
    set_seed(seed)
    if not os.path.exists(log_path): os.mkdir(log_path)
    f = open(os.path.join(log_path, 'log.txt'), 'w')
    f = Unbuffered(f)
    sys.stderr = f
    sys.stdout = f


def get_sparse_tensor(mat, device):
    coo = mat.tocoo()
    indexes = np.stack([coo.row, coo.col], axis=0)
    indexes = torch.tensor(indexes, dtype=torch.int64, device=device)
    data = torch.tensor(coo.data, dtype=torch.float32, device=device)
    sp_tensor = torch.sparse.FloatTensor(indexes, data, torch.Size(coo.shape)).coalesce()
    return sp_tensor


def graph_rank_nodes(adj_mat, ranking_metric):
    n_users, n_items = adj_mat.shape[0], adj_mat.shape[1]
    sub_mat = adj_mat
    adj_mat = sp.lil_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    adj_mat[:n_users, n_users:] = sub_mat
    adj_mat[n_users:, :n_users] = sub_mat.T
    if ranking_metric == 'degree':
        user_metrics = np.array(np.sum(adj_mat[:n_users, :], axis=1)).squeeze()
        item_metrics = np.array(np.sum(adj_mat[n_users:, :], axis=1)).squeeze()
    elif ranking_metric == 'normalized_degree':
        normalized_adj_mat = normalize(adj_mat, axis=1, norm='l1')
        user_metrics = np.array(np.sum(normalized_adj_mat[:, :n_users], axis=0)).squeeze()
        item_metrics = np.array(np.sum(normalized_adj_mat[:, n_users:], axis=0)).squeeze()
    elif ranking_metric == 'page_rank':
        g = nx.Graph()
        g.add_edges_from(np.array(np.nonzero(adj_mat)).T)
        pr = nx.pagerank(g)
        pr = np.array([pr[i] for i in range(n_users + n_items)])
        user_metrics, item_metrics = pr[:n_users], pr[n_users:]
    else:
        return None
    ranked_users = np.argsort(user_metrics)[::-1].copy()
    ranked_items = np.argsort(item_metrics)[::-1].copy()
    return ranked_users, ranked_items


class AverageMeter:
    def __init__(self):
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)

