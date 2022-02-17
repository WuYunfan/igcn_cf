import numpy as np
import torch
import random
import os
import sys
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import networkx as nx
from sortedcontainers import SortedList


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


def generate_daj_mat(dataset):
    train_array = np.array(dataset.train_array)
    users, items = train_array[:, 0], train_array[:, 1]
    row = np.concatenate([users, items + dataset.n_users], axis=0)
    column = np.concatenate([items + dataset.n_users, users], axis=0)
    adj_mat = sp.coo_matrix((np.ones(row.shape), np.stack([row, column], axis=0)),
                            shape=(dataset.n_users + dataset.n_items, dataset.n_users + dataset.n_items),
                            dtype=np.float32).tocsr()
    return adj_mat


'''
# This is for theoretical analysis.
def greedy_or_sort(part_adj, u, ranking_metric, device):
    user_norm = np.linalg.norm(u, axis=1, ord=2) ** 2
    u_u_adj = part_adj.dot(part_adj.T)
    if ranking_metric == 'sort':
        user_metrics = np.array(np.sum(u_u_adj, axis=1)).squeeze() * user_norm
        return user_metrics

    metrics_greedy = np.array(np.sum(part_adj, axis=1)).squeeze() * user_norm
    user_metrics = np.zeros_like(metrics_greedy)
    n_users = part_adj.shape[0]
    sl = SortedList(key=lambda element: element[1])
    for ui in range(n_users):
        sl.add((ui, metrics_greedy[ui]))

    for nu in range(n_users):
        ui = sl.pop(0)[0]
        user_metrics[ui] = nu
        for uj in np.nonzero(u_u_adj[ui])[0]:
            sl.discard((uj, metrics_greedy[uj]))
            metrics_greedy[uj] += user_norm[uj] * u_u_adj[ui, uj]
            sl.add((uj, metrics_greedy[uj]))
    return user_metrics



def greedy_or_sort(part_adj, u, ranking_metric, device):
    normalized_adj_mat = normalize(part_adj, axis=0, norm='l1')
    user_metrics = np.array(np.sum(normalized_adj_mat, axis=1)).squeeze()

    user_degree = np.array(np.sum(part_adj, axis=1)).squeeze()
    n_u = u / user_degree[:, None]
    u = torch.tensor(u, dtype=torch.float32, device=device)
    n_u = torch.tensor(n_u, dtype=torch.float32, device=device)
    for i in range(u.shape[0]):
        s_norm_sq = torch.norm(torch.mm(n_u, u[i, :][:, None]), p=2) ** 2
        user_metrics[i] *= s_norm_sq.item()
    return user_metrics
'''


def graph_rank_nodes(dataset, ranking_metric):
    adj_mat = generate_daj_mat(dataset)
    if ranking_metric == 'degree':
        user_metrics = np.array(np.sum(adj_mat[:dataset.n_users, :], axis=1)).squeeze()
        item_metrics = np.array(np.sum(adj_mat[dataset.n_users:, :], axis=1)).squeeze()
    elif ranking_metric == 'greedy' or ranking_metric == 'sort':
        '''
        # This is for theoretical analysis.
        part_adj = adj_mat[:dataset.n_users, dataset.n_users:]
        part_adj_tensor = get_sparse_tensor(part_adj, 'cpu')
        with torch.no_grad():
            u, s, v = torch.svd_lowrank(part_adj_tensor, 64)
            u, v = u.numpy(), v.numpy()
        user_metrics = greedy_or_sort(part_adj, u, ranking_metric, dataset.device)
        item_metrics = greedy_or_sort(part_adj.T, v, ranking_metric, dataset.device)
        '''
        normalized_adj_mat = normalize(adj_mat, axis=1, norm='l1')
        user_metrics = np.array(np.sum(normalized_adj_mat[:, :dataset.n_users], axis=0)).squeeze()
        item_metrics = np.array(np.sum(normalized_adj_mat[:, dataset.n_users:], axis=0)).squeeze()
    elif ranking_metric == 'page_rank':
        g = nx.Graph()
        g.add_edges_from(np.array(np.nonzero(adj_mat)).T)
        pr = nx.pagerank(g)
        pr = np.array([pr[i] for i in range(dataset.n_users + dataset.n_items)])
        user_metrics, item_metrics = pr[:dataset.n_users], pr[dataset.n_users:]
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

