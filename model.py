import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
from utils import get_sparse_tensor, one_hot
from torch.nn.init import kaiming_uniform_, xavier_normal, normal_, zeros_, ones_
import sys
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from torch.utils.checkpoint import checkpoint
import dgl
import multiprocessing as mp
import time


def get_model(config, dataset):
    config = config.copy()
    config['dataset'] = dataset
    model = getattr(sys.modules['model'], config['name'])
    model = model(config)
    return model


def init_one_layer(in_features, out_features):
    layer = nn.Linear(in_features, out_features)
    kaiming_uniform_(layer.weight)
    zeros_(layer.bias)
    return layer


class BasicModel(nn.Module):
    def __init__(self, model_config):
        super(BasicModel, self).__init__()
        print(model_config)
        self.config = model_config
        self.name = model_config['name']
        self.device = model_config['device']
        self.n_users = model_config['dataset'].n_users
        self.n_items = model_config['dataset'].n_items
        self.trainable = True

    def predict(self, users):
        raise NotImplementedError

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))


class MF(BasicModel):
    def __init__(self, model_config):
        super(MF, self).__init__(model_config)
        self.embedding_size = model_config['embedding_size']
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        normal_(self.user_embedding.weight, std=0.1)
        normal_(self.item_embedding.weight, std=0.1)
        self.to(device=self.device)

    def bpr_forward(self, users, pos_items, neg_items):
        users_e = self.user_embedding(users)
        pos_items_e, neg_items_e = self.item_embedding(pos_items), self.item_embedding(neg_items)
        l2_norm_sq = torch.norm(users_e, p=2, dim=1) ** 2 + torch.norm(pos_items_e, p=2, dim=1) ** 2 \
                     + torch.norm(neg_items_e, p=2, dim=1) ** 2
        return users_e, pos_items_e, neg_items_e, l2_norm_sq

    def predict(self, users):
        user_e = self.user_embedding(users)
        scores = torch.mm(user_e, self.item_embedding.weight.t())
        return scores


class LightGCN(BasicModel):
    def __init__(self, model_config):
        super(LightGCN, self).__init__(model_config)
        self.embedding_size = model_config['embedding_size']
        self.n_layers = model_config['n_layers']
        self.embedding = nn.Embedding(self.n_users + self.n_items, self.embedding_size)
        self.norm_adj = self.generate_graph(model_config['dataset'])
        normal_(self.embedding.weight, std=0.1)
        self.to(device=self.device)

    def generate_graph(self, dataset):
        sub_mat = sp.coo_matrix((np.ones((len(dataset.train_array),)), np.array(dataset.train_array).T),
                                shape=(self.n_users, self.n_items), dtype=np.float32)
        adj_mat = sp.lil_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat[:self.n_users, self.n_users:] = sub_mat
        adj_mat[self.n_users:, :self.n_users] = sub_mat.T
        adj_mat.tocsr()
        degree = np.array(np.sum(adj_mat, axis=1)).squeeze()
        degree = np.maximum(1., degree)
        d_inv = np.power(degree, -0.5)
        d_mat = sp.diags(d_inv, format='csr', dtype=np.float32)

        norm_adj = d_mat.dot(adj_mat).dot(d_mat)
        norm_adj = get_sparse_tensor(norm_adj, self.device)
        return norm_adj

    def get_rep(self):
        representations = self.embedding.weight
        all_layer_rep = [representations]
        row, column = self.norm_adj.indices()
        g = dgl.graph((column, row), num_nodes=self.norm_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=self.norm_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def bpr_forward(self, users, pos_items, neg_items):
        rep = self.get_rep()
        users_e = self.embedding(users)
        pos_items_e, neg_items_e = self.embedding(self.n_users + pos_items), self.embedding(self.n_users + neg_items)
        l2_norm_sq = torch.norm(users_e, p=2, dim=1) ** 2 + torch.norm(pos_items_e, p=2, dim=1) ** 2 \
                     + torch.norm(neg_items_e, p=2, dim=1) ** 2
        users_r = rep[users, :]
        pos_items_r, neg_items_r = rep[self.n_users + pos_items, :], rep[self.n_users + neg_items, :]
        return users_r, pos_items_r, neg_items_r, l2_norm_sq

    def predict(self, users):
        rep = self.get_rep()
        users_r = rep[users, :]
        all_items_r = rep[self.n_users:, :]
        scores = torch.mm(users_r, all_items_r.t())
        return scores


class NGCF(BasicModel):
    def __init__(self, model_config):
        super(NGCF, self).__init__(model_config)
        self.dropout = model_config['dropout']
        self.embedding_size = model_config['embedding_size']
        self.layer_sizes = model_config['layer_sizes'].copy()
        self.embedding = nn.Embedding(self.n_users + self.n_items, self.embedding_size)
        kaiming_uniform_(self.embedding.weight)
        self.n_layers = len(self.layer_sizes)
        self.layer_sizes.insert(0, self.embedding_size)
        self.gc_layers = []
        self.bi_layers = []
        for layer_idx in range(1, self.n_layers + 1):
            dense_layer = init_one_layer(self.layer_sizes[layer_idx - 1], self.layer_sizes[layer_idx])
            self.gc_layers.append(dense_layer)
            dense_layer = init_one_layer(self.layer_sizes[layer_idx - 1], self.layer_sizes[layer_idx])
            self.bi_layers.append(dense_layer)
        self.gc_layers = nn.ModuleList(self.gc_layers)
        self.bi_layers = nn.ModuleList(self.bi_layers)
        self.norm_adj = self.generate_graph(model_config['dataset'])
        self.to(device=self.device)

    def generate_graph(self, dataset):
        sub_mat = sp.coo_matrix((np.ones((len(dataset.train_array),)), np.array(dataset.train_array).T),
                                shape=(self.n_users, self.n_items), dtype=np.float32)
        adj_mat = sp.lil_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat[:self.n_users, self.n_users:] = sub_mat
        adj_mat[self.n_users:, :self.n_users] = sub_mat.T
        adj_mat = adj_mat.tocsr()
        adj_mat = adj_mat + sp.eye(adj_mat.shape[0], format='csr')

        norm_adj = normalize(adj_mat, axis=1, norm='l1')
        norm_adj = get_sparse_tensor(norm_adj, self.device)
        return norm_adj

    def dropout_sp_mat(self, mat):
        if not self.training:
            return mat
        random_tensor = 1 - self.dropout
        random_tensor += torch.rand(mat._nnz()).to(self.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = mat.indices()
        v = mat.values()

        i = i[:, dropout_mask]
        v = v[dropout_mask] / (1. - self.dropout)
        out = torch.sparse.FloatTensor(i, v, mat.shape).coalesce()
        return out

    def get_rep(self):
        representations = self.embedding.weight
        all_layer_rep = [representations]
        dropped_adj = self.dropout_sp_mat(self.norm_adj)
        row, column = dropped_adj.indices()
        g = dgl.graph((column, row), num_nodes=dropped_adj.shape[0], device=self.device)
        for layer_idx in range(self.n_layers):
            messages_0 = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=dropped_adj.values())
            messages_1 = torch.mul(representations, messages_0)
            messages_0, messages_1 = self.gc_layers[layer_idx](messages_0), self.bi_layers[layer_idx](messages_1)
            representations = F.leaky_relu(messages_0 + messages_1, negative_slope=0.2)
            representations = F.dropout(representations, p=self.dropout, training=self.training)
            all_layer_rep.append(F.normalize(representations, p=2, dim=1))
        final_rep = torch.cat(all_layer_rep, dim=1)
        return final_rep

    def bpr_forward(self, users, pos_items, neg_items):
        rep = self.get_rep()
        users_r = rep[users, :]
        pos_items_r, neg_items_r = rep[self.n_users + pos_items, :], rep[self.n_users + neg_items, :]
        l2_norm_sq = torch.norm(users_r, p=2, dim=1) ** 2 + torch.norm(pos_items_r, p=2, dim=1) ** 2 \
                     + torch.norm(neg_items_r, p=2, dim=1) ** 2
        return users_r, pos_items_r, neg_items_r, l2_norm_sq

    def predict(self, users):
        return LightGCN.predict(self, users)


class ItemKNN(BasicModel):
    def __init__(self, model_config):
        super(ItemKNN, self).__init__(model_config)
        self.k = model_config['k']
        self.data_mat, self.sim_mat = self.calculate_similarity(model_config['dataset'])
        self.trainable = False

    def calculate_similarity(self, dataset):
        data_mat = sp.coo_matrix((np.ones((len(dataset.train_array),)), np.array(dataset.train_array).T),
                                 shape=(self.n_users, self.n_items), dtype=np.float32).tocsr()
        item_degree = np.array(np.sum(data_mat, axis=0)).squeeze()
        row = []
        col = []
        data = []
        for item in range(dataset.n_items):
            intersections = data_mat.T.dot(data_mat[:, item]).toarray().squeeze()
            with np.errstate(invalid='ignore'):
                sims = intersections / (item_degree + item_degree[item] - intersections)
            sims[np.isnan(sims)] = 0.
            row.extend([item] * self.k)
            topk_items = np.argsort(sims)[-self.k:]
            col.extend(topk_items.tolist())
            data.extend(sims[topk_items].tolist())
        sim_mat = sp.coo_matrix((data, (row, col)), shape=(self.n_items, self.n_items), dtype=np.float32).tocsr()
        return data_mat, sim_mat

    def predict(self, users):
        users = users.cpu().numpy()
        profiles = self.data_mat[users, :]
        scores = torch.tensor(profiles.dot(self.sim_mat).toarray(), dtype=torch.float32, device=self.device)
        return scores


class Popularity(BasicModel):
    def __init__(self, model_config):
        super(Popularity, self).__init__(model_config)
        self.item_degree = self.calculate_degree(model_config['dataset'])
        self.trainable = False

    def calculate_degree(self, dataset):
        data_mat = sp.coo_matrix((np.ones((len(dataset.train_array),)), np.array(dataset.train_array).T),
                                 shape=(self.n_users, self.n_items), dtype=np.float32).tocsr()
        item_degree = np.array(np.sum(data_mat, axis=0)).squeeze()
        return torch.tensor(item_degree, dtype=torch.float32, device=self.device)

    def predict(self, users):
        return self.item_degree[None, :].repeat(users.shape[0], 1)


class IGCN(BasicModel):
    def __init__(self, model_config):
        super(IGCN, self).__init__(model_config)
        self.embedding_size = model_config['embedding_size']
        self.n_layers = model_config['n_layers']
        self.dropout = model_config['dropout']
        self.feature_ratio = model_config['feature_ratio']
        self.alpha = 1.
        self.delta = model_config.get('delta', 0.99)
        self.norm_adj = self.generate_graph(model_config['dataset'])
        self.feat_mat, self.user_map, self.item_map, self.row_sum = \
            self.generate_feat(model_config['dataset'], core_ranking=model_config.get('core_ranking', None))
        self.update_feat_mat()

        self.embedding = nn.Embedding(self.feat_mat.shape[1], self.embedding_size)
        normal_(self.embedding.weight, std=0.1)
        self.to(device=self.device)

    def update_feat_mat(self):
        row, _ = self.feat_mat.indices()
        edge_values = torch.pow(self.row_sum[row], (self.alpha - 1.) / 2. - 0.5)
        self.feat_mat = torch.sparse.FloatTensor(self.feat_mat.indices(), edge_values, self.feat_mat.shape).coalesce()

    def feat_mat_anneal(self):
        self.alpha *= self.delta
        self.update_feat_mat()

    def generate_graph(self, dataset):
        return LightGCN.generate_graph(self, dataset)

    def generate_feat(self, dataset, is_updating=False, core_ranking=None):
        if not is_updating:
            if core_ranking:
                core_ranking = np.load(core_ranking)
                ranked_users = core_ranking['ranked_users']
                ranked_items = core_ranking['ranked_items']
            else:
                sub_mat = sp.coo_matrix((np.ones((len(dataset.train_array),)), np.array(dataset.train_array).T),
                                        shape=(self.n_users, self.n_items), dtype=np.float32)
                user_degree = np.array(np.sum(sub_mat, axis=1)).squeeze()
                item_degree = np.array(np.sum(sub_mat, axis=0)).squeeze()
                ranked_users = np.argsort(user_degree)[::-1].copy()
                ranked_items = np.argsort(item_degree)[::-1].copy()

            core_users = ranked_users[:int(self.n_users * self.feature_ratio)]
            core_items = ranked_items[:int(self.n_items * self.feature_ratio)]
            user_map = dict()
            for idx, user in enumerate(core_users):
                user_map[user] = idx
            item_map = dict()
            for idx, item in enumerate(core_items):
                item_map[item] = idx
        else:
            user_map = self.user_map
            item_map = self.item_map

        user_dim, item_dim = len(user_map), len(item_map)
        indices = []
        for user, item in dataset.train_array:
            if item in item_map:
                indices.append([user, user_dim + item_map[item]])
            if user in user_map:
                indices.append([self.n_users + item, user_map[user]])
        for user in range(self.n_users):
            indices.append([user, user_dim + item_dim])
        for item in range(self.n_items):
            indices.append([self.n_users + item, user_dim + item_dim + 1])
        feat = sp.coo_matrix((np.ones((len(indices),)), np.array(indices).T),
                             shape=(self.n_users + self.n_items, user_dim + item_dim + 2), dtype=np.float32).tocsr()
        row_sum = torch.tensor(np.array(np.sum(feat, axis=1)).squeeze(), dtype=torch.float32, device=self.device)
        feat = get_sparse_tensor(feat, self.device)
        return feat, user_map, item_map, row_sum

    def inductive_rep_layer(self, feat_mat):
        padding_tensor = torch.empty([max(self.feat_mat.shape) - self.feat_mat.shape[1], self.embedding_size],
                                     dtype=torch.float32, device=self.device)
        padding_features = torch.cat([self.embedding.weight, padding_tensor], dim=0)

        row, column = feat_mat.indices()
        g = dgl.graph((column, row), num_nodes=max(self.feat_mat.shape), device=self.device)
        x = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=padding_features, rhs_data=feat_mat.values())
        x = x[:self.feat_mat.shape[0], :]
        return x

    def get_rep(self):
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)

        all_layer_rep = [representations]
        row, column = self.norm_adj.indices()
        g = dgl.graph((column, row), num_nodes=self.norm_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=self.norm_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def bpr_forward(self, users, pos_items, neg_items):
        return NGCF.bpr_forward(self, users, pos_items, neg_items)

    def predict(self, users):
        return LightGCN.predict(self, users)

    def save(self, path):
        params = {'sate_dict': self.state_dict(), 'user_map': self.user_map,
                  'item_map': self.item_map, 'alpha': self.alpha}
        torch.save(params, path)

    def load(self, path):
        params = torch.load(path, map_location=self.device)
        self.load_state_dict(params['sate_dict'])
        self.user_map = params['user_map']
        self.item_map = params['item_map']
        self.alpha = params['alpha']
        self.feat_mat, _, _, self.row_sum = self.generate_feat(self.config['dataset'], is_updating=True)
        self.update_feat_mat()


class AttIGCN(IGCN):
    def __init__(self, model_config):
        BasicModel.__init__(self, model_config)
        self.embedding_size = model_config['embedding_size']
        self.n_layers = model_config['n_layers']
        self.n_heads = 4
        self.dropout = model_config['dropout']
        self.feature_ratio = 1.
        self.alpha = 0.
        self.size_chunk = int(1e5)
        self.norm_adj = self.generate_graph(model_config['dataset'])
        self.feat_mat, self.user_map, self.item_map, self.row_sum = self.generate_feat(model_config['dataset'])
        self.feat_mat_anneal()

        self.embedding = nn.Embedding(self.feat_mat.shape[1], self.embedding_size)
        kaiming_uniform_(self.embedding.weight)
        self.weight_q = init_one_layer(self.embedding_size, self.embedding_size * self.n_heads)
        self.weight_k = init_one_layer(self.embedding_size, self.embedding_size * self.n_heads)
        self.to(device=self.device)

    def masked_mm(self, x_q, x_k, row, column):
        alpha = []
        n_non_zeros = row.shape[0]
        end_indices = list(np.arange(0, n_non_zeros, self.size_chunk, dtype=np.int64)) + [n_non_zeros]
        for i_chunk in range(1, len(end_indices)):
            t_q = torch.index_select(x_q, 0, row[end_indices[i_chunk - 1]:end_indices[i_chunk]])
            t_k = torch.index_select(x_k, 0, column[end_indices[i_chunk - 1]:end_indices[i_chunk]])
            alpha.append((t_q * t_k).sum(2))
        alpha = torch.cat(alpha, dim=0)
        return alpha

    def inductive_rep_layer(self, feat_mat, return_alpha=False):
        x_k = self.embedding.weight.detach()
        x_k = self.weight_k(x_k).view(-1, self.n_heads, self.embedding_size)

        row, column = feat_mat.indices()
        g = dgl.graph((column, row), num_nodes=self.feat_mat.shape[1], device=self.device)
        x_q = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=self.embedding.weight.detach(), rhs_data=feat_mat.values())
        x_q = self.weight_q(x_q).view(-1, self.n_heads, self.embedding_size)
        if self.training:
            alpha = checkpoint(self.masked_mm, x_q, x_k, row, column, preserve_rng_state=False)
        else:
            alpha = self.masked_mm(x_q, x_k, row, column)

        row_max_alpha = dgl.ops.gspmm(g, 'copy_rhs', 'max', lhs_data=None, rhs_data=alpha)
        alpha = alpha - row_max_alpha[row, :]
        alpha = torch.exp(alpha / np.sqrt(self.embedding_size))
        row_sum_alpha = dgl.ops.gspmm(g, 'copy_rhs', 'sum', lhs_data=None, rhs_data=alpha)
        alpha = alpha / row_sum_alpha[row, :]
        alpha = alpha.mean(-1)
        if return_alpha:
            return alpha

        out = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=self.embedding.weight, rhs_data=alpha)
        out = out[:self.feat_mat.shape[0], :]
        return out

    def bpr_forward(self, users, pos_items, neg_items):
        users_r, pos_items_r, neg_items_r, l2_norm_sq = NGCF.bpr_forward(self, users, pos_items, neg_items)
        l2_norm_sq += torch.norm(self.weight_q.weight, p=2) ** 2 + torch.norm(self.weight_k.weight, p=2) ** 2
        return users_r, pos_items_r, neg_items_r, l2_norm_sq


class IMF(IGCN):
    def __init__(self, model_config):
        super(IMF, self).__init__(model_config)

    def get_rep(self):
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = IGCN.inductive_rep_layer(self, feat_mat)
        return representations


class IMCGAE(BasicModel):
    def __init__(self, model_config):
        super(IMCGAE, self).__init__(model_config)
        self.embedding_size = model_config['embedding_size']
        self.n_layers = model_config['n_layers']
        self.dropout = model_config['dropout']
        self.embedding = nn.Embedding(self.n_users + self.n_items + 3, self.embedding_size)
        self.norm_adj = self.generate_graph(model_config['dataset'])
        normal_(self.embedding.weight, std=0.1)
        self.to(device=self.device)

    def generate_graph(self, dataset):
        return LightGCN.generate_graph(self, dataset)

    def get_rep(self):
        personal_user_embedding = self.embedding.weight[:self.n_users, :]
        personal_item_embedding = self.embedding.weight[self.n_users:self.n_users + self.n_items, :]
        identical_embedding = self.embedding.weight[self.n_users + self.n_items, :]
        general_user_embedding = self.embedding.weight[self.n_users + self.n_items + 1, :]
        general_item_embedding = self.embedding.weight[self.n_users + self.n_items + 2, :]
        u_representations = torch.cat([personal_user_embedding,
                                       general_user_embedding[None, :].expand(personal_user_embedding.shape),
                                       identical_embedding[None, :].expand(personal_user_embedding.shape)], dim=1)
        i_representations = torch.cat([personal_item_embedding,
                                       general_item_embedding[None, :].expand(personal_item_embedding.shape),
                                       identical_embedding[None, :].expand(personal_item_embedding.shape)], dim=1)

        representations = torch.cat([u_representations, i_representations], dim=0)
        all_layer_rep = [representations]
        row, column = self.norm_adj.indices()
        g = dgl.graph((column, row), num_nodes=self.norm_adj.shape[0], device=self.device)
        for i in range(self.n_layers):
            node_dropout_masks = torch.ones(self.n_users + self.n_items, dtype=torch.float32, device=self.device)
            node_dropout_masks = F.dropout(node_dropout_masks, p=self.dropout - 0.1 * i, training=self.training)
            representations = representations * node_dropout_masks[:, None]
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=self.norm_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def bpr_forward(self, users, pos_items, neg_items):
        return NGCF.bpr_forward(self, users, pos_items, neg_items)

    def predict(self, users):
        return LightGCN.predict(self, users)


class IGMC(BasicModel):
    def __init__(self, model_config):
        super(IGMC, self).__init__(model_config)
        self.h_hop = model_config['h_hop']
        self.layer_sizes = model_config['layer_sizes'].copy()
        self.layer_sizes.insert(0, 4)
        self.dropout = model_config['dropout']
        self.adj_mat = self.generate_graph(model_config['dataset'])
        self.neighbors = self.generate_neighbors()
        self.layers = []
        for layer_idx in range(1, len(self.layer_sizes)):
            conv = dgl.nn.GraphConv(self.layer_sizes[layer_idx - 1], self.layer_sizes[layer_idx],
                                    norm='both', allow_zero_in_degree=True)
            self.layers.append(conv)
        self.layers = nn.ModuleList(self.layers)
        self.lin1 = init_one_layer(sum(self.layer_sizes[1:]) * 2, self.layer_sizes[-1])
        self.lin2 = init_one_layer(self.layer_sizes[-1], 1)
        self.to(device=self.device)

    def generate_graph(self, dataset):
        sub_mat = sp.coo_matrix((np.ones((len(dataset.train_array),)), np.array(dataset.train_array).T),
                                shape=(self.n_users, self.n_items), dtype=np.float32)
        adj_mat = sp.lil_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat[:self.n_users, self.n_users:] = sub_mat
        adj_mat[self.n_users:, :self.n_users] = sub_mat.T
        return adj_mat.tocsr()

    def generate_neighbors(self):
        neighbors = []
        for node in range(self.adj_mat.shape[0]):
            nei = np.nonzero(self.adj_mat[node, :])[1].tolist()
            tmp = [[node] + nei, [0] + [1] * len(nei)]
            neighbors.append(tmp)
        return neighbors

    @staticmethod
    def subgraph_extraction_labeling(index, adj_mat, u_nei, i_nei):
        user, item = u_nei[0], i_nei[0]
        pos = False
        if user in i_nei[0]:
            i = i_nei[0].index(user)
            i_nei = [i_nei[0][:i] + i_nei[0][i+1:], i_nei[1][:i] + i_nei[1][i+1:]]
            i = u_nei[0].index(item)
            u_nei = [u_nei[0][:i] + u_nei[0][i+1:], u_nei[1][:i] + u_nei[1][i+1:]]
            pos = True
        nodes = u_nei[0] + i_nei[0]
        labels = u_nei[1] + [x + 2 for x in i_nei[1]]

        graph = adj_mat[nodes, :][:, nodes]
        if pos:
            graph[0, len(u_nei[0])] = 0.
        r, c, _ = sp.find(graph)
        return index, r, c, one_hot(np.array(labels), 4)

    def forward(self, users, items):
        pool = mp.Pool(4)
        results = pool.starmap_async(self.subgraph_extraction_labeling,
                                     [(i, self.adj_mat, self.neighbors[user], self.neighbors[self.n_users + item])
                                      for i, (user, item) in enumerate(zip(users, items))])
        pool.close()
        pool.join()
        results = results.get()

        ordered = [None for _ in range(users.shape[0])]
        for index, r, c, labels in results:
            ordered[index] = [r, c, labels]

        row, column, feats = [], [], []
        num_nodes = 0
        for r, c, labels in ordered:
            row.append(r + num_nodes)
            column.append(c + num_nodes)
            feats.append(torch.tensor(labels, dtype=torch.float32, device=self.device))
            num_nodes += labels.shape[0]

        row = np.concatenate(row, axis=0)
        column = np.concatenate(column, axis=0)
        feats = torch.cat(feats, dim=0)
        if self.training:
            n_edges = row.shape[0]
            sampled_edges = np.random.permutation(n_edges)[int(n_edges * self.dropout):]
            row = row[sampled_edges]
            column = column[sampled_edges]

        g = dgl.graph((row, column), device=self.device)
        all_rep = []
        x = feats
        for layer in self.layers:
            x = torch.tanh(layer(g, x))
            all_rep.append(x)
        all_rep = torch.cat(all_rep, dim=1)
        users = feats[:, 0] == 1.
        items = feats[:, 2] == 1.
        x = torch.cat([all_rep[users, :], all_rep[items, :]], dim=1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x.squeeze()

    def bpr_forward(self, users, pos_items, neg_items):
        users, pos_items, neg_items = users.cpu().numpy(), pos_items.cpu().numpy(), neg_items.cpu().numpy()
        pos_scores = self.forward(users, pos_items)
        neg_scores = self.forward(users, neg_items)
        l2_norm = torch.norm(self.lin1.weight, p=2) ** 2 + torch.norm(self.lin1.weight, p=2) ** 2
        for layer in self.layers:
            l2_norm += torch.norm(layer.weight, p=2) ** 2
        ones = torch.ones([1, 1], dtype=torch.float32, device=self.device)
        return ones, pos_scores[:, None], neg_scores[:, None], l2_norm[None]

    def predict(self, users):
        items = torch.arange(self.n_items, dtype=torch.int64, device=self.device).repeat(users.shape[0])
        users = users[:, None].repeat(1, self.n_items).flatten()
        scores = self.forward(users, items)
        scores = scores.reshape(-1, self.n_items)
        return scores


class MultiVAE(BasicModel):
    def __init__(self, model_config):
        super(MultiVAE, self).__init__(model_config)
        self.dropout = model_config['dropout']
        self.normalized_data_mat = self.get_data_mat(model_config['dataset'])

        self.e_layer_sizes = model_config['layer_sizes'].copy()
        self.e_layer_sizes.insert(0, self.normalized_data_mat.shape[1])
        self.d_layer_sizes = self.e_layer_sizes[::-1].copy()
        self.mid_size = self.e_layer_sizes[-1]
        self.e_layer_sizes[-1] = self.mid_size * 2
        self.encoder_layers = []
        self.decoder_layers = []
        for layer_idx in range(1, len(self.e_layer_sizes)):
            encoder_layer = init_one_layer(self.e_layer_sizes[layer_idx - 1], self.e_layer_sizes[layer_idx])
            self.encoder_layers.append(encoder_layer)
            decoder_layer = init_one_layer(self.d_layer_sizes[layer_idx - 1], self.d_layer_sizes[layer_idx])
            self.decoder_layers.append(decoder_layer)
        self.encoder_layers = nn.ModuleList(self.encoder_layers)
        self.decoder_layers = nn.ModuleList(self.decoder_layers)
        self.to(device=self.device)

    def get_data_mat(self, dataset):
        data_mat = sp.coo_matrix((np.ones((len(dataset.train_array),)), np.array(dataset.train_array).T),
                                 shape=(self.n_users, self.n_items), dtype=np.float32).tocsr()

        normalized_data_mat = normalize(data_mat, axis=1, norm='l2')
        return normalized_data_mat

    def ml_forward(self, users):
        users = users.cpu().numpy()
        profiles = self.normalized_data_mat[users, :]
        representations = get_sparse_tensor(profiles, self.device)

        representations = NGCF.dropout_sp_mat(self, representations)
        representations = torch.sparse.mm(representations, self.encoder_layers[0].weight.t())
        representations += self.encoder_layers[0].bias[None, :]
        l2_norm_sq = torch.norm(self.encoder_layers[0].weight, p=2)[None] ** 2
        for layer in self.encoder_layers[1:]:
            representations = layer(torch.tanh(representations))
            l2_norm_sq += torch.norm(layer.weight, p=2)[None] ** 2

        mean, log_var = representations[:, :self.mid_size], representations[:, -self.mid_size:]
        std = torch.exp(0.5 * log_var)
        kl = torch.sum(-log_var + torch.exp(log_var) + mean ** 2, dim=1)
        epsilon = torch.randn(mean.shape[0], mean.shape[1], device=self.device)
        representations = mean + float(self.training) * epsilon * std

        for layer in self.decoder_layers[:-1]:
            representations = torch.tanh(layer(representations))
            l2_norm_sq += torch.norm(layer.weight, p=2)[None] ** 2
        scores = self.decoder_layers[-1](representations)
        l2_norm_sq += torch.norm(self.decoder_layers[-1].weight, p=2)[None] ** 2
        return scores, kl, l2_norm_sq

    def predict(self, users):
        scores, _, _ = self.ml_forward(users)
        return scores


class NeuMF(BasicModel):
    def __init__(self, model_config):
        super(NeuMF, self).__init__(model_config)
        self.embedding_size = model_config['embedding_size']
        self.layer_sizes = model_config['layer_sizes']
        self.mf_user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.mf_item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.mlp_user_embedding = nn.Embedding(self.n_users, self.layer_sizes[0] // 2)
        self.mlp_item_embedding = nn.Embedding(self.n_items, self.layer_sizes[0] // 2)
        self.mlp_layers = []
        for layer_idx in range(1, len(self.layer_sizes)):
            dense_layer = nn.Linear(self.layer_sizes[layer_idx - 1], self.layer_sizes[layer_idx])
            self.mlp_layers.append(dense_layer)
        self.mlp_layers = nn.ModuleList(self.mlp_layers)
        self.output_layer = nn.Linear(self.layer_sizes[-1] + self.embedding_size, 1, bias=False)

        kaiming_uniform_(self.mf_user_embedding.weight)
        kaiming_uniform_(self.mf_item_embedding.weight)
        kaiming_uniform_(self.mlp_user_embedding.weight)
        kaiming_uniform_(self.mlp_item_embedding.weight)
        self.init_mlp_layers()
        self.arch = 'gmf'
        self.to(device=self.device)

    def init_mlp_layers(self):
        for layer in self.mlp_layers:
            kaiming_uniform_(layer.weight)
            zeros_(layer.bias)
        ones_(self.output_layer.weight)

    def bce_forward(self, users, items):
        users_mf_e, items_mf_e = self.mf_user_embedding(users), self.mf_item_embedding(items)
        users_mlp_e, items_mlp_e = self.mlp_user_embedding(users), self.mlp_item_embedding(items)

        mf_vectors = users_mf_e * items_mf_e
        mlp_vectors = torch.cat([users_mlp_e, items_mlp_e], dim=1)
        for layer in self.mlp_layers:
            mlp_vectors = F.leaky_relu(layer(mlp_vectors))

        if self.arch == 'gmf':
            vectors = [mf_vectors, torch.zeros_like(mlp_vectors, device=self.device, dtype=torch.float32)]
        elif self.arch == 'mlp':
            vectors = [torch.zeros_like(mf_vectors, device=self.device, dtype=torch.float32), mlp_vectors]
        else:
            vectors = [mf_vectors, mlp_vectors]
        predict_vectors = torch.cat(vectors, dim=1)
        scores = predict_vectors * self.output_layer.weight
        l2_norm_sq = torch.norm(scores, p=2, dim=1) ** 2
        scores = scores.sum(dim=1)
        return scores, l2_norm_sq

    def predict(self, users):
        items = torch.arange(self.n_items, dtype=torch.int64, device=self.device).repeat(users.shape[0])
        users = users[:, None].repeat(1, self.n_items).flatten()
        scores, _ = self.bce_forward(users, items)
        scores = scores.reshape(-1, self.n_items)
        return scores
