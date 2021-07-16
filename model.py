import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
from utils import get_sparse_tensor
from torch.nn.init import kaiming_uniform_, calculate_gain, normal_, zeros_, ones_
import sys
import torch.nn.functional as F
from sklearn.preprocessing import normalize
import dgl


def get_model(config, dataset):
    config = config.copy()
    config['dataset'] = dataset
    model = getattr(sys.modules['model'], config['name'])
    model = model(config)
    return model


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
        self.load_state_dict(torch.load(path))


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
        for _ in range(self.n_layers):
            representations = torch.sparse.mm(self.norm_adj, representations)
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
        self.n_layers = len(self.layer_sizes)
        self.layer_sizes.insert(0, self.embedding_size)
        self.gc_layers = []
        self.bi_layers = []
        for layer_idx in range(1, self.n_layers + 1):
            dense_layer = nn.Linear(self.layer_sizes[layer_idx - 1], self.layer_sizes[layer_idx])
            self.gc_layers.append(dense_layer)
            dense_layer = nn.Linear(self.layer_sizes[layer_idx - 1], self.layer_sizes[layer_idx])
            self.bi_layers.append(dense_layer)
        self.gc_layers = nn.ModuleList(self.gc_layers)
        self.bi_layers = nn.ModuleList(self.bi_layers)
        self.norm_adj = self.generate_graph(model_config['dataset'])
        for layer in self.gc_layers:
            kaiming_uniform_(layer.weight, nonlinearity='leaky_relu')
            zeros_(layer.bias)
        for layer in self.bi_layers:
            kaiming_uniform_(layer.weight, nonlinearity='leaky_relu')
            zeros_(layer.bias)
        kaiming_uniform_(self.embedding.weight, nonlinearity='leaky_relu')
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
        for layer_idx in range(self.n_layers):
            messages_0 = torch.sparse.mm(dropped_adj, representations)
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
        self.n_heads = model_config['n_heads']
        self.dropout = model_config['dropout']
        self.feature_ratio = model_config['feature_ratio']
        self.norm_adj = self.generate_graph(model_config['dataset'])
        self.feat_mat, self.user_map, self.item_map = self.generate_feat(model_config['dataset'])

        self.embedding = nn.Embedding(self.feat_mat.shape[1], self.embedding_size)
        self.weight_q = nn.Linear(self.embedding_size, self.embedding_size * self.n_heads, bias=False)
        self.weight_k = nn.Linear(self.embedding_size, self.embedding_size * self.n_heads, bias=False)
        self.weight_v = nn.Linear(self.embedding_size, self.embedding_size * self.n_heads, bias=False)
        self.weight_o = nn.Linear(self.embedding_size * self.n_heads, self.embedding_size, bias=False)
        kaiming_uniform_(self.embedding.weight)
        kaiming_uniform_(self.weight_q.weight)
        kaiming_uniform_(self.weight_k.weight)
        kaiming_uniform_(self.weight_v.weight)
        kaiming_uniform_(self.weight_o.weight)
        self.to(device=self.device)

    def generate_graph(self, dataset):
        return LightGCN.generate_graph(self, dataset)

    def generate_feat(self, dataset, is_updating=False):
        if not is_updating:
            sub_mat = sp.coo_matrix((np.ones((len(dataset.train_array),)), np.array(dataset.train_array).T),
                                    shape=(self.n_users, self.n_items), dtype=np.float32)
            user_degree = np.array(np.sum(sub_mat, axis=1)).squeeze()
            popular_users = np.argsort(user_degree)[-int(self.n_users * self.feature_ratio):]
            print('User degree ratio {:.3f}%'.format(np.sum(user_degree[popular_users]) * 100. / np.sum(user_degree)))
            user_map = dict()
            for idx, user in enumerate(popular_users):
                user_map[user] = idx
            item_degree = np.array(np.sum(sub_mat, axis=0)).squeeze()
            popular_items = np.argsort(item_degree)[-int(self.n_items * self.feature_ratio):]
            print('Item degree ratio {:.3f}%'.format(np.sum(item_degree[popular_items]) * 100. / np.sum(item_degree)))
            item_map = dict()
            for idx, item in enumerate(popular_items):
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
        degree = np.array(np.sum(feat, axis=1)).squeeze()
        print('User feat number {:.3f}, Item feat number {:.3f}'
              .format(degree[:self.n_users].mean(), degree[self.n_users:].mean()))
        feat = normalize(feat, axis=1, norm='l1')
        feat = get_sparse_tensor(feat, self.device)
        return feat, user_map, item_map

    def inductive_rep_layer(self, feat_mat):
        embedding_q = self.weight_q(self.embedding.weight).view(-1, self.n_heads, self.embedding_size)
        embedding_k = self.weight_q(self.embedding.weight).view(-1, self.n_heads, self.embedding_size)
        embedding_v = self.weight_q(self.embedding.weight).view(-1, self.n_heads, self.embedding_size)
        row, column = feat_mat.indices()
        g = dgl.graph((column, row), num_nodes=max(self.feat_mat.shape), device=self.device)
        x_q = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=embedding_q, rhs_data=feat_mat.values())
        x_q = torch.index_select(x_q, 0, row)
        x_k = torch.index_select(embedding_k, 0, column)
        alpha = (x_q * x_k).sum(2)
        alpha = torch.stack(alpha, dim=0)
        row_max_alpha = dgl.ops.gspmm(g, 'copy_rhs', 'max', lhs_data=None, rhs_data=alpha)
        alpha = alpha - row_max_alpha[row, :]
        alpha = torch.exp(alpha)
        row_sum_alpha = dgl.ops.gspmm(g, 'copy_rhs', 'sum', lhs_data=None, rhs_data=alpha)
        alpha = alpha / row_sum_alpha[row, :]

        out = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=embedding_v, rhs_data=alpha.unsqueeze(-1))
        out = out[:self.feat_mat.shape[0], :, :] / torch.sqrt(self.embedding_size)
        out = self.weight_o(out.view(-1, self.n_heads * self.embedding_size))
        return out

    def get_rep(self):
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)
        all_layer_rep = [representations]
        dropped_adj = NGCF.dropout_sp_mat(self, self.norm_adj)
        for _ in range(self.n_layers):
            row, column = dropped_adj.indices()
            g = dgl.graph((column, row), num_nodes=self.feat_mat.shape[0], device=self.device)
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=dropped_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def bpr_forward(self, users, pos_items, neg_items):
        return NGCF.bpr_forward(self, users, pos_items, neg_items)

    def predict(self, users):
        return LightGCN.predict(self, users)

    def save(self, path):
        params = {'sate_dict': self.state_dict(), 'user_map': self.user_map, 'item_map': self.item_map}
        torch.save(params, path)

    def load(self, path):
        params = torch.load(path)
        self.load_state_dict(params['sate_dict'])
        self.user_map = params['user_map']
        self.item_map = params['item_map']
        self.feat_mat, _, _ = self.generate_feat(self.config['dataset'], is_updating=True)


class IMF(BasicModel):
    def __init__(self, model_config):
        super(IMF, self).__init__(model_config)
        self.embedding_size = model_config['embedding_size']
        self.feature_ratio = model_config['feature_ratio']
        self.dropout = model_config['dropout']
        self.feat_mat, self.user_map, self.item_map = self.generate_feat(model_config['dataset'])

        self.dense_layer = nn.Linear(self.feat_mat.shape[1], self.embedding_size)
        normal_(self.dense_layer.weight, std=0.1)
        zeros_(self.dense_layer.bias)
        self.to(device=self.device)

    def generate_feat(self, dataset, is_updating=False):
        return IGCN.generate_feat(self, dataset, is_updating)

    def get_rep(self):
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = torch.sparse.mm(feat_mat, self.dense_layer.weight.t()) + self.dense_layer.bias[None, :]
        return representations

    def bpr_forward(self, users, pos_items, neg_items):
        return IGCN.bpr_forward(self, users, pos_items, neg_items)

    def predict(self, users):
        return IGCN.predict(self, users)

    def save(self, path):
        IGCN.save(self, path)

    def load(self, path):
        IGCN.load(self, path)


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
            encoder_layer = nn.Linear(self.e_layer_sizes[layer_idx - 1], self.e_layer_sizes[layer_idx])
            self.encoder_layers.append(encoder_layer)
            decoder_layer = nn.Linear(self.d_layer_sizes[layer_idx - 1], self.d_layer_sizes[layer_idx])
            self.decoder_layers.append(decoder_layer)
        self.encoder_layers = nn.ModuleList(self.encoder_layers)
        self.decoder_layers = nn.ModuleList(self.decoder_layers)
        for layer in self.encoder_layers:
            kaiming_uniform_(layer.weight, nonlinearity='tanh')
            zeros_(layer.bias)
        for layer in self.decoder_layers:
            kaiming_uniform_(layer.weight, nonlinearity='tanh')
            zeros_(layer.bias)
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
        if scores.shape[1] < self.n_items:
            padding = torch.full([scores.shape[0], self.n_items - scores.shape[1]], -np.inf, device=self.device)
            scores = torch.cat([scores, padding], dim=1)
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

        normal_(self.mf_user_embedding.weight, std=0.1)
        normal_(self.mf_item_embedding.weight, std=0.1)
        normal_(self.mlp_user_embedding.weight, std=0.1)
        normal_(self.mlp_item_embedding.weight, std=0.1)
        self.init_mlp_layers()
        self.arch = 'gmf'
        self.to(device=self.device)

    def init_mlp_layers(self):
        for layer in self.mlp_layers:
            kaiming_uniform_(layer.weight, nonlinearity='leaky_relu')
            zeros_(layer.bias)
        ones_(self.output_layer.weight)

    def bce_forward(self, users, items):
        users_mf_e, items_mf_e = self.mf_user_embedding(users), self.mf_item_embedding(items)
        users_mlp_e, items_mlp_e = self.mlp_user_embedding(users), self.mlp_item_embedding(items)
        l2_norm_sq = torch.norm(users_mf_e, p=2, dim=1) ** 2 + torch.norm(items_mf_e, p=2, dim=1) ** 2 \
                     + torch.norm(users_mlp_e, p=2, dim=1) ** 2 + torch.norm(items_mlp_e, p=2, dim=1) ** 2

        mf_vectors = users_mf_e * items_mf_e
        mlp_vectors = torch.cat([users_mlp_e, items_mlp_e], dim=1)
        for layer in self.mlp_layers:
            mlp_vectors = F.leaky_relu(layer(mlp_vectors))
            l2_norm_sq += torch.norm(layer.weight, p=2)[None] ** 2

        if self.arch == 'gmf':
            vectors = [mf_vectors, torch.zeros_like(mlp_vectors, device=self.device, dtype=torch.float32)]
        elif self.arch == 'mlp':
            vectors = [torch.zeros_like(mf_vectors, device=self.device, dtype=torch.float32), mlp_vectors]
        else:
            vectors = [mf_vectors, mlp_vectors]
        predict_vectors = torch.cat(vectors, dim=1)
        scores = self.output_layer(predict_vectors).squeeze()
        l2_norm_sq += torch.norm(self.output_layer.weight, p=2)[None] ** 2
        return scores, l2_norm_sq

    def predict(self, users):
        items = torch.arange(self.n_items, dtype=torch.int64, device=self.device).repeat(users.shape[0])
        users = users[:, None].repeat(1, self.n_items).flatten()
        scores, _ = self.bce_forward(users, items)
        scores = scores.reshape(-1, self.n_items)
        return scores
