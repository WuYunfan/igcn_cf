import torch
import sys
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam, SGD
import time
import numpy as np
import os
from utils import AverageMeter, get_sparse_tensor
import torch.nn.functional as F
import scipy.sparse as sp
from dataset import AuxiliaryDataset


def get_trainer(config, dataset, model):
    config = config.copy()
    config['dataset'] = dataset
    config['model'] = model
    trainer = getattr(sys.modules['trainer'], config['name'])
    trainer = trainer(config)
    return trainer


class BasicTrainer:
    def __init__(self, trainer_config):
        print(trainer_config)
        self.config = trainer_config
        self.name = trainer_config['name']
        self.dataset = trainer_config['dataset']
        self.model = trainer_config['model']
        self.topks = trainer_config['topks']
        self.device = trainer_config['device']
        self.n_epochs = trainer_config['n_epochs']
        self.max_patience = trainer_config.get('max_patience', 50)
        self.val_interval = trainer_config.get('val_interval', 1)
        self.epoch = 0
        self.best_ndcg = -np.inf
        self.save_path = None
        self.opt = None

        test_user = TensorDataset(torch.arange(self.dataset.n_users, dtype=torch.int64, device=self.device))
        self.test_user_loader = DataLoader(test_user, batch_size=trainer_config['test_batch_size'])

    def initialize_optimizer(self):
        opt = getattr(sys.modules[__name__], self.config['optimizer'])
        self.opt = opt(self.model.parameters(), lr=self.config['lr'])

    def train_one_epoch(self):
        raise NotImplementedError

    def record(self, writer, stage, metrics):
        for metric in metrics:
            for k in self.topks:
                writer.add_scalar('{:s}_{:s}/{:s}_{:s}@{:d}'
                                  .format(self.model.name, self.name, stage, metric, k)
                                  , metrics[metric][k], self.epoch)

    def train(self, verbose=True, writer=None):
        if not self.model.trainable:
            results, metrics = self.eval('val')
            if verbose:
                print('Validation result. {:s}'.format(results))
            ndcg = metrics['NDCG'][self.topks[0]]
            return ndcg

        if not os.path.exists('checkpoints'): os.mkdir('checkpoints')
        patience = self.max_patience
        for self.epoch in range(self.n_epochs):
            start_time = time.time()
            self.model.train()
            loss = self.train_one_epoch()
            _, metrics = self.eval('train')
            consumed_time = time.time() - start_time
            if verbose:
                print('Epoch {:d}/{:d}, Loss: {:.6f}, Time: {:.3f}s'
                      .format(self.epoch, self.n_epochs, loss, consumed_time))
            if writer:
                writer.add_scalar('{:s}_{:s}/train_loss'.format(self.model.name, self.name), loss, self.epoch)
                self.record(writer, 'train', metrics)

            if (self.epoch + 1) % self.val_interval != 0:
                continue

            start_time = time.time()
            results, metrics = self.eval('val')
            consumed_time = time.time() - start_time
            if verbose:
                print('Validation result. {:s}Time: {:.3f}s'.format(results, consumed_time))
            if writer:
                self.record(writer, 'validation', metrics)

            ndcg = metrics['NDCG'][self.topks[0]]
            if ndcg > self.best_ndcg:
                if self.save_path:
                    os.remove(self.save_path)
                self.save_path = os.path.join('checkpoints', '{:s}_{:s}_{:s}_{:.3f}.pth'
                                              .format(self.model.name, self.name, self.dataset.name, ndcg * 100))
                self.best_ndcg = ndcg
                self.model.save(self.save_path)
                patience = self.max_patience
                print('Best NDCG, save model to {:s}'.format(self.save_path))
            else:
                patience -= self.val_interval
                if patience <= 0:
                    print('Early stopping!')
                    break
        self.model.load(self.save_path)
        return self.best_ndcg

    def calculate_metrics(self, eval_data, rec_items):
        results = {'Precision': {}, 'Recall': {}, 'NDCG': {}}
        hit_matrix = np.zeros_like(rec_items, dtype=np.float32)
        for user in range(rec_items.shape[0]):
            for item_idx in range(rec_items.shape[1]):
                if rec_items[user, item_idx] in eval_data[user]:
                    hit_matrix[user, item_idx] = 1.
        eval_data_len = np.array([len(items) for items in eval_data], dtype=np.int32)

        for k in self.topks:
            hit_num = np.sum(hit_matrix[:, :k], axis=1)
            precisions = hit_num / k
            with np.errstate(invalid='ignore'):
                recalls = hit_num / eval_data_len

            max_hit_num = np.minimum(eval_data_len, k)
            max_hit_matrix = np.zeros_like(hit_matrix[:, :k], dtype=np.float32)
            for user, num in enumerate(max_hit_num):
                max_hit_matrix[user, :num] = 1.
            denominator = np.log2(np.arange(2, k + 2, dtype=np.float32))[None, :]
            dcgs = np.sum(hit_matrix[:, :k] / denominator, axis=1)
            idcgs = np.sum(max_hit_matrix / denominator, axis=1)
            with np.errstate(invalid='ignore'):
                ndcgs = dcgs / idcgs

            user_masks = (max_hit_num > 0)
            results['Precision'][k] = precisions[user_masks].mean()
            results['Recall'][k] = recalls[user_masks].mean()
            results['NDCG'][k] = ndcgs[user_masks].mean()
        return results

    def eval(self, val_or_test, banned_items=None):
        self.model.eval()
        eval_data = getattr(self.dataset, val_or_test + '_data')
        rec_items = []
        with torch.no_grad():
            for users in self.test_user_loader:
                users = users[0]
                scores = self.model.predict(users)

                if val_or_test != 'train':
                    users = users.cpu().numpy().tolist()
                    exclude_user_indexes = []
                    exclude_items = []
                    for user_idx, user in enumerate(users):
                        items = self.dataset.train_data[user]
                        if val_or_test == 'test':
                            items = items + self.dataset.val_data[user]
                        exclude_user_indexes.extend([user_idx] * len(items))
                        exclude_items.extend(items)
                    scores[exclude_user_indexes, exclude_items] = -np.inf
                if banned_items is not None:
                    scores[:, banned_items] = -np.inf

                _, items = torch.topk(scores, k=max(self.topks))
                rec_items.append(items.cpu().numpy())

        rec_items = np.concatenate(rec_items, axis=0)
        metrics = self.calculate_metrics(eval_data, rec_items)

        precison = ''
        recall = ''
        ndcg = ''
        for k in self.topks:
            precison += '{:.3f}%@{:d}, '.format(metrics['Precision'][k] * 100., k)
            recall += '{:.3f}%@{:d}, '.format(metrics['Recall'][k] * 100., k)
            ndcg += '{:.3f}%@{:d}, '.format(metrics['NDCG'][k] * 100., k)
        results = 'Precision: {:s}Recall: {:s}NDCG: {:s}'.format(precison, recall, ndcg)
        return results, metrics

    def inductive_eval(self, n_old_users, n_old_items):
        test_data = self.dataset.test_data.copy()

        results, _ = self.eval('test')
        print('All users and all items result. {:s}'.format(results))

        for user in range(n_old_users, self.dataset.n_users):
            self.dataset.test_data[user] = []
        results, _ = self.eval('test')
        print('Old users and all items result. {:s}'.format(results))

        self.dataset.test_data = test_data.copy()
        for user in range(n_old_users):
            self.dataset.test_data[user] = []
        results, _ = self.eval('test')
        print('New users and all items result. {:s}'.format(results))

        self.dataset.test_data = test_data.copy()
        for user in range(self.dataset.n_users):
            test_items = np.array(self.dataset.test_data[user])
            self.dataset.test_data[user] = test_items[test_items < n_old_items].tolist()
        results, _ = self.eval('test', banned_items=np.arange(n_old_items, self.dataset.n_items))
        print('All users and old items result. {:s}'.format(results))

        self.dataset.test_data = test_data.copy()
        for user in range(self.dataset.n_users):
            test_items = np.array(self.dataset.test_data[user])
            self.dataset.test_data[user] = test_items[test_items >= n_old_items].tolist()
        results, _ = self.eval('test', banned_items=np.arange(n_old_items))
        print('All users and new items result. {:s}'.format(results))

        self.dataset.test_data = test_data.copy()
        for user in range(n_old_users, self.dataset.n_users):
            self.dataset.test_data[user] = []
        for user in range(n_old_users):
            test_items = np.array(self.dataset.test_data[user])
            self.dataset.test_data[user] = test_items[test_items < n_old_items].tolist()
        results, _ = self.eval('test', banned_items=np.arange(n_old_items, self.dataset.n_items))
        print('Old users and old items result. {:s}'.format(results))

        self.dataset.test_data = test_data.copy()


class BPRTrainer(BasicTrainer):
    def __init__(self, trainer_config):
        super(BPRTrainer, self).__init__(trainer_config)

        self.dataloader = DataLoader(self.dataset, batch_size=trainer_config['batch_size'],
                                     num_workers=trainer_config['dataloader_num_workers'])
        self.initialize_optimizer()
        self.l2_reg = trainer_config['l2_reg']

    def train_one_epoch(self):
        losses = AverageMeter()
        for batch_data in self.dataloader:
            inputs = batch_data[:, 0, :].to(device=self.device, dtype=torch.int64)
            users, pos_items, neg_items = inputs[:, 0],  inputs[:, 1],  inputs[:, 2]

            users_r, pos_items_r, neg_items_r, l2_norm_sq = self.model.bpr_forward(users, pos_items, neg_items)
            pos_scores = torch.sum(users_r * pos_items_r, dim=1)
            neg_scores = torch.sum(users_r * neg_items_r, dim=1)

            bpr_loss = F.softplus(neg_scores - pos_scores).mean()
            reg_loss = self.l2_reg * l2_norm_sq.mean()
            loss = bpr_loss + reg_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.update(loss.item(), inputs.shape[0])
        return losses.avg


class IDCFTrainer(BasicTrainer):
    def __init__(self, trainer_config):
        super(IDCFTrainer, self).__init__(trainer_config)

        self.dataloader = DataLoader(self.dataset, batch_size=trainer_config['batch_size'],
                                     num_workers=trainer_config['dataloader_num_workers'])
        self.initialize_optimizer()
        self.l2_reg = trainer_config['l2_reg']
        self.contrastive_reg = trainer_config['contrastive_reg']

    def train_one_epoch(self):
        losses = AverageMeter()
        for batch_data in self.dataloader:
            inputs = batch_data[:, 0, :].to(device=self.device, dtype=torch.int64)
            users, pos_items, neg_items = inputs[:, 0],  inputs[:, 1],  inputs[:, 2]

            users_r, pos_items_r, neg_items_r, l2_norm_sq, contrastive_loss = self.model.bpr_forward(users, pos_items, neg_items)
            pos_scores = torch.sum(users_r * pos_items_r, dim=1)
            neg_scores = torch.sum(users_r * neg_items_r, dim=1)

            bpr_loss = F.softplus(neg_scores - pos_scores).mean()
            reg_loss = self.l2_reg * l2_norm_sq.mean() + self.contrastive_reg * contrastive_loss.mean()
            loss = bpr_loss + reg_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.update(loss.item(), inputs.shape[0])
        return losses.avg


class IGCNTrainer(BasicTrainer):
    def __init__(self, trainer_config):
        super(IGCNTrainer, self).__init__(trainer_config)

        self.dataloader = DataLoader(self.dataset, batch_size=trainer_config['batch_size'],
                                     num_workers=trainer_config['dataloader_num_workers'])
        self.aux_dataloader = DataLoader(AuxiliaryDataset(self.dataset, self.model.user_map, self.model.item_map),
                                         batch_size=trainer_config['batch_size'],
                                         num_workers=trainer_config['dataloader_num_workers'])
        self.initialize_optimizer()
        self.l2_reg = trainer_config['l2_reg']
        self.aux_reg = trainer_config['aux_reg']

    def train_one_epoch(self):
        losses = AverageMeter()
        for batch_data, a_batch_data in zip(self.dataloader, self.aux_dataloader):
            inputs = batch_data[:, 0, :].to(device=self.device, dtype=torch.int64)
            users, pos_items, neg_items = inputs[:, 0],  inputs[:, 1],  inputs[:, 2]
            users_r, pos_items_r, neg_items_r, l2_norm_sq = self.model.bpr_forward(users, pos_items, neg_items)
            pos_scores = torch.sum(users_r * pos_items_r, dim=1)
            neg_scores = torch.sum(users_r * neg_items_r, dim=1)
            bpr_loss = F.softplus(neg_scores - pos_scores).mean()

            inputs = a_batch_data[:, 0, :].to(device=self.device, dtype=torch.int64)
            users, pos_items, neg_items = inputs[:, 0],  inputs[:, 1],  inputs[:, 2]
            users_r = self.model.embedding(users)
            pos_items_r = self.model.embedding(pos_items + len(self.model.user_map))
            neg_items_r = self.model.embedding(neg_items + len(self.model.user_map))
            pos_scores = torch.sum(users_r * pos_items_r * self.model.w[None, :], dim=1)
            neg_scores = torch.sum(users_r * neg_items_r * self.model.w[None, :], dim=1)
            aux_loss = F.softplus(neg_scores - pos_scores).mean()

            reg_loss = self.l2_reg * l2_norm_sq.mean() + self.aux_reg * aux_loss
            loss = bpr_loss + reg_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.update(loss.item(), inputs.shape[0])
        self.model.feat_mat_anneal()
        return losses.avg


class BCETrainer(BasicTrainer):
    def __init__(self, trainer_config):
        super(BCETrainer, self).__init__(trainer_config)

        self.dataloader = DataLoader(self.dataset, batch_size=trainer_config['batch_size'],
                                     num_workers=trainer_config['dataloader_num_workers'])
        self.initialize_optimizer()
        self.l2_reg = trainer_config['l2_reg']
        self.mf_pretrain_epochs = trainer_config['mf_pretrain_epochs']
        self.mlp_pretrain_epochs = trainer_config['mlp_pretrain_epochs']

    def train_one_epoch(self):
        if self.epoch == self.mf_pretrain_epochs:
            self.model.arch = 'mlp'
            self.initialize_optimizer()
            self.best_ndcg = -np.inf
            self.model.load(self.save_path)
        if self.epoch == self.mf_pretrain_epochs + self.mlp_pretrain_epochs:
            self.model.arch = 'neumf'
            self.initialize_optimizer()
            self.best_ndcg = -np.inf
            self.model.load(self.save_path)
            self.model.init_mlp_layers()
        losses = AverageMeter()
        for batch_data in self.dataloader:
            inputs = batch_data.to(device=self.device, dtype=torch.int64)
            users, pos_items = inputs[:, 0, 0], inputs[:, 0, 1]
            logits, l2_norm_sq_p = self.model.bce_forward(users, pos_items)
            bce_loss_p = F.softplus(-logits)

            inputs = inputs.reshape(-1, 3)
            users, neg_items = inputs[:, 0], inputs[:, 2]
            logits, l2_norm_sq_n = self.model.bce_forward(users, neg_items)
            bce_loss_n = F.softplus(logits)

            bce_loss = torch.cat([bce_loss_p, bce_loss_n], dim=0).mean()
            l2_norm_sq = torch.cat([l2_norm_sq_p, l2_norm_sq_n], dim=0)
            reg_loss = self.l2_reg * l2_norm_sq.mean()
            loss = bce_loss + reg_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.update(loss.item(), l2_norm_sq.shape[0])
        return losses.avg


class MLTrainer(BasicTrainer):
    def __init__(self, trainer_config):
        super(MLTrainer, self).__init__(trainer_config)

        train_user = TensorDataset(torch.arange(self.dataset.n_users, dtype=torch.int64, device=self.device))
        self.train_user_loader = DataLoader(train_user, batch_size=trainer_config['batch_size'], shuffle=True)
        self.data_mat = sp.coo_matrix((np.ones((len(self.dataset.train_array),)), np.array(self.dataset.train_array).T),
                                      shape=(self.dataset.n_users, self.dataset.n_items), dtype=np.float32).tocsr()
        self.initialize_optimizer()
        self.l2_reg = trainer_config['l2_reg']
        self.kl_reg = trainer_config['kl_reg']

    def train_one_epoch(self):
        kl_reg = min(self.kl_reg, 1. * self.epoch / self.n_epochs)

        losses = AverageMeter()
        for users in self.train_user_loader:
            users = users[0]

            scores, kl, l2_norm_sq = self.model.ml_forward(users)
            scores = F.log_softmax(scores, dim=1)
            users = users.cpu().numpy()
            profiles = self.data_mat[users, :]
            profiles = get_sparse_tensor(profiles, self.device).to_dense()
            ml_loss = -torch.sum(profiles * scores, dim=1).mean()

            reg_loss = kl_reg * kl.mean() + self.l2_reg * l2_norm_sq.mean()
            loss = ml_loss + reg_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.update(loss.item(), users.shape[0])
        return losses.avg

