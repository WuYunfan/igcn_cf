import os
import numpy as np
from torch.utils.data import Dataset
import random
import sys


def get_dataset(config):
    config = config.copy()
    dataset = getattr(sys.modules['dataset'], config['name'])
    dataset = dataset(config)
    return dataset


class BasicDataset(Dataset):
    def __init__(self, dataset_config):
        print(dataset_config)
        self.config = dataset_config
        self.name = dataset_config['name']
        self.min_interactions = dataset_config.get('min_inter')
        self.split_ratio = dataset_config.get('split_ratio')
        self.device = dataset_config['device']
        self.negative_sample_ratio = dataset_config.get('neg_ratio', 1)
        self.n_users = 0
        self.n_items = 0
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.train_array = None
        print('init dataset ' + dataset_config['name'])

    def remove_sparse_ui(self, user_inter_set, item_inter_set):
        not_stop = True
        while not_stop:
            not_stop = False
            for user in user_inter_set:
                if len(user_inter_set[user]) < self.min_interactions:
                    not_stop = True
                    for item in user_inter_set[user]:
                        item_inter_set[item].remove(user)
                    user_inter_set.pop(user)
                    break
            for item in item_inter_set:
                if len(item_inter_set[item]) < self.min_interactions:
                    not_stop = True
                    for user in item_inter_set[item]:
                        user_inter_set[user].remove(item)
                    item_inter_set.pop(item)
                    break
        user_map = dict()
        for idx, user in enumerate(user_inter_set):
            user_map[user] = idx
        item_map = dict()
        for idx, item in enumerate(item_inter_set):
            item_map[item] = idx
        self.n_users = len(user_map)
        self.n_items = len(item_map)
        return user_map, item_map

    def generate_data(self, user_inter_lists):
        self.train_data = [[] for _ in range(self.n_users)]
        self.val_data = [[] for _ in range(self.n_users)]
        self.test_data = [[] for _ in range(self.n_users)]
        self.train_array = []
        average_inters = []
        for user in range(self.n_users):
            n_inter_items = len(user_inter_lists[user])
            average_inters.append(n_inter_items)
            n_train_items = int(n_inter_items * self.split_ratio[0])
            n_test_items = int(n_inter_items * self.split_ratio[2])
            self.train_data[user] += [item for item in user_inter_lists[user][:n_train_items]]
            self.val_data[user] += [item for item in user_inter_lists[user][n_train_items:-n_test_items]]
            self.test_data[user] += [item for item in user_inter_lists[user][-n_test_items:]]
            self.train_array.extend([[user, item] for item in self.train_data[user]])
        average_inters = np.mean(average_inters)
        print('Users {:d}, Items {:d}, Average number of interactions {:.3f}'
              .format(self.n_users, self.n_items, average_inters))

    def __len__(self):
        return len(self.train_array)

    def __getitem__(self, index):
        user = random.randint(0, self.n_users - 1)
        while not self.train_data[user]:
            user = random.randint(0, self.n_users - 1)
        pos_item = np.random.choice(self.train_data[user])
        data_with_negs = [[user, pos_item] for _ in range(self.negative_sample_ratio)]
        for idx in range(self.negative_sample_ratio):
            neg_item = random.randint(0, self.n_items - 1)
            while neg_item in self.train_data[user]:
                neg_item = random.randint(0, self.n_items - 1)
            data_with_negs[idx].append(neg_item)
        data_with_negs = np.array(data_with_negs, dtype=np.int64)
        return data_with_negs


class ML1MDataset(BasicDataset):
    def __init__(self, dataset_config):
        super(ML1MDataset, self).__init__(dataset_config)

        rating_file_path = os.path.join(dataset_config['path'], 'ratings.dat')
        user_inter_sets, item_inter_sets = dict(), dict()
        with open(rating_file_path, 'r') as f:
            lines = f.read().strip().split('\n')
        for line in lines:
            u, i, r, _ = line.split('::')
            u, i, r = int(u), int(i), int(r)
            if r < 4:
                continue
            if u in user_inter_sets:
                user_inter_sets[u].add(i)
            else:
                user_inter_sets[u] = {i}
            if i in item_inter_sets:
                item_inter_sets[i].add(u)
            else:
                item_inter_sets[i] = {u}
        user_map, item_map = self.remove_sparse_ui(user_inter_sets, item_inter_sets)

        user_inter_lists = [[] for _ in range(self.n_users)]
        for line in lines:
            u, i, r, t = line.split('::')
            u, i, r, t = int(u), int(i), int(r), int(t)
            if r > 3 and u in user_map and i in item_map:
                duplicate = False
                for i_t in user_inter_lists[user_map[u]]:
                    if i_t[0] == item_map[i]:
                        i_t[1] = min(i_t[1], t)
                        duplicate = True
                        break
                if not duplicate:
                    user_inter_lists[user_map[u]].append([item_map[i], t])
        for user in range(self.n_users):
            user_inter_lists[user].sort(key=lambda entry: entry[1])
            user_inter_lists[user] = [i_t[0] for i_t in user_inter_lists[user]]
        self.generate_data(user_inter_lists)


class LGCNDataset(BasicDataset):
    def __init__(self, dataset_config):
        super(LGCNDataset, self).__init__(dataset_config)

        val_ratio = dataset_config['val_ratio']
        train_file_path = os.path.join(dataset_config['path'], 'train.txt')
        test_file_path = os.path.join(dataset_config['path'], 'test.txt')
        self.n_items = 0
        self.train_data = self.read_data(train_file_path)
        self.test_data = self.read_data(test_file_path)
        assert len(self.train_data) == len(self.test_data)
        self.n_users = len(self.train_data)

        self.val_data = []
        for user in range(self.n_users):
            val_items = np.random.choice(self.train_data[user], int(len(self.train_data[user]) * val_ratio), replace=False)
            val_items = val_items.tolist()
            if 'ml1m' in dataset_config['path']:
                val_items = self.train_data[user][-int(len(self.train_data[user]) * val_ratio):]
            self.train_data[user] = list(set(self.train_data[user]) - set(val_items))
            self.val_data.append(val_items)

        self.train_array = []
        for user in range(self.n_users):
            self.train_array.extend([[user, item] for item in self.train_data[user]])

    def read_data(self, file_path):
        data = []
        with open(file_path, 'r') as f:
            lines = f.read().strip().split('\n')
        for line in lines:
            items = line.split(' ')[1:]
            items = [int(item) for item in items]
            if items:
                self.n_items = max(self.n_items, max(items) + 1)
            data.append(items)
        return data
