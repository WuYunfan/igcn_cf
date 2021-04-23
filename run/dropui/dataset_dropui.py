import numpy as np
import torch
from dataset import get_dataset
from utils import set_seed, output_dataset


def shuffle_dataset(dataset):
    u_permutation = np.random.permutation(dataset.n_users)
    i_permutation = np.random.permutation(dataset.n_items)
    train_data = [[] for _ in range(dataset.n_users)]
    test_data = [[] for _ in range(dataset.n_users)]
    for o_user in range(dataset.n_users):
        train_items = [i_permutation[item] for item in dataset.train_data[o_user]]
        train_data[u_permutation[o_user]] = train_items

        test_items = [i_permutation[item] for item in dataset.test_data[o_user]]
        test_data[u_permutation[o_user]] = test_items
    dataset.train_data = train_data
    dataset.test_data = test_data


def resize_dataset(dataset, ratio):
    n_users = int(dataset.n_users * ratio)
    n_items = int(dataset.n_items * ratio)
    for user in range(n_users):
        train_items = np.array(dataset.train_data[user])
        train_items = train_items[train_items < n_items]
        dataset.train_data[user] = train_items.tolist()

        test_items = np.array(dataset.test_data[user])
        test_items = test_items[test_items < n_items]
        dataset.test_data[user] = test_items.tolist()

    dataset.n_users = n_users
    dataset.n_items = n_items
    dataset.train_data = dataset.train_data[:n_users]
    dataset.test_data = dataset.test_data[:n_users]


def main():
    set_seed(2021)

    device = torch.device('cuda')
    dataset_config = {'name': 'LGCNDataset', 'path': 'data/LGCN/gowalla',
                      'device': device, 'val_ratio': 0.}
    dataset = get_dataset(dataset_config)
    shuffle_dataset(dataset)
    output_dataset(dataset, 'data/LGCN/gowalla_shuffled')
    resize_dataset(dataset, 0.8)
    output_dataset(dataset, 'data/LGCN/gowalla_ui_0_8')


if __name__ == '__main__':
    main()