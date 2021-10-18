import numpy as np
import torch
from dataset import get_dataset
from utils import set_seed


def resize_dataset(dataset, ratio):
    n_users = int(dataset.n_users * ratio)
    n_items = int(dataset.n_items * ratio)
    for user in range(n_users):
        train_items = np.array(dataset.train_data[user])
        train_items = train_items[train_items < n_items]
        dataset.train_data[user] = train_items.tolist()

        val_items = np.array(dataset.val_data[user])
        val_items = val_items[val_items < n_items]
        dataset.val_data[user] = val_items.tolist()

        test_items = np.array(dataset.test_data[user])
        test_items = test_items[test_items < n_items]
        dataset.test_data[user] = test_items.tolist()

    dataset.n_users = n_users
    dataset.n_items = n_items
    dataset.train_data = dataset.train_data[:n_users]
    dataset.val_data = dataset.val_data[:n_users]
    dataset.test_data = dataset.test_data[:n_users]


def main():
    for i in range(5):
        n = str(i)
        device = torch.device('cuda')
        dataset_config = {'name': 'ProcessedDataset', 'path': 'data/Amazon/' + n,
                          'device': device}
        dataset = get_dataset(dataset_config)
        resize_dataset(dataset, 0.8)
        dataset.output_dataset('data/Amazon/' + n + '_dropui')


if __name__ == '__main__':
    main()