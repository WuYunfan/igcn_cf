import numpy as np
import torch
from dataset import get_dataset


def dropit_dataset(dataset, ratio):
    for user in range(dataset.n_users):
        num_items = int(len(dataset.train_data[user]) * ratio)
        dataset.train_data[user] = dataset.train_data[user][:num_items]


def main():
    for i in range(5):
        n = str(i)
        device = torch.device('cuda')
        dataset_config = {'name': 'ProcessedDataset', 'path': 'data/Amazon/' + n,
                          'device': device}
        dataset = get_dataset(dataset_config)
        dropit_dataset(dataset, 0.8)
        dataset.output_dataset('data/Amazon/' + n + '_dropit')


if __name__ == '__main__':
    main()