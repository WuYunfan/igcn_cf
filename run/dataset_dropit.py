import numpy as np
import torch
from dataset import get_dataset
from utils import set_seed, output_dataset


def dropit_dataset(dataset, ratio):
    for user in range(dataset.n_users):
        num_items = int(len(dataset.train_data[user]) * ratio)
        dataset.train_data[user] = dataset.train_data[user][:num_items]


def main():
    device = torch.device('cuda')
    dataset_config = {'name': 'LGCNDataset', 'path': 'data/LGCN/gowalla',
                      'device': device, 'neg_ratio': 1, 'val_ratio': 0.}
    dataset = get_dataset(dataset_config)
    dropit_dataset(dataset, 0.8)
    output_dataset(dataset, 'data/LGCN/gowalla_it_0_8')


if __name__ == '__main__':
    main()