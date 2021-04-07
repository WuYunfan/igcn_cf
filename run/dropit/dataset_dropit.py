import numpy as np
import torch
from dataset import get_dataset
from utils import set_seed, output_dataset


def dropit_dataset(dataset, ratio):
    for user in range(dataset.n_users):
        num_items = int(len(dataset.train_data[user]) * ratio)
        dropped_items = np.random.choice(dataset.train_data[user], size=num_items, replace=False)
        dataset.train_data[user] = list(set(dataset.train_data[user]) - set(dropped_items))


def main():
    set_seed(2021)
    device = torch.device('cuda')
    dataset_config = {'name': 'LGCNDataset', 'path': 'data/LGCN/gowalla',
                      'device': device, 'val_ratio': 0.}
    dataset = get_dataset(dataset_config)
    dropit_dataset(dataset, 0.8)
    output_dataset(dataset, 'data/LGCN/gowalla_it_0_8')


if __name__ == '__main__':
    main()