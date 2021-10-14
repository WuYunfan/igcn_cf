from sklearn.model_selection import ParameterGrid
import torch
import numpy as np
from dataset import get_dataset
from utils import set_seed, init_run
from model import get_model
from trainer import get_trainer


def fitness(lr, l2_reg, dropout, aux_reg):
    set_seed(2021)
    device = torch.device('cuda')
    dataset_config = {'name': 'ProcessedDataset', 'path': 'data/Gowalla/time',
                      'device': device}
    model_config = {'name': 'IGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': dropout, 'feature_ratio': 1.}
    trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': lr, 'l2_reg': l2_reg, 'aux_reg': aux_reg,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    trainer = get_trainer(trainer_config, dataset, model)
    return trainer.train(verbose=True)


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2021)
    param_grid = {'lr': [1.e-3], 'l2_reg': [0., 1.e-5], 'dropout': [0., 0.1, 0.3],
                  'aux_reg': [1.e-3, 1.e-2, 1.e-1]}
    grid = ParameterGrid(param_grid)
    max_ndcg = -np.inf
    best_params = None
    for params in grid:
        ndcg = fitness(params['lr'], params['l2_reg'], params['dropout'], params['aux_reg'])
        print('NDCG: {:.3f}, Parameters: {:s}'.format(ndcg, str(params)))
        if ndcg > max_ndcg:
            max_ndcg = ndcg
            best_params = params
            print('Maximum NDCG!')
    print('Maximum NDCG: {:.3f}, Best Parameters: {:s}'.format(max_ndcg, str(best_params)))


if __name__ == '__main__':
    main()
