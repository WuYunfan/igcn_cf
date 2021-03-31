from dataset import get_dataset
from model import get_model
from trainer import get_trainer
import torch
from utils import init_run, set_seed
from tensorboardX import SummaryWriter


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2021)

    device = torch.device('cuda')
    dataset_config = {'name': 'LGCNDataset', 'path': 'data/LGCN/gowalla_ui_0_8',
                      'device': device, 'val_ratio': 0.1}
    model_config = {'name': 'IGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.5, 'feature_ratio': 1.}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}

    writer = SummaryWriter(log_path)
    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    trainer = get_trainer(trainer_config, dataset, model)
    trainer.train(verbose=True, writer=writer)
    writer.close()

    set_seed(2021)
    dataset_config = {'name': 'LGCNDataset', 'path': 'data/LGCN/gowalla',
                      'device': device, 'neg_ratio': 1, 'val_ratio': 0.1}
    new_dataset = get_dataset(dataset_config)
    model.n_users = new_dataset.n_users
    model.n_items = new_dataset.n_items
    model.norm_adj, model.feat_mat, _, _ = model.generate_graph(new_dataset, is_updating=True)
    trainer = get_trainer(trainer_config, new_dataset, model)

    default_model = get_model(model_config, new_dataset)
    default_model.load('checkpoints/default.pth')
    default_trainer = get_trainer(trainer_config, new_dataset, default_model)
    results, _ = trainer.eval('test')
    print('All user test result. {:s}'.format(results))
    results, _ = default_trainer.eval('test')
    print('Default model all user test result. {:s}'.format(results))

    test_data = new_dataset.test_data.copy()
    for user in range(dataset.n_users, new_dataset.n_users):
        new_dataset.test_data[user] = []
    results, _ = trainer.eval('test')
    print('Old user test result. {:s}'.format(results))
    results, _ = default_trainer.eval('test')
    print('Default model old user test result. {:s}'.format(results))

    new_dataset.test_data = test_data.copy()
    for user in range(dataset.n_users):
        new_dataset.test_data[user] = []
    results, _ = trainer.eval('test')
    print('New user test result. {:s}'.format(results))
    results, _ = default_trainer.eval('test')
    print('Default model new user test result. {:s}'.format(results))


if __name__ == '__main__':
    main()
