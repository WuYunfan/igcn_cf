from dataset import get_dataset
from model import get_model
from trainer import get_trainer
import torch
from utils import init_run, set_seed
from tensorboardX import SummaryWriter
from torch.nn.init import normal_, zeros_
from config import get_gowalla_config, get_yelp_config, get_ml1m_config


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2021)

    device = torch.device('cuda')
    config = get_gowalla_config(device)
    dataset_config, model_config, trainer_config = config[2]
    dataset_config['path'] = 'data/LGCN/gowalla_ui_0_8'

    writer = SummaryWriter(log_path)
    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    trainer = get_trainer(trainer_config, dataset, model)
    trainer.train(verbose=True, writer=writer)
    writer.close()

    set_seed(2021)
    dataset_config['path'] = 'data/LGCN/gowalla'
    new_dataset = get_dataset(dataset_config)
    model.config['dataset'] = new_dataset
    model.n_users, model.n_items = new_dataset.n_users, new_dataset.n_items
    model.norm_adj = model.generate_graph(new_dataset)
    model.feat_mat, _, _ = model.generate_feat(new_dataset, is_updating=True)
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

    writer = SummaryWriter(log_path)
    normal_(model.dense_layer.weight, std=0.1)
    zeros_(model.dense_layer.bias)
    trainer.train(verbose=True, writer=writer)
    writer.close()
    print('Default model with partial features')
    new_dataset.test_data = test_data.copy()
    results, _ = trainer.eval('test')
    print('All user test result. {:s}'.format(results))
    for user in range(dataset.n_users, new_dataset.n_users):
        new_dataset.test_data[user] = []
    results, _ = trainer.eval('test')
    print('Old user test result. {:s}'.format(results))
    new_dataset.test_data = test_data.copy()
    for user in range(dataset.n_users):
        new_dataset.test_data[user] = []
    results, _ = trainer.eval('test')
    print('New user test result. {:s}'.format(results))


if __name__ == '__main__':
    main()
