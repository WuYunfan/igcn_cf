from dataset import get_dataset
from model import get_model
from trainer import get_trainer
import torch
from utils import init_run, set_seed
from tensorboardX import SummaryWriter
from torch.nn.init import normal_, zeros_
from config import get_gowalla_config, get_yelp_config, get_ml1m_config
import numpy as np
import scipy.sparse as sp


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2021)

    device = torch.device('cuda')
    config = get_gowalla_config(device)
    dataset_config, model_config, trainer_config = config[5]
    dataset_config['path'] = 'data/LGCN/gowalla_ui_0_8'

    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)

    set_seed(2021)
    dataset_config['path'] = 'data/LGCN/gowalla_shuffled'
    new_dataset = get_dataset(dataset_config)
    model.config['dataset'] = new_dataset
    model.n_users, model.n_items = new_dataset.n_users, new_dataset.n_items
    data_mat = model.get_data_mat(new_dataset)[:, dataset.n_items]
    sq_sum = np.array(np.sum(data_mat ** 2, axis=1)).squeeze()
    sq_sum[sq_sum == 0.] = 1.
    quotient = np.power(sq_sum, -0.5)
    d_mat = sp.diags(quotient, format='csr', dtype=np.float32)
    model.normalized_data_mat = d_mat.dot(data_mat)
    trainer = get_trainer(trainer_config, new_dataset, model)
    trainer.inductive_eval(dataset.n_users, dataset.n_items)


if __name__ == '__main__':
    main()
