from dataset import get_dataset
from model import get_model
from trainer import get_trainer
import torch
from utils import init_run
from tensorboardX import SummaryWriter
from config import get_gowalla_config, get_yelp_config, get_amazon_config


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2021)

    device = torch.device('cuda')
    config = get_gowalla_config(device)
    dataset_config, model_config, trainer_config = config[6]
    dataset_config['path'] = dataset_config['path'][:-4] + '0_dropit'

    writer = SummaryWriter(log_path)
    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    trainer = get_trainer(trainer_config, dataset, model)
    trainer.train(verbose=True, writer=writer)
    writer.close()

    dataset_config['path'] = dataset_config['path'][:-7]
    new_dataset = get_dataset(dataset_config)
    model.config['dataset'] = new_dataset
    trainer = get_trainer(trainer_config, new_dataset, model)
    results, _ = trainer.eval('test')
    print('Previous interactions test result. {:s}'.format(results))

    model.feat_mat, _, _, model.row_sum = model.generate_feat(new_dataset, is_updating=True)
    model.update_feat_mat()
    results, _ = trainer.eval('test')
    print('Updated interactions test result. {:s}'.format(results))


if __name__ == '__main__':
    main()
