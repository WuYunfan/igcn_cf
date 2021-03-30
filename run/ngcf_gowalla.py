from dataset import get_dataset
from model import get_model
from trainer import get_trainer
import torch
from utils import init_run
from tensorboardX import SummaryWriter


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2021)

    device = torch.device('cuda')
    dataset_config = {'name': 'LGCNDataset', 'path': 'data/LGCN/gowalla',
                      'device': device, 'neg_ratio': 1, 'val_ratio': 0.1}
    model_config = {'name': 'NGCF', 'embedding_size': 64, 'layer_sizes': [64, 64, 64],
                    'drop_out': 0.1, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-4, 'l2_reg': 5.e-6,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}

    writer = SummaryWriter(log_path)
    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    trainer = get_trainer(trainer_config, dataset, model)
    trainer.train(verbose=True, writer=writer)
    writer.close()
    results, _ = trainer.eval('test')
    print('Test result. {:s}'.format(results))


if __name__ == '__main__':
    main()
