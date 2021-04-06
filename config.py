def get_gowalla_config(device):
    dataset_config = {'name': 'LGCNDataset', 'path': 'data/LGCN/gowalla',
                      'device': device, 'val_ratio': 0.1}

    gowalla_config = []
    model_config = {'name': 'MF', 'embedding_size': 64, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-3,
                      'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20, 100], 'device': device}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5,
                      'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20, 100], 'device': device}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IGCN', 'embedding_size': 64, 'n_layers': 3, 'dropout': 0.3,
                    'feature_ratio': 1., 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0.,
                      'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20, 100], 'device': device}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'ItemKNN', 'k': 1000, 'device': device}
    trainer_config = {'name': 'BasicTrainer',  'n_epochs': 0, 'test_batch_size': 512,
                      'topks': [20, 100], 'device': device}
    gowalla_config.append((dataset_config, model_config, trainer_config))
    return gowalla_config


def get_yelp_config(device):
    dataset_config = {'name': 'LGCNDataset', 'path': 'data/LGCN/yelp2018',
                      'device': device, 'val_ratio': 0.1}

    yelp_config = []
    model_config = {'name': 'MF', 'embedding_size': 64, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-3,
                      'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20, 100], 'device': device}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20, 100], 'device': device}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IGCN', 'embedding_size': 64, 'n_layers': 3, 'dropout': 0.3,
                    'feature_ratio': 1., 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0.,
                      'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20, 100], 'device': device}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'ItemKNN', 'k': 1000, 'device': device}
    trainer_config = {'name': 'BasicTrainer',  'n_epochs': 0, 'test_batch_size': 512,
                      'topks': [20, 100], 'device': device}
    yelp_config.append((dataset_config, model_config, trainer_config))
    return yelp_config


def get_ml1m_config(device):
    dataset_config = {'name': 'ML1MDataset', 'path': 'data/ML1M',
                      'device': device, 'split_ratio': [0.8, 0.1, 0.1], 'min_inter': 10}

    yelp_config = []
    model_config = {'name': 'MF', 'embedding_size': 64, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-2,
                      'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20, 100], 'device': device}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20, 100], 'device': device}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IGCN', 'embedding_size': 64, 'n_layers': 3, 'dropout': 0.3,
                    'feature_ratio': 1., 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5,
                      'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20, 100], 'device': device}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'ItemKNN', 'k': 1000, 'device': device}
    trainer_config = {'name': 'BasicTrainer',  'n_epochs': 0, 'test_batch_size': 512,
                      'topks': [20, 100], 'device': device}
    yelp_config.append((dataset_config, model_config, trainer_config))
    return yelp_config
