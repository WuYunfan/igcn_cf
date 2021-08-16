from dataset import get_dataset
from model import get_model
import torch
from config import get_gowalla_config
import numpy as np
import matplotlib.pyplot as plt
import dgl


def main():
    device = torch.device('cpu')
    config = get_gowalla_config(device)
    dataset_config, model_config, _ = config[2]
    dataset_config = {'name': 'LGCNDataset', 'path': 'data/LGCN/gowalla',
                      'device': device, 'val_ratio': 0.1}

    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    model.load('checkpoints/IGCN_BPRTrainer_LGCNDataset_11.169.pth')
    with torch.no_grad():
        _, alpha_t = model.inductive_rep_layer(model.feat_mat)
    alpha = alpha_t.cpu().numpy()
    normalized_edge = model.feat_mat.values().cpu().numpy()
    eps = 0.005
    bins = np.arange(0., 0.3 + eps, eps)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    #axes = ax.flatten()
    ax.hist(x=alpha, bins=bins, alpha=0.5)
    #axes[1].hist(x=normalized_edge, bins=bins, alpha=0.5)
    #axes[0].set_xlim(0., 0.3)
    #axes[0].set_ylim(0., 80000.)
    #axes[1].set_xlim(0., 0.3)
    #axes[1].set_ylim(0., 80000.)
    plt.savefig('edge.png')
    plt.close()

    row, column = model.feat_mat.indices()
    g = dgl.graph((row, column), num_nodes=max(model.feat_mat.shape), device=model.device)
    column_sum_alpha = dgl.ops.gspmm(g, 'copy_rhs', 'sum', lhs_data=None, rhs_data=alpha_t).cpu().numpy()
    column_sum_alpha = np.sort(column_sum_alpha)[::-1].copy()
    all_sum_alpha = np.sum(column_sum_alpha)

    fig, ax = plt.subplots(nrows=1, ncols=2)
    axes = ax.flatten()
    eps = 0.01
    bins = np.arange(0., 5. + eps, eps)
    axes[0].hist(x=column_sum_alpha, bins=bins, alpha=0.5)
    ratios = np.arange(eps, 1. + eps, eps)
    alpha_ratios = []
    for rat in ratios:
        n_feats = int(model.feat_mat.shape[1] * rat)
        alpha_ratios.append(column_sum_alpha[:n_feats].sum() / all_sum_alpha)
    axes[1].plot(ratios, alpha_ratios)
    plt.savefig('node.png')
    plt.close()


if __name__ == '__main__':
    main()
