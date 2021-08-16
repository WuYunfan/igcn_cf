from dataset import get_dataset
from model import get_model
import torch
from config import get_gowalla_config
import numpy as np
import matplotlib.pyplot as plt
import dgl
import scipy.sparse as sp


def main():
    device = torch.device('cuda')
    config = get_gowalla_config(device)
    dataset_config, model_config, _ = config[2]

    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    model.load('checkpoints/IGCN_BPRTrainer_LGCNDataset_11.169.pth')
    with torch.no_grad():
        _, alpha = model.inductive_rep_layer(model.feat_mat)
        feat_l2_norm = torch.norm(model.embbeding.weigh, p=2, dim=1)
        row, column = model.feat_mat.indices()
        alpha = alpha * feat_l2_norm[column]
        g = dgl.graph((row, column), num_nodes=max(model.feat_mat.shape), device=model.device)
        node_alpha = dgl.ops.gspmm(g, 'copy_rhs', 'sum', lhs_data=None, rhs_data=alpha).cpu().numpy()[:-2]
        alpha = alpha.cpu().numpy()

    ranked_users = np.argsort(node_alpha[:model.n_users])[::-1].copy()
    ranked_items = np.argsort(node_alpha[model.n_users:])[::-1].copy()
    np.savez('core_ranking.npz', ranked_users=ranked_users, ranked_items=ranked_items)

    sub_mat = sp.coo_matrix((np.ones((len(dataset.train_array),)), np.array(dataset.train_array).T),
                            shape=(model.n_users, model.n_items), dtype=np.float32)
    user_degree = np.array(np.sum(sub_mat, axis=1)).squeeze()
    item_degree = np.array(np.sum(sub_mat, axis=0)).squeeze()
    selected_user = np.argsort(user_degree)[int(model.n_users * 0.9)]
    selected_item = np.argsort(item_degree)[int(model.n_items * 0.9)]
    user_alpha = alpha[row == selected_user]
    item_alpha = alpha[row == (model.n_users + selected_item)]

    eps = 0.005
    bins = np.arange(0., 1. + eps, eps)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    axes = ax.flatten()
    axes[0].hist(x=user_alpha, bins=bins, alpha=0.5)
    axes[0].set_xlim(0., 1.)
    axes[0].set_xlabel('Weight')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Attention weights around a user')
    axes[1].hist(x=item_alpha, bins=bins, alpha=0.5)
    axes[1].set_xlim(0., 1.)
    axes[1].set_xlabel('Weight')
    axes[1].set_ylabel('Frequency')
    axes[0].set_title('Attention weights around a item')
    plt.savefig('local_neighbors.png')
    plt.close(fig)

    fig, ax = plt.subplots(nrows=1, ncols=2)
    axes = ax.flatten()
    eps = 0.01
    bins = np.arange(0., 10. + eps, eps)
    axes[0].hist(x=node_alpha[:model.n_users], bins=bins, alpha=0.5)
    axes[0].set_xlim(0., 10.)
    axes[0].set_xlabel('Contribution')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Sum of user weights contributed to their interacted items')
    axes[1].hist(x=node_alpha[model.n_users:], bins=bins, alpha=0.5)
    axes[1].set_xlim(0., 10.)
    axes[1].set_xlabel('Contribution')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Sum of item weights contributed to their interacted users')
    plt.savefig('global_contribution.png')
    plt.close(fig)

    ratios = np.arange(0., 1. + eps, eps)
    alpha_ratios_user = []
    alpha_ratios_item = []
    for rat in ratios:
        n_users = int(model.n_users * rat)
        alpha_ratios_user.append(node_alpha[ranked_users[:n_users]].sum() / node_alpha[:model.n_users].sum())
        n_items = int(model.n_items * rat)
        alpha_ratios_item.append(node_alpha[model.n_users + ranked_items[:n_items]].sum() / node_alpha[model.n_users:].sum())
    axes[0].plot(ratios, alpha_ratios_user)
    axes[0].set_xlim(0., 1.)
    axes[0].set_ylim(0., 1.)
    axes[0].set_xlabel('Percentage of users')
    axes[0].set_ylabel('Percentage of contribution')
    axes[1].plot(ratios, alpha_ratios_item)
    axes[1].set_xlim(0., 1.)
    axes[1].set_ylim(0., 1.)
    axes[1].set_xlabel('Percentage of items')
    axes[1].set_ylabel('Percentage of contribution')
    plt.savefig('global_coverage.png')
    plt.close()


if __name__ == '__main__':
    main()
