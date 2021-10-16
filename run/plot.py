from dataset import get_dataset
from model import get_model
import torch
from config import get_gowalla_config
import numpy as np
import matplotlib.pyplot as plt
import dgl
import scipy.sparse as sp
from matplotlib.backends.backend_pdf import PdfPages


def main():
    device = torch.device('cpu')
    config = get_gowalla_config(device)
    dataset_config, model_config, _ = config[2]
    model_config['name'] = 'AttIGCN'

    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    model.load('checkpoints/')
    with torch.no_grad():
        alpha = model.inductive_rep_layer(model.feat_mat, return_alpha=True)
        row, column = model.feat_mat.indices()
        g = dgl.graph((row, column), num_nodes=max(model.feat_mat.shape), device=model.device)
        contribution = dgl.ops.gspmm(g, 'copy_rhs', 'sum', lhs_data=None, rhs_data=alpha)
        alpha = alpha.cpu().numpy()
        contribution = contribution[:-2].cpu().numpy()

    sub_mat = sp.coo_matrix((np.ones((len(dataset.train_array),)), np.array(dataset.train_array).T),
                            shape=(model.n_users, model.n_items), dtype=np.float32)
    user_degree = np.array(np.sum(sub_mat, axis=1)).squeeze()
    item_degree = np.array(np.sum(sub_mat, axis=0)).squeeze()
    ranked_users_degree = np.argsort(user_degree)[::-1].copy()
    ranked_items_degree = np.argsort(item_degree)[::-1].copy()
    selected_user = ranked_users_degree[2]
    selected_item = ranked_items_degree[2]
    user_alpha = alpha[row == selected_user]
    item_alpha = alpha[row == (model.n_users + selected_item)]

    pdf = PdfPages('plt_figures.pdf')

    fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
    axes = ax.flatten()
    eps = 1.e-6
    bins = np.arange(0., 5.e-4 + eps, eps)
    axes[0].hist(x=user_alpha, bins=bins, alpha=0.5)
    axes[0].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    axes[0].set_xlim(0., 5.e-4)
    axes[0].set_xlabel('Weight')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Attention weights around a user')
    eps = 1.e-7
    bins = np.arange(0., 2.e-5 + eps, eps)
    axes[1].hist(x=item_alpha, bins=bins, alpha=0.5)
    axes[1].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    axes[1].set_xlim(0., 2.e-5)
    axes[1].set_xlabel('Weight')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Attention weights around a item')
    pdf.savefig()
    plt.close(fig)

    fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
    axes = ax.flatten()
    eps = 0.3
    bins = np.arange(0., 100. + eps, eps)
    axes[0].hist(x=contribution[:model.n_users], bins=bins, alpha=0.5)
    axes[0].set_xlim(0., 100.)
    axes[0].set_xlabel('Contribution')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Global user contribution distribution')
    eps = 0.1
    bins = np.arange(0., 100. + eps, eps)
    axes[1].hist(x=contribution[model.n_users:], bins=bins, alpha=0.5)
    axes[1].set_xlim(0., 60.)
    axes[1].set_xlabel('Contribution')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Global item contribution distribution')
    pdf.savefig()
    plt.close(fig)

    pdf.close()


if __name__ == '__main__':
    main()
