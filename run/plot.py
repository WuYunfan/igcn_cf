from dataset import get_dataset
from model import get_model
import torch
from config import get_gowalla_config
import numpy as np
import matplotlib.pyplot as plt
import dgl
import scipy.sparse as sp
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


def main():
    """
    device = torch.device('cpu')
    config = get_gowalla_config(device)
    dataset_config, model_config, _ = config[2]
    dataset_config['path'] = dataset_config['path'][:-4] + str(1)
    model_config['name'] = 'AttIGCN'

    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    model.eval()
    model.load('checkpoints/AttIGCN_BPRTrainer_ProcessedDataset_12.221.pth')
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
    print(user_alpha)
    print(item_alpha)

    pdf = PdfPages('figure_0.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(11, 3))
    axes = ax.flatten()
    eps = 5.e-5
    bins = np.arange(0., 1.e-2 + eps, eps)
    axes[0].hist(x=user_alpha, bins=bins, alpha=0.5)
    axes[0].set_xlim(0.5e-3, 3.5e-3)
    axes[0].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    axes[0].set_xlabel('Weight', fontsize=17)
    axes[0].set_ylabel('Frequency', fontsize=17)
    axes[0].set_title('Attention weights on the interactions of a user', fontsize=17)
    axes[0].tick_params(labelsize=14)

    eps = 2.e-5
    bins = np.arange(0., 1.e-2 + eps, eps)
    axes[1].hist(x=item_alpha, bins=bins, alpha=0.5)
    axes[1].set_xlim(0., 2.e-3)
    axes[1].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    axes[1].set_xlabel('Weight', fontsize=17)
    axes[1].set_ylabel('Frequency', fontsize=17)
    axes[1].set_title('Attention weights on the interactions of an item', fontsize=17)
    axes[1].tick_params(labelsize=14)
    pdf.savefig()
    plt.close(fig)
    pdf.close()

    pdf = PdfPages('figure_1.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(11, 3))
    axes = ax.flatten()
    eps = 0.1
    bins = np.arange(0., 10. + eps, eps)
    axes[0].hist(x=contribution[:model.n_users], bins=bins, alpha=0.5)
    axes[0].set_xlabel('Contribution', fontsize=17)
    axes[0].set_ylabel('Frequency', fontsize=17)
    axes[0].set_title('Global user contribution distribution', fontsize=17)
    axes[0].tick_params(labelsize=14)
    axes[0].set_xlim(0., 6.)

    eps = 0.05
    bins = np.arange(0., 5. + eps, eps)
    axes[1].hist(x=contribution[model.n_users:], bins=bins, alpha=0.5)
    axes[1].set_xlabel('Contribution', fontsize=17)
    axes[1].set_ylabel('Frequency', fontsize=17)
    axes[1].set_title('Global item contribution distribution', fontsize=17)
    axes[1].tick_params(labelsize=14)
    axes[1].set_xlim(0., 3.)
    pdf.savefig()
    plt.close(fig)
    pdf.close()
    """

    mf = [11.934] * 10
    imf_d = [8.925, 10.876, 12.014, 12.762, 13.251, 13.648, 13.775, 13.926, 14.096, 14.095]
    imf_nd = [9.289, 11.197, 12.335, 13, 13.512, 13.795, 13.888, 13.964, 14.164, 14.095]
    imf_pr = [9.344, 11.141, 12.289, 12.975, 13.336, 13.786, 13.879, 14.022, 14.101, 14.095]
    lgcn = [14.037] * 10
    igcn_d = [13.162, 14.231, 14.671, 14.963, 15.109, 15.253, 15.19, 15.344, 15.329, 15.341]
    igcn_nd = [13.416, 14.468, 15.008, 15.137, 15.241, 15.323, 15.329, 15.367, 15.406, 15.341]
    igcn_pr = [13.376, 14.37, 14.873, 15.09, 15.191, 15.321, 15.337, 15.38, 15.413, 15.341]
    ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    pdf = PdfPages('figure_2.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(11, 4))
    axes = ax.flatten()
    axes[0].plot(ratio, np.array(mf) / 100., label='MF', marker='s', color='green')
    axes[0].plot(ratio, np.array(imf_d) / 100., label='INMO-MF-degree', marker='o', color='blue')
    axes[0].plot(ratio, np.array(imf_nd) / 100., label='INMO-MF-error_sort', marker='v', color='red')
    axes[0].plot(ratio, np.array(imf_pr) / 100., label='INMO-MF-page_rank', marker='d', color='orange')
    axes[0].set_xticks(ratio)
    axes[0].legend(fontsize=13)
    axes[0].set_xlabel('Percentage of template users/items', fontsize=17)
    axes[0].set_ylabel('NDCG@20', fontsize=17)
    axes[0].set_title('INMO-MF', fontsize=17)
    axes[1].plot(ratio, np.array(lgcn) / 100., label='LightGCN', marker='s', color='green')
    axes[1].plot(ratio, np.array(igcn_d) / 100., label='INMO-LGCN-degree', marker='o', color='blue')
    axes[1].plot(ratio, np.array(igcn_nd) / 100., label='INMO-LGCN-error_sort', marker='v', color='red')
    axes[1].plot(ratio, np.array(igcn_pr) / 100., label='INMO-LGCN-page_rank', marker='d', color='orange')
    axes[1].set_xticks(ratio)
    axes[1].legend(fontsize=13)
    axes[1].set_xlabel('Percentage of template users/items', fontsize=17)
    axes[1].set_ylabel('NDCG@20', fontsize=17)
    axes[1].set_title('INMO-LGCN', fontsize=17)
    pdf.savefig()
    plt.close(fig)
    pdf.close()

    """
    imf = [13.848, 13.862, 13.906, 13.958, 13.751]
    igcn = [15.315, 15.378, 15.391, 15.172, 14.639]
    beta = ['0', '0.001', '0.01', '0.1', '1']
    pdf = PdfPages('figure_3.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(11, 4))
    axes = ax.flatten()
    axes[0].plot(np.array(imf) / 100., marker='s')
    axes[0].set_xticks([0, 1, 2, 3, 4])
    axes[0].set_xticklabels(beta)
    axes[0].set_xlabel('Weight of self-enhanced loss', fontsize=17)
    axes[0].set_ylabel('NDCG@20', fontsize=17)
    axes[0].set_title('INMO-MF', fontsize=17)
    axes[1].plot(np.array(igcn) / 100., marker='s')
    axes[1].set_xticks([0, 1, 2, 3, 4])
    axes[1].set_xticklabels(beta)
    axes[1].set_xlabel('Weight of self-enhanced loss', fontsize=17)
    axes[1].set_ylabel('NDCG@20', fontsize=17)
    axes[1].set_title('INMO-LGCN', fontsize=17)
    pdf.savefig()
    plt.close(fig)
    pdf.close()

    def read_csv_log(path):
        df = pd.read_csv(path)
        x = df['Step'].values
        y = df['Value'].values
        return x, y

    pdf = PdfPages('figure_4.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(11, 4))
    axes = ax.flatten()
    x, y = read_csv_log('/Users/wuyunfan/work/papers/21-3-code/igcn/logs/final/imf/gowalla/1/csv.csv')
    axes[0].plot(x, y, label='INMO-MF')
    x, y = read_csv_log('/Users/wuyunfan/work/papers/21-3-code/igcn/logs/ablation/anneal/imf/csv.csv')
    axes[0].plot(x, y, label='INMO-MF w/o NA')
    axes[0].set_xlabel('Epoch', fontsize=17)
    axes[0].set_ylabel('NDCG@20', fontsize=17)
    axes[0].set_title('INMO-MF', fontsize=17)
    axes[0].legend(fontsize=14, loc=4)
    x, y = read_csv_log('/Users/wuyunfan/work/papers/21-3-code/igcn/logs/final/igcn/gowalla/1/csv.csv')
    axes[1].plot(x, y, label='INMO-LGCN')
    x, y = read_csv_log('/Users/wuyunfan/work/papers/21-3-code/igcn/logs/ablation/anneal/igcn/csv.csv')
    axes[1].plot(x, y, label='INMO-LGCN w/o NA')
    axes[1].set_xlabel('Epoch', fontsize=17)
    axes[1].set_ylabel('NDCG@20', fontsize=17)
    axes[1].set_title('INMO-LGCN', fontsize=17)
    axes[1].legend(fontsize=14, loc=4)
    pdf.savefig()
    plt.close(fig)
    pdf.close()
    """


if __name__ == '__main__':
    main()
