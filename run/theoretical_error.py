from dataset import get_dataset
import torch
from utils import init_run, generate_daj_mat, get_sparse_tensor, graph_rank_nodes
from config import get_gowalla_config, get_yelp_config, get_amazon_config
import numpy as np
from sortedcontainers import SortedList
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def plot_error(part_adj, u, ranked_users, ax, device, entity):
    sort_ranked_users, degree_ranked_users, pr_ranked_users = ranked_users
    sort_ranked_users = sort_ranked_users[::-1].copy()
    degree_ranked_users = degree_ranked_users[::-1].copy()
    pr_ranked_users = pr_ranked_users[::-1].copy()
    n_users = part_adj.shape[0]

    num_users = list(np.arange(0, n_users, 2000, dtype=np.int64)) + [n_users]
    num_users = np.array(num_users)
    errors_sort, errors_degree, errors_pr = [0.],  [0.],  [0.]
    Lu_sort = sp.dok_matrix((n_users, n_users), dtype=np.float32)
    Lu_dgree = sp.dok_matrix((n_users, n_users), dtype=np.float32)
    Lu_pr = sp.dok_matrix((n_users, n_users), dtype=np.float32)
    for nu in range(n_users):
        ui = sort_ranked_users[nu]
        Lu_sort[ui, ui] = 1.
        ui = degree_ranked_users[nu]
        Lu_dgree[ui, ui] = 1.
        ui = pr_ranked_users[nu]
        Lu_pr[ui, ui] = 1.
        if nu + 1 in num_users:
            l_tensor = torch.tensor(u, dtype=torch.float32, device=device)
            r_tensor = torch.tensor(sp.csr_matrix.dot(u.T, Lu_sort.dot(part_adj)),
                                    dtype=torch.float32, device=device)
            errors_sort.append(torch.norm(torch.mm(l_tensor, r_tensor), p=2).item() ** 2)

            r_tensor = torch.tensor(sp.csr_matrix.dot(u.T, Lu_dgree.dot(part_adj)),
                                    dtype=torch.float32, device=device)
            errors_degree.append(torch.norm(torch.mm(l_tensor, r_tensor), p=2).item() ** 2)

            r_tensor = torch.tensor(sp.csr_matrix.dot(u.T, Lu_pr.dot(part_adj)),
                                    dtype=torch.float32, device=device)
            errors_pr.append(torch.norm(torch.mm(l_tensor, r_tensor), p=2).item() ** 2)
            print((nu + 1) * 1. / n_users, errors_sort[-1], errors_degree[-1], errors_pr[-1])
    maxi = errors_sort[-1]
    errors_sort = np.array(errors_sort) / maxi
    errors_degree = np.array(errors_degree) / maxi
    errors_pr = np.array(errors_pr) / maxi
    ax.plot(num_users / n_users, errors_sort, label='error-sort', marker='v', color='red')
    ax.plot(num_users / n_users, errors_degree, label='degree', marker='o', color='blue')
    ax.plot(num_users / n_users, errors_pr, label='page rank', marker='d', color='orange')
    ax.set_xlabel('Ratio of non-template ' + entity, fontsize=17)
    if entity == 'users':
        ax.set_ylabel('Ratio of squared Frobenius \n norm of the error term', fontsize=17)
    ax.set_title('Different strategies to determine \n the template ' + entity, fontsize=17)
    ax.legend(fontsize=14, loc=2)


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2021)

    device = torch.device('cuda')
    config = get_gowalla_config(device)
    dataset_config, model_config, trainer_config = config[2]
    dataset_config['path'] = dataset_config['path'][:-4] + str(1)

    dataset = get_dataset(dataset_config)
    adj = generate_daj_mat(dataset)
    part_adj = adj[:dataset.n_users, dataset.n_users:]
    part_adj_tensor = get_sparse_tensor(part_adj, 'cpu')
    with torch.no_grad():
        u, s, v = torch.svd_lowrank(part_adj_tensor, 64)

    sort_ranked_users, sort_ranked_items = graph_rank_nodes(dataset, 'sort')
    degree_ranked_users, degree_ranked_items = graph_rank_nodes(dataset, 'degree')
    pr_ranked_users, pr_ranked_items = graph_rank_nodes(dataset, 'page_rank')
    ranked_users = (sort_ranked_users, degree_ranked_users, pr_ranked_users)
    ranked_items = (sort_ranked_items, degree_ranked_items, pr_ranked_items)
    pdf = PdfPages('figure_5.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(11, 4))
    axes = ax.flatten()
    plot_error(part_adj, u.cpu().numpy(), ranked_users, axes[0], device, 'users')
    plot_error(part_adj.T, v.cpu().numpy(), ranked_items, axes[1], device, 'items', )
    pdf.savefig()
    plt.close(fig)
    pdf.close()


if __name__ == '__main__':
    main()
