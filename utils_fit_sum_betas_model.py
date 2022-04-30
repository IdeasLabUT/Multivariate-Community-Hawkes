import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import utils_sum_betas_bp as sum_betas_bp


#%% mulch (sum of kernel) log-likelihood functions

def model_LL_kernel_sum(param, events_dict, node_mem, k, end_time, ref=False):
    block_pair_M, n_nodes_c = num_nodes_pairs_per_block_pair(node_mem, k)
    events_dict_block_pair = events_dict_to_events_dict_bp(events_dict, node_mem, k)
    if len(param) == 5:
        return model_LL_2_alpha_kernel_sum(param, events_dict_block_pair, end_time, block_pair_M, n_nodes_c, k, ref)
    elif len(param) == 7:
        return model_LL_4_alpha_kernel_sum(param, events_dict_block_pair, end_time, block_pair_M, n_nodes_c, k, ref)
    elif len(param) == 9:
        return model_LL_6_alpha_kernel_sum(param, events_dict_block_pair, end_time, block_pair_M, n_nodes_c, k, ref)

# different alpha-versions of sum of kernel model
def model_LL_2_alpha_kernel_sum(params_tup, events_dict_bp, end_time, M_bp, n_nodes_c, n_classes, ref=False):
    mu_bp, alpha_n_bp, alpha_r_bp, C_bp, betas = params_tup
    LL_bp = np.zeros((n_classes, n_classes))
    num_events = 0
    for i in range(n_classes):
        for j in range(n_classes):
            par = (mu_bp[i, j], alpha_n_bp[i, j], alpha_r_bp[i, j], C_bp[i, j, :], betas)
            if i == j:  # diagonal block pair
                ll_dia = sum_betas_bp.LL_2_alpha_kernel_sum_dia(par, events_dict_bp[i][j], end_time, n_nodes_c[j, 0], M_bp[i, j])
                LL_bp[i, j] = ll_dia
            else:  # off-diagonal block pair
                ll_off = sum_betas_bp.LL_2_alpha_kernel_sum_off(par, events_dict_bp[i][j], events_dict_bp[j][i], end_time,
                                                               n_nodes_c[j, 0], M_bp[i, j])
                LL_bp[i, j] = ll_off
            # number of event of block_pair
            num_events += sum_betas_bp.cal_num_events(events_dict_bp[i][j])
    if ref:
        return LL_bp, num_events
    else:
        return np.sum(LL_bp), num_events
def model_LL_4_alpha_kernel_sum(params_tup, events_dict_bp, end_time, M_bp, n_nodes_c, n_classes, ref=False):
    mu_bp, alpha_n_bp, alpha_r_bp, alpha_br_bp, alpha_gr_bp, C_bp, betas = params_tup
    LL_bp = np.zeros((n_classes, n_classes))
    num_events = 0
    for i in range(n_classes):
        for j in range(n_classes):
            par = (mu_bp[i, j], alpha_n_bp[i, j], alpha_r_bp[i, j], alpha_br_bp[i, j], alpha_gr_bp[i, j],
                   C_bp[i, j],betas)
            if i == j:  # diagonal block pair
                ll_dia = sum_betas_bp.LL_4_alpha_kernel_sum_dia(par, events_dict_bp[i][j], end_time, n_nodes_c[j, 0], M_bp[i, j])
                LL_bp[i, j] = ll_dia
            else:  # off-diagonal block pair
                ll_off = sum_betas_bp.LL_4_alpha_kernel_sum_off(par, events_dict_bp[i][j], events_dict_bp[j][i], end_time,
                                                               n_nodes_c[j, 0], M_bp[i, j])
                LL_bp[i, j] = ll_off
            # number of event of block_pair
            num_events += sum_betas_bp.cal_num_events(events_dict_bp[i][j])
    if ref:
        return LL_bp, num_events
    else:
        return np.sum(LL_bp), num_events
def model_LL_6_alpha_kernel_sum(params_tup, events_dict_bp, end_time, M_bp, n_nodes_c, n_classes, ref=False):
    mu_bp, alpha_n_bp, alpha_r_bp, alpha_br_bp, alpha_gr_bp, alpha_al_bp, alpha_alr_bp, C_bp, betas = params_tup
    LL_bp = np.zeros((n_classes, n_classes))
    num_events = 0
    for i in range(n_classes):
        for j in range(n_classes):
            par = (mu_bp[i, j], alpha_n_bp[i, j], alpha_r_bp[i, j], alpha_br_bp[i, j], alpha_gr_bp[i, j],
                   alpha_al_bp[i, j], alpha_alr_bp[i, j], C_bp[i, j], betas)
            if i == j:  # diagonal block pair
                ll_dia = sum_betas_bp.LL_6_alpha_kernel_sum_dia(par, events_dict_bp[i][j], end_time, n_nodes_c[j, 0], M_bp[i, j])
                LL_bp[i, j] = ll_dia
            else:  # off-diagonal block pair
                ll_off = sum_betas_bp.LL_6_alpha_kernel_sum_off(par, events_dict_bp[i][j], events_dict_bp[j][i], end_time,
                                                               n_nodes_c[j, 0], M_bp[i, j])
                LL_bp[i, j] = ll_off
            # number of event of block_pair
            num_events += sum_betas_bp.cal_num_events(events_dict_bp[i][j])
    if ref:
        return LL_bp, num_events
    else:
        return np.sum(LL_bp), num_events


#%% MULCH (sum of kernel) full model fit functions
def model_fit_kernel_sum(n_alpha, events_dict, node_mem, n_classes, end_time, betas, ref=False):
    if n_alpha == 2:
        return fit_2_alpha_kernel_sum(events_dict, node_mem, n_classes, end_time, betas, ref)
    elif n_alpha == 4:
        return fit_4_alpha_kernel_sum(events_dict, node_mem, n_classes, end_time, betas, ref)
    elif n_alpha == 6:
        return fit_6_alpha_kernel_sum(events_dict, node_mem, n_classes, end_time, betas, ref)
    else:
        print(" number of alpha parameter should be 2, 4, or 6")

# different alpha-versions of sum of kernel model
def fit_2_alpha_kernel_sum(events_dict, node_mem, n_classes, end_time, betas, ref=False):
    # return number of node pairs within a block pair, number of nodes per class
    block_pair_M_train, n_nodes_c = num_nodes_pairs_per_block_pair(node_mem, n_classes)
    block_pairs_train = events_dict_to_events_dict_bp(events_dict, node_mem, n_classes)
    # initialize paramters matrices
    mu_bp = np.zeros((n_classes, n_classes))
    alpha_n_bp = np.zeros((n_classes, n_classes))
    alpha_r_bp = np.zeros((n_classes,n_classes))
    C_bp = np.zeros((n_classes, n_classes, np.size(betas)))
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:  # diagonal block pair
                params_est = sum_betas_bp.fit_2_alpha_kernel_sum_dia(block_pairs_train[i][j], end_time, n_nodes_c[j, 0],
                                                                    block_pair_M_train[i, j], betas)
            else:   # off-diagonal block pair
                params_est = sum_betas_bp.fit_2_alpha_kernel_sum_off(block_pairs_train[i][j], block_pairs_train[j][i],
                                                                    end_time, n_nodes_c[j,0], block_pair_M_train[i, j], betas)
            mu_bp[i, j] = params_est[0]
            alpha_n_bp[i, j] = params_est[1]
            alpha_r_bp[i, j] = params_est[2]
            C_bp[i,j,:] = params_est[3]
    # calclate log-likelihood on train, test, all datasets
    params_tuple = (mu_bp, alpha_n_bp, alpha_r_bp, C_bp, betas)
    ll_train, num_events_train = model_LL_2_alpha_kernel_sum(params_tuple, block_pairs_train, end_time,
                                                                  block_pair_M_train, n_nodes_c, n_classes, ref)
    if ref:
        return params_tuple, ll_train, num_events_train, block_pairs_train, n_nodes_c
    else:
        return params_tuple, ll_train, num_events_train
def fit_4_alpha_kernel_sum(events_dict, node_mem, n_classes, end_time, betas, ref=False):
    # return number of node pairs within a block pair, number of nodes per class
    block_pair_M_train, n_nodes_c = num_nodes_pairs_per_block_pair(node_mem, n_classes)
    block_pairs_train = events_dict_to_events_dict_bp(events_dict, node_mem, n_classes)
    # initialize paramters matrices
    mu_bp = np.zeros((n_classes, n_classes))
    alpha_n_bp = np.zeros((n_classes, n_classes))
    alpha_r_bp = np.zeros((n_classes,n_classes))
    alpha_br_bp = np.zeros((n_classes, n_classes))
    alpha_gr_bp = np.zeros((n_classes, n_classes))
    C_bp = np.zeros((n_classes, n_classes, np.size(betas)))
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:  # diagonal block pair
                params_est = sum_betas_bp.fit_4_alpha_kernel_sum_dia(block_pairs_train[i][j], end_time, n_nodes_c[j, 0],
                                                                    block_pair_M_train[i, j], betas)
            else:   # off-diagonal block pair
                params_est = sum_betas_bp.fit_4_alpha_kernel_sum_off(block_pairs_train[i][j], block_pairs_train[j][i],
                                                                    end_time, n_nodes_c[j,0], block_pair_M_train[i, j], betas)
            mu_bp[i, j] = params_est[0]
            alpha_n_bp[i, j] = params_est[1]
            alpha_r_bp[i, j] = params_est[2]
            alpha_br_bp[i, j] = params_est[3]
            alpha_gr_bp[i, j] = params_est[4]
            C_bp[i,j,:] = params_est[5]
    # calclate log-likelihood on train, test, all datasets
    params_tuple = (mu_bp, alpha_n_bp, alpha_r_bp, alpha_br_bp, alpha_gr_bp, C_bp, betas)
    ll_train, num_events_train = model_LL_4_alpha_kernel_sum(params_tuple, block_pairs_train, end_time,
                                                                  block_pair_M_train, n_nodes_c, n_classes, ref)
    if ref:
        return params_tuple, ll_train, num_events_train, block_pairs_train, n_nodes_c
    else:
        return params_tuple, ll_train, num_events_train
def fit_6_alpha_kernel_sum(events_dict, node_mem, n_classes, end_time, betas, ref=False):
    # return number of node pairs within a block pair, number of nodes per class
    block_pair_M_train, n_nodes_c = num_nodes_pairs_per_block_pair(node_mem, n_classes)
    block_pairs_train = events_dict_to_events_dict_bp(events_dict, node_mem, n_classes)
    # initialize paramters matrices
    mu_bp = np.zeros((n_classes, n_classes))
    alpha_n_bp = np.zeros((n_classes, n_classes))
    alpha_r_bp = np.zeros((n_classes,n_classes))
    alpha_br_bp = np.zeros((n_classes, n_classes))
    alpha_gr_bp = np.zeros((n_classes, n_classes))
    alpha_al_bp = np.zeros((n_classes, n_classes))
    alpha_alr_bp = np.zeros((n_classes, n_classes))
    C_bp = np.zeros((n_classes, n_classes, np.size(betas)))
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:  # diagonal block pair
                params_est = sum_betas_bp.fit_6_alpha_kernel_sum_dia(block_pairs_train[i][j], end_time, n_nodes_c[j, 0],
                                                                    block_pair_M_train[i, j], betas)
            else:   # off-diagonal block pair
                params_est = sum_betas_bp.fit_6_alpha_kernel_sum_off(block_pairs_train[i][j], block_pairs_train[j][i],
                                                                    end_time, n_nodes_c[j,0], block_pair_M_train[i, j], betas)
            mu_bp[i, j] = params_est[0]
            alpha_n_bp[i, j] = params_est[1]
            alpha_r_bp[i, j] = params_est[2]
            alpha_br_bp[i, j] = params_est[3]
            alpha_gr_bp[i, j] = params_est[4]
            alpha_al_bp[i, j] = params_est[5]
            alpha_alr_bp[i, j] = params_est[6]
            C_bp[i,j,:] = params_est[7]
    # calclate log-likelihood on train, test, all datasets
    params_tuple = (mu_bp, alpha_n_bp, alpha_r_bp, alpha_br_bp, alpha_gr_bp, alpha_al_bp, alpha_alr_bp, C_bp, betas)
    ll_train, num_events_train = model_LL_6_alpha_kernel_sum(params_tuple, block_pairs_train, end_time,
                                                                  block_pair_M_train, n_nodes_c, n_classes, ref)
    if ref:
        return params_tuple, ll_train, num_events_train, block_pairs_train, n_nodes_c
    else:
        return params_tuple, ll_train, num_events_train

#%% spectral clustering
def spectral_cluster1(adj, num_classes=2, n_kmeans_init=100, normalize_z=True, multiply_s=True,
                      verbose=False, plot_eigenvalues=False, plot_save_path=''):
    """
    Runs spectral clustering on weighted or unweighted adjacency matrix

    Adapted from Makan Arastuie's implementation in CHIP model

    :param adj: weighted, unweighted or regularized adjacency matrix
    :param num_classes: number of classes for spectral clustering
    :param n_kmeans_init: number of initializations for k-means
    :param normalize_z: If True, vector z is normalized to sum to 1
    :param verbose: if True, prints the eigenvalues
    :param plot_eigenvalues: if True, plots the first `num_classes` singular values
    :param plot_save_path: directory to save the plot

    :return: predicted clustering membership
    """
    # Compute largest num_classes singular values and vectors of adjacency matrix
    u, s, v = svds(adj, k=num_classes, which='LM')
    v = v.T

    if verbose:
        print("Eigenvalues: \n", s)
    if plot_eigenvalues:
        fig, ax = plt.subplots()
        plt.scatter(np.arange(num_classes, 0, -1),
                    s,
                    s=80,
                    marker='*',
                    color='blue')
        plt.xlabel('Rank', fontsize=24)
        plt.ylabel('Singular Values', fontsize=24)
        plt.grid(True)
        ax.tick_params(labelsize=20)
        plt.tight_layout()
        # plt.savefig(join(plot_save_path, 'singular_values.pdf'))
        plt.show()

    # Sort in decreasing order of magnitude
    sorted_ind = np.argsort(-s)
    s = s[sorted_ind]
    u = u[:, sorted_ind]
    v = v[:, sorted_ind]
    # multiply both u and v by sqrt(s)
    if multiply_s:
        s_sqrt = np.diag(np.sqrt(s))
        u = u @ s_sqrt
        v = v @ s_sqrt
    z = np.c_[u, v]
    # L2 row normalization
    if normalize_z:
        z = normalize(z, norm='l2', axis=1)

    # multiply both u and v by sqrt(s)
    km = KMeans(n_clusters=num_classes, init='k-means++', n_init=n_kmeans_init, verbose=0)
    cluster_pred = km.fit_predict(z)
    # # scipy implemntation of KMeans
    # print("using scipy implementation - ++")
    # _, cluster_pred = kmeans2(z, k=num_classes, iter=n_kmeans_init, minit='++', check_finite=False)

    return cluster_pred
#%% model fitting helper function

def plot_adj(agg_adj, node_mem, K, s=""):
    nodes_per_class, n_class = [], []
    N = len(node_mem)
    for k in range(K):
        class_k_nodes = np.where(node_mem == k)[0]
        nodes_per_class.append(class_k_nodes)
        n_class.append(len(class_k_nodes))
    adj_ordered = np.zeros((N, N), dtype=int)
    i_a, i_b = 0, 0
    for a in range(K):
        for b in range(K):
            adj_ordered[i_a: i_a+n_class[a], i_b:i_b+ n_class[b]] =  agg_adj[nodes_per_class[a], :][:, nodes_per_class[b]]
            i_b += n_class[b]
        i_b = 0
        i_a += n_class[a]
    plt.figure()
    plt.pcolor(adj_ordered)
    plt.title("permuted count matrix "+ s)
    plt.show()

def print_model_param_single_beta(params_est):
    print("mu")
    print(params_est[0])
    print("\nalpha_n")
    print(params_est[1])
    print("\nalpha_r")
    print(params_est[2])
    if len(params_est)==4:
        print("\nbeta")
        print(params_est[3])
    else:
        print("\nalpha_br")
        print(params_est[3])
        print("\nalpha_gr")
        print(params_est[4])
        if len(params_est)==6:
            print("\nbeta")
            print(params_est[5])
        else:
            print("\nalpha_al")
            print(params_est[5])
            print("\nalpha_alr")
            print(params_est[6])
            print("\nbeta")
            print(params_est[7])

def print_model_param_kernel_sum(params_est):
    print("mu")
    print(params_est[0])
    print("\nalpha_n")
    print(params_est[1])
    print("\nalpha_r")
    print(params_est[2])
    classes = np.shape(params_est[0])[0]
    if len(params_est) == 5:
        print("\nC")
        for i in range(classes):
            for j in range(classes):
                print(params_est[3][i, j, :], end='\t')
            print(" ")
    else:
        print("\nalpha_br")
        print(params_est[3])
        print("\nalpha_gr")
        print(params_est[4])
        if len(params_est) == 7:
            print("\nC")
            for i in range(classes):
                for j in range(classes):
                    print(params_est[5][i, j, :], end='\t')
                print(" ")
        elif len(params_est) == 9:
            print("\nalpha_al")
            print(params_est[5])
            print("\nalpha_alr")
            print(params_est[6])
            print("\nC")
            for i in range(classes):
                for j in range(classes):
                    print(params_est[7][i, j, :], end='\t')
                print(" ")

def assign_node_membership_for_missing_nodes(node_membership, missing_nodes):
    """
    Assigns the missing nodes to the largest community
    @author: Makan Arastuie
    :param node_membership: (list) membership of every node (except missing ones) to one of K classes
    :param missing_nodes: (list) nodes to be assigned a community

    :return: node_membership including missing nodes
    """
    class_idx, class_count = np.unique(node_membership, return_counts=True)
    largest_class_idx = class_idx[np.argmax(class_count)]

    combined_node_membership = np.copy(node_membership)

    missing_nodes.sort()
    for n in missing_nodes:
        combined_node_membership = np.insert(combined_node_membership, n, largest_class_idx)

    return combined_node_membership

def events_dict_to_events_dict_bp(events_dict, node_mem, n_classes):
    # each block_pair is a dict of events
    block_pair_events_dict = [[None]*n_classes for _ in range(n_classes)]
    for i in range(n_classes):
        for j in range(n_classes):
            block_pair_events_dict[i][j] = {}
    for u, v in events_dict:
        block_pair_events_dict[node_mem[u]][node_mem[v]][(u,v)] = np.array(events_dict[u,v])
    return block_pair_events_dict

def num_nodes_pairs_per_block_pair(node_mem, n_classes):
    classes, n_node_per_class = np.unique(node_mem, return_counts=True)
    n_node_per_class = n_node_per_class.reshape((n_classes, 1))
    return (np.matmul(n_node_per_class, n_node_per_class.T)- np.diagflat(n_node_per_class), n_node_per_class)

def split_train(events_dict, split_ratio=0.8):
    events_list = list(events_dict.values())
    # find spliting point
    events_array = np.sort(np.concatenate(events_list))
    split_point = round(events_array.shape[0] * split_ratio)
    split_time = events_array[split_point]
    events_dict_t = {}
    for (u, v) in events_dict:
        p = np.array(events_dict[(u,v)])
        p_train = p[p<split_time]
        if len(p_train)!= 0 :
            events_dict_t[(u,v)] = p_train
    return events_dict_t, split_time

def get_node_id_maps(node_set):
    nodes = list(node_set)
    nodes.sort()
    node_id_map = {}
    id_node_map = {}
    for i, n in enumerate(nodes):
        node_id_map[n] = int(i)
        id_node_map[i] = n
    return node_id_map, id_node_map

def read_cvs_split_train(data_file_name, split_ratio=0.8, timestamp_max=1000):
    data = np.loadtxt(data_file_name, np.float)

    # sort data by timestamp and adjust timestamps to start from 0
    data = data[data[:, 2].argsort()]
    data[:, 2] = data[:, 2] - data[0, 2]

    # scale timestamps to 0 to timestamp_max
    if timestamp_max is not None:
        data[:, 2] = (data[:, 2] - min(data[:, 2])) / (max(data[:, 2])- min(data[:, 2])) * timestamp_max

    end_time_all = data[-1, 2]
    node_set_all = set(data[:, 0].astype(np.int)).union(data[:, 1].astype(np.int))
    n_nodes_all = len(node_set_all)
    node_id_map_all, id_node_map_all = get_node_id_maps(node_set_all)

    train_split_point = int(len(data) * split_ratio)    # point not included in train
    end_time_train = data[train_split_point - 1, 2]
    node_set_train = set(data[:train_split_point, 0].astype(np.int)).union(data[:train_split_point, 1].astype(np.int))
    n_nodes_train = len(node_set_train)
    node_id_map_train, id_node_map_train = get_node_id_maps(node_set_train)

    events_dict_all = {}
    events_dict_train = {}
    for i in range(train_split_point):
        u_all = node_id_map_all[np.int(data[i, 0])]
        v_all = node_id_map_all[np.int(data[i, 1])]
        u_train = node_id_map_train[np.int(data[i, 0])]
        v_train = node_id_map_train[np.int(data[i, 1])]
        if (u_all, v_all) not in events_dict_all:
            events_dict_all[(u_all, v_all)] = []
            events_dict_train[(u_train, v_train)] = []
        events_dict_all[(u_all, v_all)].append(data[i, 2])
        events_dict_train[(u_train, v_train)].append(data[i, 2])
    for i in range(train_split_point, len(data)):
        u_all = node_id_map_all[np.int(data[i, 0])]
        v_all = node_id_map_all[np.int(data[i, 1])]
        if (u_all, v_all) not in events_dict_all:
            events_dict_all[(u_all, v_all)] = []
        events_dict_all[(u_all, v_all)].append(data[i, 2])

    # node not in train list
    nodes_not_in_train = []
    for n in (node_set_all - node_set_train):
        nodes_not_in_train.append(node_id_map_all[n])

    train_tuple = events_dict_train, n_nodes_train, end_time_train
    all_tuple = events_dict_all, n_nodes_all, end_time_all
    return train_tuple, all_tuple, nodes_not_in_train

def event_dict_to_aggregated_adjacency(num_nodes, event_dicts, dtype=np.float):
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=dtype)

    for (u, v), event_times in event_dicts.items():
        adjacency_matrix[u, v] = len(event_times)

    return adjacency_matrix

def event_dict_to_adjacency(num_nodes, event_dicts, dtype=np.float):
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=dtype)

    for (u, v), event_times in event_dicts.items():
        if len(event_times) != 0:
            adjacency_matrix[u, v] = 1

    return adjacency_matrix


