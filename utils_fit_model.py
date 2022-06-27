"""
MULCH fit and log-likelihood functions (full model level)

The script also includes:
 - read_csv_split_train() to read dataset csv files
 - visualization functions:
    - analyze_block()
    - print_mulch_param()
    - plot_mulch_param()
    - plot_kernel()
    - plot_adj()
    -

@author: Hadeel Soliman
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import utils_fit_bp


# %% mulch log-likelihood functions

def log_likelihood_mulch(params, events_dict, node_mem, k, end_time, ref=False):
    """
    calculate full MULCH log-likelihood for all block pairs

    :param tuple params: (mu_bp, alpha_1_bp, ..., alpha_s_bp, C, betas)
        where mu_bp, alpha_i_bp are (K, K) arrays & C is (K, K, Q) array & betas is (Q,) array
    :param dict events_dict: dataset formatted as a dictionary {(u, v) node pairs in network : [t1, t2, ...] array of
        events between (u, v)}
    :param node_mem: (n,) array(int) of nodes membership
    :param int k: number of blocks
    :param float end_time: network duration (T)
    :param ref: (optional) only set to True when called from fit_refinement_mulch() function
    :return: model's log-likelihood and number of events in dataset
    :rtype: (float, int)
    """
    block_pair_M, n_nodes_c = num_nodes_pairs_per_block_pair(node_mem, k)
    events_dict_block_pair = events_dict_to_events_dict_bp(events_dict, node_mem, k)
    # check which version of model, either 2, 4, or 6 types of excitations
    if len(params) == 5:
        return model_LL_2_alpha(params, events_dict_block_pair, end_time, block_pair_M, n_nodes_c,
                                k, ref)
    elif len(params) == 7:
        return model_LL_4_alpha(params, events_dict_block_pair, end_time, block_pair_M, n_nodes_c,
                                k, ref)
    elif len(params) == 9:
        return model_LL_6_alpha(params, events_dict_block_pair, end_time, block_pair_M, n_nodes_c,
                                k, ref)


def model_LL_2_alpha(params, events_dict_bp, end_time, m_bp, n_K, K, ref=False):
    """
    calculate full MULCH log-likelihood for 2-alphas model version

    :param tuple params: (mu_bp, alpha_1_bp, ..., alpha_s_bp, C, betas)
        where mu_bp, alpha_i_bp are (K, K) arrays & C is (K, K, Q) array & betas is (Q,) array
    :param events_dict_bp: KxK list, each element events_dict[a][b] is events_dict of the block pair (a, b)
    :param float end_time: network duration
    :param m_bp: (K, K) array number of node pairs per block pair
    :param n_K: (K,) array number of nodes per block
    :param int K: number of blocks
    :param ref: (optional) only set to True when called from fit_refinement_mulch() function
    :return: model's log-likelihood and number of events in dataset
    :rtype: (float, int)
    """
    mu_bp, alpha_s_bp, alpha_r_bp, C_bp, betas = params
    LL_bp = np.zeros((K, K))
    num_events = 0
    for i in range(K):
        for j in range(K):
            par = (mu_bp[i, j], alpha_s_bp[i, j], alpha_r_bp[i, j], C_bp[i, j, :], betas)
            if i == j:  # diagonal block pair
                ll_dia = utils_fit_bp.LL_2_alpha_dia_bp(par, events_dict_bp[i][j], end_time,
                                                        n_K[j, 0], m_bp[i, j])
                LL_bp[i, j] = ll_dia
            else:  # off-diagonal block pair
                ll_off = utils_fit_bp.LL_2_alpha_off_bp(par, events_dict_bp[i][j],
                                                        events_dict_bp[j][i], end_time,
                                                        n_K[j, 0], m_bp[i, j])
                LL_bp[i, j] = ll_off
            # number of event of block_pair
            num_events += utils_fit_bp.cal_num_events(events_dict_bp[i][j])
    if ref:
        return LL_bp, num_events
    else:
        return np.sum(LL_bp), num_events


def model_LL_4_alpha(params, events_dict_bp, end_time, m_bp, n_K, K, ref=False):
    """
    calculate full MULCH log-likelihood for 4-alphas model version

    :param tuple params: (mu_bp, alpha_1_bp, ..., alpha_n_bp, C, betas)
        where mu_bp, alpha_i_bp are (K, K) arrays & C is (K, K, Q) array & betas is (Q,) array
    :param events_dict_bp: KxK list, each element events_dict[a][b] is events_dict of the block pair (a, b)
    :param float end_time: network duration
    :param m_bp: (K, K) array number of node pairs per block pair
    :param n_K: (K,) array number of nodes per block
    :param int K: number of blocks
    :param ref: (optional) only set to True when called from fit_refinement_mulch() function
    :return: model's log-likelihood and number of events in dataset
    :rtype: (float, int)
    """
    mu_bp, alpha_s_bp, alpha_r_bp, alpha_tc_bp, alpha_gr_bp, C_bp, betas = params
    LL_bp = np.zeros((K, K))
    num_events = 0
    for i in range(K):
        for j in range(K):
            par = (
            mu_bp[i, j], alpha_s_bp[i, j], alpha_r_bp[i, j], alpha_tc_bp[i, j], alpha_gr_bp[i, j],
            C_bp[i, j], betas)
            if i == j:  # diagonal block pair
                ll_dia = utils_fit_bp.LL_4_alpha_dia_bp(par, events_dict_bp[i][j], end_time,
                                                        n_K[j, 0], m_bp[i, j])
                LL_bp[i, j] = ll_dia
            else:  # off-diagonal block pair
                ll_off = utils_fit_bp.LL_4_alpha_off_bp(par, events_dict_bp[i][j],
                                                        events_dict_bp[j][i], end_time,
                                                        n_K[j, 0], m_bp[i, j])
                LL_bp[i, j] = ll_off
            # number of event of block_pair
            num_events += utils_fit_bp.cal_num_events(events_dict_bp[i][j])
    if ref:
        return LL_bp, num_events
    else:
        return np.sum(LL_bp), num_events


def model_LL_6_alpha(params, events_dict_bp, end_time, m_bp, n_K, K, ref=False):
    """
    calculate full MULCH log-likelihood for 6-alphas model version

    :param tuple params: (mu_bp, alpha_1_bp, ..., alpha_s_bp, C, betas)
        where mu_bp, alpha_i_bp are (K, K) arrays & C is (K, K, Q) array & betas is (Q,) array
    :param events_dict_bp: KxK list, each element events_dict[a][b] is events_dict of the block pair (a, b)
    :param float end_time: network duration
    :param m_bp: (K, K) array number of node pairs per block pair
    :param n_K: (K,) array number of nodes per block
    :param int K: number of blocks
    :param ref: (optional) only set to True when called from fit_refinement_mulch() function
    :return: model's log-likelihood and number of events in dataset
    :rtype: (float, int)
    """
    mu_bp, alpha_s_bp, alpha_r_bp, alpha_tc_bp, alpha_gr_bp, alpha_al_bp, alpha_alr_bp, C_bp, betas = params
    LL_bp = np.zeros((K, K))
    num_events = 0
    for i in range(K):
        for j in range(K):
            par = (
            mu_bp[i, j], alpha_s_bp[i, j], alpha_r_bp[i, j], alpha_tc_bp[i, j], alpha_gr_bp[i, j],
            alpha_al_bp[i, j], alpha_alr_bp[i, j], C_bp[i, j], betas)
            if i == j:  # diagonal block pair
                ll_dia = utils_fit_bp.LL_6_alpha_dia_bp(par, events_dict_bp[i][j], end_time,
                                                        n_K[j, 0], m_bp[i, j])
                LL_bp[i, j] = ll_dia
            else:  # off-diagonal block pair
                ll_off = utils_fit_bp.LL_6_alpha_off_bp(par, events_dict_bp[i][j],
                                                        events_dict_bp[j][i], end_time,
                                                        n_K[j, 0], m_bp[i, j])
                LL_bp[i, j] = ll_off
            # number of event of block_pair
            num_events += utils_fit_bp.cal_num_events(events_dict_bp[i][j])
    if ref:
        return LL_bp, num_events
    else:
        return np.sum(LL_bp), num_events


# %% MULCH full model fit functions
def model_fit(n_alpha, events_dict, node_mem, K, end_time, betas, ref=False):
    """
    estimate MULCH's parameters given nodes membership (no refinement)

    :param int n_alpha: number of excitation types. Choose between 2, 4, or 6
    :param dataset events_dict: dataset formatted as a dictionary {(u, v) node pairs in network : [t1, t2, ...] array of
        events between (u, v)}
    :param node_mem: nodes membership
    :param K: number of blocks
    :param end_time: network duration
    :param betas: (Q,) array of decays
    :param ref: (optional) only set to True when called from fit_refinement_mulch() function
    :return: mulch_parameters, log-likelihood, #events. if ref=True, also return events_dict_bp, n_K
    """
    # check which version of model, either 2, 4, or 6 types of excitations
    if n_alpha == 2:
        return model_fit_2_alpha(events_dict, node_mem, K, end_time, betas, ref)
    elif n_alpha == 4:
        return model_fit_4_alpha(events_dict, node_mem, K, end_time, betas, ref)
    elif n_alpha == 6:
        return model_fit_6_alpha(events_dict, node_mem, K, end_time, betas, ref)
    else:
        print(" number of alpha parameter should be 2, 4, or 6")


# different alpha-versions of sum of kernel model
def model_fit_2_alpha(events_dict, node_mem, K, end_time, betas, ref=False):
    """
    estimate MULCH's parameters given nodes membership (2-alphas model)

    :param dataset events_dict: dataset formatted as a dictionary {(u, v) node pairs in network : [t1, t2, ...] array of
        events between (u, v)}
    :param node_mem: nodes membership
    :param K: number of blocks
    :param end_time: network duration
    :param betas: (Q,) array of decays
    :param ref: (optional) only set to True when called from fit_refinement_mulch() function
    :return: mulch_parameters, log-likelihood, #events. if ref=True, also return events_dict_bp, n_K
    """
    # return number of node pairs within a block pair, number of nodes per class
    block_pair_M_train, n_nodes_c = num_nodes_pairs_per_block_pair(node_mem, K)
    block_pairs_train = events_dict_to_events_dict_bp(events_dict, node_mem, K)
    # initialize parameters matrices
    mu_bp = np.zeros((K, K))
    alpha_s_bp = np.zeros((K, K))
    alpha_r_bp = np.zeros((K, K))
    C_bp = np.zeros((K, K, np.size(betas)))
    for i in range(K):
        for j in range(K):
            if i == j:  # diagonal block pair
                params_est = utils_fit_bp.fit_2_alpha_dia_bp(block_pairs_train[i][j], end_time,
                                                             n_nodes_c[j, 0],
                                                             block_pair_M_train[i, j], betas)
            else:  # off-diagonal block pair
                params_est = utils_fit_bp.fit_2_alpha_off_bp(block_pairs_train[i][j],
                                                             block_pairs_train[j][i],
                                                             end_time, n_nodes_c[j, 0],
                                                             block_pair_M_train[i, j], betas)
            mu_bp[i, j] = params_est[0]
            alpha_s_bp[i, j] = params_est[1]
            alpha_r_bp[i, j] = params_est[2]
            C_bp[i, j, :] = params_est[3]
    # calclate log-likelihood on train, test, all datasets
    params_tuple = (mu_bp, alpha_s_bp, alpha_r_bp, C_bp, betas)
    ll_train, num_events_train = model_LL_2_alpha(params_tuple, block_pairs_train, end_time,
                                                  block_pair_M_train, n_nodes_c, K, ref)
    if ref:
        return params_tuple, ll_train, num_events_train, block_pairs_train, n_nodes_c
    else:
        return params_tuple, ll_train, num_events_train


def model_fit_4_alpha(events_dict, node_mem, K, end_time, betas, ref=False):
    """
    estimate MULCH's parameters given nodes membership (4-alphas model)

    :param dataset events_dict: dataset formatted as a dictionary {(u, v) node pairs in network : [t1, t2, ...] array of
        events between (u, v)}
    :param node_mem: nodes membership
    :param K: number of blocks
    :param end_time: network duration
    :param betas: (Q,) array of decays
    :param ref: (optional) only set to True when called from fit_refinement_mulch() function
    :return: mulch_parameters, log-likelihood, #events. if ref=True, also return events_dict_bp, n_K
    """
    # return number of node pairs within a block pair, number of nodes per class
    block_pair_M_train, n_nodes_c = num_nodes_pairs_per_block_pair(node_mem, K)
    block_pairs_train = events_dict_to_events_dict_bp(events_dict, node_mem, K)
    # initialize paramters matrices
    mu_bp = np.zeros((K, K))
    alpha_s_bp = np.zeros((K, K))
    alpha_r_bp = np.zeros((K, K))
    alpha_tc_bp = np.zeros((K, K))
    alpha_gr_bp = np.zeros((K, K))
    C_bp = np.zeros((K, K, np.size(betas)))
    for i in range(K):
        for j in range(K):
            if i == j:  # diagonal block pair
                params_est = utils_fit_bp.fit_4_alpha_dia_bp(block_pairs_train[i][j], end_time,
                                                             n_nodes_c[j, 0],
                                                             block_pair_M_train[i, j], betas)
            else:  # off-diagonal block pair
                params_est = utils_fit_bp.fit_4_alpha_off_bp(block_pairs_train[i][j],
                                                             block_pairs_train[j][i],
                                                             end_time, n_nodes_c[j, 0],
                                                             block_pair_M_train[i, j], betas)
            mu_bp[i, j] = params_est[0]
            alpha_s_bp[i, j] = params_est[1]
            alpha_r_bp[i, j] = params_est[2]
            alpha_tc_bp[i, j] = params_est[3]
            alpha_gr_bp[i, j] = params_est[4]
            C_bp[i, j, :] = params_est[5]
    # calclate log-likelihood on train, test, all datasets
    params_tuple = (mu_bp, alpha_s_bp, alpha_r_bp, alpha_tc_bp, alpha_gr_bp, C_bp, betas)
    ll_train, num_events_train = model_LL_4_alpha(params_tuple, block_pairs_train, end_time,
                                                  block_pair_M_train, n_nodes_c, K, ref)
    if ref:
        return params_tuple, ll_train, num_events_train, block_pairs_train, n_nodes_c
    else:
        return params_tuple, ll_train, num_events_train


def model_fit_6_alpha(events_dict, node_mem, n_classes, end_time, betas, ref=False):
    """
    estimate MULCH's parameters given nodes membership (6-alphas model)

    :param dataset events_dict: dataset formatted as a dictionary {(u, v) node pairs in network : [t1, t2, ...] array of
        events between (u, v)}
    :param node_mem: nodes membership
    :param K: number of blocks
    :param end_time: network duration
    :param betas: (Q,) array of decays
    :param ref: (optional) only set to True when called from fit_refinement_mulch() function
    :return: mulch_parameters, log-likelihood, #events. if ref=True, also return events_dict_bp, n_K
    """
    # return number of node pairs within a block pair, number of nodes per class
    block_pair_M_train, n_nodes_c = num_nodes_pairs_per_block_pair(node_mem, n_classes)
    block_pairs_train = events_dict_to_events_dict_bp(events_dict, node_mem, n_classes)
    # initialize paramters matrices
    mu_bp = np.zeros((n_classes, n_classes))
    alpha_s_bp = np.zeros((n_classes, n_classes))
    alpha_r_bp = np.zeros((n_classes, n_classes))
    alpha_tc_bp = np.zeros((n_classes, n_classes))
    alpha_gr_bp = np.zeros((n_classes, n_classes))
    alpha_al_bp = np.zeros((n_classes, n_classes))
    alpha_alr_bp = np.zeros((n_classes, n_classes))
    C_bp = np.zeros((n_classes, n_classes, np.size(betas)))
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:  # diagonal block pair
                params_est = utils_fit_bp.fit_6_alpha_dia_bp(block_pairs_train[i][j], end_time,
                                                             n_nodes_c[j, 0],
                                                             block_pair_M_train[i, j], betas)
            else:  # off-diagonal block pair
                params_est = utils_fit_bp.fit_6_alpha_off_bp(block_pairs_train[i][j],
                                                             block_pairs_train[j][i],
                                                             end_time, n_nodes_c[j, 0],
                                                             block_pair_M_train[i, j], betas)
            mu_bp[i, j] = params_est[0]
            alpha_s_bp[i, j] = params_est[1]
            alpha_r_bp[i, j] = params_est[2]
            alpha_tc_bp[i, j] = params_est[3]
            alpha_gr_bp[i, j] = params_est[4]
            alpha_al_bp[i, j] = params_est[5]
            alpha_alr_bp[i, j] = params_est[6]
            C_bp[i, j, :] = params_est[7]
    # calclate log-likelihood on train, test, all datasets
    params_tuple = (
    mu_bp, alpha_s_bp, alpha_r_bp, alpha_tc_bp, alpha_gr_bp, alpha_al_bp, alpha_alr_bp, C_bp, betas)
    ll_train, num_events_train = model_LL_6_alpha(params_tuple, block_pairs_train, end_time,
                                                  block_pair_M_train, n_nodes_c, n_classes, ref)
    if ref:
        return params_tuple, ll_train, num_events_train, block_pairs_train, n_nodes_c
    else:
        return params_tuple, ll_train, num_events_train


# %% spectral clustering
def spectral_cluster1(adj, num_classes=2, n_kmeans_init=100, normalize_z=True, multiply_s=True,
                      verbose=False,
                      plot_eigenvalues=False):
    """
    Runs spectral clustering on weighted or unweighted adjacency matrix

    Adapted from Makan Arastuie's implementation in CHIP model

    :param adj: weighted, unweighted or regularized adjacency matrix
    :param num_classes: number of classes for spectral clustering
    :param n_kmeans_init: number of initializations for k-means
    :param normalize_z: If True, vector z is normalized to sum to 1
    :param multiply_s: if true, multiply both u and v by sqrt(s)
    :param verbose: if True, prints the eigenvalues
    :param plot_eigenvalues: if True, plots the first `num_classes` singular values
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


# %% model fitting helper function


def assign_node_membership_for_missing_nodes(node_membership, missing_nodes):
    """
    Assigns new nodes, that appeared in test but not in train dataset, to the largest community

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


def events_dict_to_events_dict_bp(events_dict, node_mem, K):
    """
    split events_dict into KxK part. Each part has only events within a block pair

    the function computes events_dict_bp a (kxK) list, each element events_dict_bp[a][b] is events_dict of the
    block_pair (a, b) i.e. a key (u, v) in events_dict_bp[a][b] means u in block(a) and v in block(b)

    :param events_dict: dataset formatted as a dictionary {(u, v) node pairs in network : [t1, t2, ...] array of
        events between (u, v)}
    :param node_mem: (n,) array(int) nodes membership
    :param int K: number of blocks
    :return: events_dict_bp is (kxK) list, each element events_dict_bp[a][b] is events_dict of the block pair (a, b)
    """
    block_pair_events_dict = [[None] * K for _ in range(K)]
    for i in range(K):
        for j in range(K):
            block_pair_events_dict[i][j] = {}
    for u, v in events_dict:
        block_pair_events_dict[node_mem[u]][node_mem[v]][(u, v)] = np.array(events_dict[u, v])
    return block_pair_events_dict


def num_nodes_pairs_per_block_pair(node_mem, K):
    """ return (K, K) array of #node_pairs in each block_pair (a, b), (K,) array #nodes per block"""
    classes, n_node_per_class = np.unique(node_mem, return_counts=True)
    n_node_per_class = n_node_per_class.reshape((K, 1))
    return (np.matmul(n_node_per_class, n_node_per_class.T) - np.diagflat(n_node_per_class),
            n_node_per_class)


def get_node_id_maps(node_set):
    """
    map each node to a unique id between [0 : #nodes-1]

    :param node_set: set of unique nodes in the network (could be string or number
    :return: dict(node:id) map, dict(id:node) map
    """
    nodes = list(node_set)
    nodes.sort()
    node_id_map = {}
    id_node_map = {}
    for i, n in enumerate(nodes):
        node_id_map[n] = int(i)
        id_node_map[i] = n
    return node_id_map, id_node_map


def read_csv_split_train(data_file_name, delimiter, remove_not_train=False, split_ratio=0.8,
                         timestamp_max=1000):
    """
    read network csv (or txt) file and return train and full dataset

    file is assumed to have 3 columns [sender node, receiver node, timestamp] with no header

    :param str data_file_name: csv file path
    :param str delimiter: delimiter to use for reading file
    :param float split_ratio: (optional) train:test ratio, choose between [0, 1]. Default=0.8
    :param remove_not_train: (optional) if True, remove nodes (and corresponding events) that
        appeared in test set but not in train.
    :param timestamp_max: (optional) scale network's timestamps between [0, timestamp_max].
        Default=1000 (used in all MULCH dataset experiments)
    :return: train_tuple, full_tuple, nodes_not_in_train.
        Both train and full dataset tuple=(events_dict, number_nodes, duration, end_time, number_events),
        where events_dict is the dataset format passed to MULCH fit function (see fit_refinement_mulch()).
        nodes_not_in_train is list of nodes appeared in test dataset, but not in train
    :rtype: (tuple, tuple, list)
    """
    data_df = pd.read_csv(data_file_name, sep=delimiter, header=None, usecols=[0, 1, 2])

    # sort data by timestamp and adjust timestamps to start from 0
    data_df.sort_values(by=2)
    data_df.iloc[:, 2] = data_df.iloc[:, 2] - data_df.iloc[0, 2]

    # scale timestamps to 0 to timestamp_max
    if timestamp_max is not None:
        data_df.iloc[:, 2] = data_df.iloc[:, 2] * timestamp_max / (
                    data_df.iloc[-1, 2] - data_df.iloc[0, 2])
    # full network duration
    end_time_all = data_df.iloc[-1, 2]

    # computing train split
    train_split_point = int(len(data_df) * split_ratio)  # point not included in train
    end_time_train = data_df.iloc[train_split_point - 1, 2]
    # unique set of nodes in train dataset
    node_set_train = set(data_df.iloc[:train_split_point, 0].unique()).union(
        data_df.iloc[:train_split_point, 1].unique())
    n_nodes_train = len(node_set_train)
    node_id_map_train, id_node_map_train = get_node_id_maps(node_set_train)

    if remove_not_train:  # remove nodes in full dataset but not in train split
        # no nodes in full dataset not in train
        nodes_not_in_train = []

        events_dict_all = {}
        events_dict_train = {}
        for i in range(train_split_point):
            if data_df.iloc[i, 0] in node_id_map_train and data_df.iloc[i, 1] in node_id_map_train:
                u = node_id_map_train[data_df.iloc[i, 0]]
                v = node_id_map_train[data_df.iloc[i, 1]]
                if (u, v) not in events_dict_all:
                    events_dict_all[(u, v)] = []
                    events_dict_train[(u, v)] = []
                events_dict_all[(u, v)].append(data_df.iloc[i, 2])
                events_dict_train[(u, v)].append(data_df.iloc[i, 2])
        for i in range(train_split_point, len(data_df)):
            if data_df.iloc[i, 0] in node_id_map_train and data_df.iloc[i, 1] in node_id_map_train:
                u = node_id_map_train[data_df.iloc[i, 0]]
                v = node_id_map_train[data_df.iloc[i, 1]]
                if (u, v) not in events_dict_all:
                    events_dict_all[(u, v)] = []
                events_dict_all[(u, v)].append(data_df.iloc[i, 2])

        n_events_train = utils_fit_bp.cal_num_events(events_dict_train)
        n_events_all = utils_fit_bp.cal_num_events(events_dict_all)
        # both n_nodes and node_id of train and full are equal
        n_nodes_all = n_nodes_train
        id_node_map_all = id_node_map_train

    else:  # keep all nodes
        # some node in full dataset might not have appeared in train
        node_set_all = set(data_df[0].unique()).union(data_df[1].unique())
        # unique set of all nodes in network
        n_nodes_all = len(node_set_all)
        # map each node to a unique id between [0 : #nodes -1]
        node_id_map_all, id_node_map_all = get_node_id_maps(node_set_all)

        # list to save ids of nodes in full dataset, but didn't appear in train
        nodes_not_in_train = []
        for n in (node_set_all - node_set_train):
            nodes_not_in_train.append(node_id_map_all[n])

        events_dict_all = {}
        events_dict_train = {}
        for i in range(train_split_point):
            u_all = node_id_map_all[data_df.iloc[i, 0]]
            v_all = node_id_map_all[data_df.iloc[i, 1]]
            u_train = node_id_map_train[data_df.iloc[i, 0]]
            v_train = node_id_map_train[data_df.iloc[i, 1]]
            if (u_all, v_all) not in events_dict_all:
                events_dict_all[(u_all, v_all)] = []
                events_dict_train[(u_train, v_train)] = []
            events_dict_all[(u_all, v_all)].append(data_df.iloc[i, 2])
            events_dict_train[(u_train, v_train)].append(data_df.iloc[i, 2])
        for i in range(train_split_point, len(data_df)):
            u_all = node_id_map_all[data_df.iloc[i, 0]]
            v_all = node_id_map_all[data_df.iloc[i, 1]]
            if (u_all, v_all) not in events_dict_all:
                events_dict_all[(u_all, v_all)] = []
            events_dict_all[(u_all, v_all)].append(data_df.iloc[i, 2])

        n_events_train = len(data_df.iloc[:train_split_point, 0])
        n_events_all = len(data_df)

    train_tuple = events_dict_train, n_nodes_train, end_time_train, n_events_train, id_node_map_train
    all_tuple = events_dict_all, n_nodes_all, end_time_all, n_events_all, id_node_map_all
    return train_tuple, all_tuple, nodes_not_in_train


def event_dict_to_aggregated_adjacency(num_nodes, event_dicts, dtype=np.float):
    """ return (n,n) network aggregated adjacency matrix """
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=dtype)

    for (u, v), event_times in event_dicts.items():
        adjacency_matrix[u, v] = len(event_times)

    return adjacency_matrix


def event_dict_to_adjacency(num_nodes, event_dicts, dtype=np.float):
    """ return (n,n) network adjacency matrix """
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=dtype)

    for (u, v), event_times in event_dicts.items():
        if len(event_times) != 0:
            adjacency_matrix[u, v] = 1

    return adjacency_matrix


# %% visualization functions
def plot_adj(agg_adj, node_mem, K, s=""):
    """plot adjacency matrix permuted by nodes membership"""
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
            adj_ordered[i_a: i_a + n_class[a], i_b:i_b + n_class[b]] = agg_adj[nodes_per_class[a],
                                                                       :][:, nodes_per_class[b]]
            i_b += n_class[b]
        i_b = 0
        i_a += n_class[a]
    plt.figure()
    plt.pcolor(adj_ordered)
    plt.title("permuted count matrix " + s)
    plt.show()


def plot_mulch_param(params, n_alpha):
    """plot MULCH model parameters

    :param params: (mu_bp, alpha_1_bp, ..., alpha_n_bp, C, betas)
        where mu_bp, alpha_i_bp are (K, K) arrays & C is (K, K, Q) array & betas is (Q,) array
    :return: None
    """
    param_name = ["mu", "alpha_self", "alpha_recip", "alpha_turn_cont", "alpha_generalized_recip",
                  "alpha_allied_cont", "alpha_allied_recip"]
    for param, i in zip(params, range(n_alpha + 1)):
        fig, ax = plt.subplots(figsize=(5, 4))
        plot = ax.pcolor(param, cmap='gist_yarg')
        ax.set_xticks(np.arange(0.5, 4))
        ax.set_xticklabels(np.arange(1, 5))
        ax.set_yticks(np.arange(0.5, 4))
        ax.set_yticklabels(np.arange(1, 5))
        ax.invert_yaxis()
        fig.colorbar(plot, ax=ax)
        ax.set_title(param_name[i])  # comment to save as pdf
        plt.show()


def plot_kernel(alpha, betas, C, time_range):
    """plot a block pair decay kernel"""
    lambda_sum = []
    for t in time_range:
        lambda_sum.append(alpha * np.sum(betas * C * np.exp(-t * betas)))
    plt.figure()
    plt.plot(time_range, lambda_sum, color='red', label=f"betas1={betas}")
    plt.xlabel("t(s)")
    plt.ylabel("lambda(t)")
    plt.yscale('log')
    plt.title('sum of kernels C=[0.33, 0.33, 0.34] - y-log scale ')
    plt.legend()
    plt.grid(True)
    plt.show()


def print_mulch_param(params):
    """print MULCH model estimated parameters

    :param params: (mu_bp, alpha_1_bp, ..., alpha_n_bp, C, betas)
        where mu_bp, alpha_i_bp are (K, K) arrays & C is (K, K, Q) array & betas is (Q,) array
    :return: None
    """
    print("mu")
    print(params[0])
    print("\nalpha_s")
    print(params[1])
    print("\nalpha_r")
    print(params[2])
    classes = np.shape(params[0])[0]
    if len(params) > 5:
        print("\nalpha_tc")
        print(params[3])
        print("\nalpha_gr")
        print(params[4])
        if len(params) > 7:
            print("\nalpha_al")
            print(params[5])
            print("\nalpha_alr")
            print(params[6])
    print("\nC")
    for i in range(classes):
        for j in range(classes):
            print(params[-2][i, j, :], end='\t')
        print(" ")
    print("\nbetas")
    print(params[-1])


def analyze_block(node_mem, K, id_node_map):
    """print nodes in each block, given id_node_map

    :param node_mem: (n,) nodes membership
    :param K: number of blocks
    :param id_node_map: dictionary {node_id : node_name}
    :return: None
    """
    # print(np.histogram(node_mem, bins=K))
    for i in range(K):
        print(f"Block {i}")
        nodes_in_class_i = np.where(node_mem == i)[0]
        for id in nodes_in_class_i:
            print(id_node_map[id], end=' ')
        print()


def plot_motif_matrix(motif, vmax, mape=0, title=''):
    """plot motifs count (6, 6) matrix

    :param motif: (6,6) array of motif counts
    :param vmax: maximum value for color bar
    :param mape: (optional) MAPE score
    :param title: (optional) figure's title
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    c = ax.pcolor(motif, cmap='Blues', vmin=0, vmax=vmax)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(6) + 0.5)
    ax.set_yticks(np.arange(6) + 0.5)
    ax.set_xticklabels(np.arange(1, 7))
    ax.set_yticklabels(np.arange(1, 7))
    fig.colorbar(c, ax=ax)
    if title != '':
        ax.set_title(f'{title}, MAPE={mape:.1f}')
    plt.show()
