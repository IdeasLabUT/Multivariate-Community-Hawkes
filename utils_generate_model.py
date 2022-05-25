""" helper functions for generating networks from MULCH model

Reference: MULCH simulation class MHP_Kernels.py is adapted from
hawkes GitHub repository by Steven Morse https://github.com/stmorse/hawkes


@author: Hadeel Soliman"""
import random
import numpy as np
from hawkes.MHP_Kernels import MHP_Kernels_2


#%% simulation functions

def simulate_mulch(sim_param, n_nodes, n_classes, p, duration):
    """
    simulate networks from MULCH

    :param tuple sim_param: MULCH parameters (mu_bp, alphas_1_bp, .., alpha_n_bp, C_bp, betas )
    :param n_nodes: number nodes in network (n)
    :param n_classes: number of blocks
    :param p: (K,) array of class membership probabilities (should sum to one) - ex: np.array([0.1, 0.4, 0.5])
    :param duration: network duration (T)
    :return: tuple of (events_dict, nodes_membership). (events_dict): dataset formatted as a dictionary
        {(u, v) node pairs in network : [t1, t2, ...] array of events between (u, v)}.
        (nodes_membership): (n,) array of block membership of each node.
    """
    if len(sim_param) == 5:
        mu_sim, alpha_s_sim, alpha_r_sim, C_sim, betas_sim = sim_param
    elif len(sim_param) == 7:
        mu_sim, alpha_s_sim, alpha_r_sim, alpha_tc_sim, alpha_gr_sim, C_sim , betas_sim = sim_param
    elif len(sim_param) == 9:
        mu_sim, alpha_s_sim, alpha_r_sim, alpha_tc_sim, alpha_gr_sim, alpha_al_sim, alpha_alr_sim, C_sim , betas_sim = sim_param
    # list (n_classes) elements, each element is array of nodes that belong to same class
    nodes_list = list(range(n_nodes))
    random.shuffle(nodes_list)
    p = np.round(np.cumsum(p) * n_nodes).astype(int)
    class_nodes_list = np.array_split(nodes_list, p[:-1])
    node_mem_actual = np.zeros((n_nodes,), dtype=int)
    for c in range(n_classes):
        node_mem_actual[class_nodes_list[c]] = c
    events_dict_all = {}
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:
                # blocks with only one node have 0 processes
                if len(class_nodes_list[i]) > 1:
                    if len(sim_param) == 5:
                        par = (mu_sim[i, j], alpha_s_sim[i, j], alpha_r_sim[i, j], C_sim[i, j], betas_sim)
                    elif len(sim_param) == 7:
                        par = (mu_sim[i, j], alpha_s_sim[i, j], alpha_r_sim[i, j], alpha_tc_sim[i, j], alpha_gr_sim[i, j],
                           np.array(C_sim[i, j]), betas_sim)
                    elif len(sim_param) == 9:
                        par = (mu_sim[i, j], alpha_s_sim[i, j], alpha_r_sim[i, j], alpha_tc_sim[i, j], alpha_gr_sim[i, j],
                           alpha_al_sim[i, j], alpha_alr_sim[i, j], np.array(C_sim[i, j]), betas_sim)
                    events_dict = simulate_dia_bp(par, list(class_nodes_list[i]), duration)
                    events_dict_all.update(events_dict)
            elif i < j:
                if len(sim_param) == 5:
                    par_ab = (mu_sim[i, j], alpha_s_sim[i, j], alpha_r_sim[i, j], C_sim[i, j], betas_sim)
                    par_ba = (mu_sim[j, i], alpha_s_sim[j, i], alpha_r_sim[j, i], C_sim[j, i], betas_sim)
                elif len(sim_param) == 7:
                    par_ab = (mu_sim[i, j], alpha_s_sim[i, j], alpha_r_sim[i, j], alpha_tc_sim[i, j], alpha_gr_sim[i, j],
                              np.array(C_sim[i, j]), betas_sim)
                    par_ba = (mu_sim[j, i], alpha_s_sim[j, i], alpha_r_sim[j, i], alpha_tc_sim[j, i], alpha_gr_sim[j, i],
                              np.array(C_sim[j, i]) ,betas_sim)
                elif len(sim_param) == 9:
                    par_ab = (mu_sim[i, j], alpha_s_sim[i, j], alpha_r_sim[i, j], alpha_tc_sim[i, j], alpha_gr_sim[i, j],
                              alpha_al_sim[i, j], alpha_alr_sim[i, j], np.array(C_sim[i, j]), betas_sim)
                    par_ba = (mu_sim[j, i], alpha_s_sim[j, i], alpha_r_sim[j, i], alpha_tc_sim[j, i], alpha_gr_sim[j, i],
                              alpha_al_sim[j, i], alpha_alr_sim[j, i], np.array(C_sim[j, i]) ,betas_sim)
                d_ab, d_ba = simulate_off_bp(par_ab, par_ba, list(class_nodes_list[i]), list(class_nodes_list[j]),
                                             duration)
                events_dict_all.update(d_ab)
                events_dict_all.update(d_ba)
    return events_dict_all, node_mem_actual


def simulate_dia_bp(par, a_nodes, duration, return_list=False):
    """
    simulate one MULCH diagonal block pair (a, b) (i.e. a=b)

    :param tuple par: block pair parameters (mu, alphas_1, .., alpha_s, C, betas)
    :param a_nodes: array of ids of nodes in block (a).
    :param duration: network's duration (T)
    :param return_list: only used for additional functionality checking
    :return: events_dict: events in block pair (a, b) formatted as a dictionary
        {(u, v) node pairs in (a, b) : [t1, t2, ...] array of events between (u, v)}.
    """

    # get (m, m) excitation matrix, m is # of node pair per block pair
    n_alpha = len(par) - 3  # number of types of excitations
    # pass mu, alphas parameters
    mu_array, alpha_matrix = get_mu_array_alpha_matrix_dia_bp(par[0], par[1: n_alpha + 1], len(a_nodes))
    P = MHP_Kernels_2(mu=mu_array, alpha=alpha_matrix, C=par[-2], betas=par[-1])
    P.generate_seq(duration)
    # assume that timestamps list is ordered ascending with respect to u then v [(0,1), (0,2), .., (1,0), (1,2), ...]
    events_list = []
    for m in range(len(mu_array)):
        i = np.nonzero(P.data[:, 1] == m)[0]
        events_list.append(P.data[i, 0])
    events_dict = events_list_to_events_dict_remove_empty_np(events_list, a_nodes)
    if return_list:
        return events_list, events_dict
    return events_dict


def simulate_off_bp(par_ab, par_ba, a_nodes, b_nodes, duration, return_list=False):
    """
    simulate two MULCH off-diagonal block pairs (a, b) & (b, a) , where a != b

    :param tuple par_ab: block pair (a, b) parameters (mu, alphas_1, .., alpha_s, C, betas)
    :param tuple par_ba: block pair (b, a) parameters (mu, alphas_1, .., alpha_s, C, betas)
    :param a_nodes: array of ids of nodes in block (a).
    :param b_nodes: array of ids of nodes in block (b).
    :param duration: network's duration (T)
    :param return_list: only used for additional functionality checking
    :return: events_dict_ab, events_dict_ba - Two events_dict for events in block pair (a, b) and (b, a) respectively.
        events_dict_ab is a dictionary {(u, v) node pairs in (a, b) : [t1, t2, ...] array of events between (u, v)}.
        events_dict_ba is for (b, a).
    """

    # get (2m, 2m) excitation matrix, m = # of node pair per block pair (a, b) = (b, a)
    n_alpha = len(par_ba) - 3   # number of types of excitations
    mu_array, alpha_matrix = get_mu_array_alpha_matrix_off_bp(par_ab[0], par_ab[1: n_alpha+1], par_ba[0]
                                                              , par_ba[1: n_alpha+1], len(a_nodes), len(b_nodes))
    P = MHP_Kernels_2(mu=mu_array, alpha=alpha_matrix, C=par_ab[-2], C_r=par_ba[-2], betas=par_ab[-1])
    P.generate_seq(duration)

    # assume that timestamps list is ordered ascending with respect to u then v [(0,1), (0,2), .., (1,0), (1,2), ...]
    events_list = []
    for m in range(len(mu_array)):
        i = np.nonzero(P.data[:, 1] == m)[0]
        events_list.append(P.data[i, 0])
    M = len(a_nodes) * len(b_nodes)
    events_list_ab = events_list[:M]
    events_list_ba = events_list[M:]
    events_dict_ab = events_list_to_events_dict_remove_empty_np_off(events_list_ab, a_nodes, b_nodes)
    events_dict_ba = events_list_to_events_dict_remove_empty_np_off(events_list_ba, b_nodes, a_nodes)
    if return_list:
        return events_list, events_dict_ab, events_dict_ba
    return events_dict_ab, events_dict_ba
#%% Excitation matrix and baseline array

def get_6_alphas_matrix_dia_bp(alphas, n_a):
    """Get (m, m) excitation matrix for on diagonal block pair, m = number of node pair in block pair

    :param alphas: (6,) array of values of excitations of block pair (a, a)
    :param n_a: number of nodes in block (a)
    :return: (m, m) excitation matrix
    """

    alpha_s, alpha_r, alpha_tc, alpha_gr, alpha_al, alpha_alr = alphas
    # add alpha_s, alpha_tc to alpha_matrix
    block = (np.ones((n_a - 1, n_a - 1)) - np.identity(n_a - 1)) * alpha_tc + np.identity(n_a - 1) * alpha_s
    alpha_matrix = np.kron(np.eye(n_a), block)
    np_list = get_np_dia_list(n_a)
    # loop through node pairs in block pair (a1, b1)
    for from_idx, (i, j) in enumerate(np_list):
        # loop through all node pairs
        for to_idx, (x, y) in enumerate(np_list):
            # alpha_r
            if (i, j) == (y, x):
                alpha_matrix[from_idx, to_idx] = alpha_r
            # alpha_gr
            elif i==y and j!=x:
                alpha_matrix[from_idx, to_idx] = alpha_gr
            # alpha_al
            elif j==y and i!=x:
                alpha_matrix[from_idx, to_idx] = alpha_al
            # alpha_alr
            elif j==x and i!=y:
                alpha_matrix[from_idx, to_idx] = alpha_alr
    return alpha_matrix


def get_6_alphas_matrix_off_bp(alphas_ab, alphas_ba, n_a, n_b):
    """Get (2m, 2m) excitation matrix for two off-diagonal block pair, m = number of node pair in block pair (a,b)

    :param alphas_ab: (6,) array of values of excitations of block pair (a, b)
    :param alphas_ba: (6,) array of values of excitations of block pair (b, a)
    :param n_a: number of nodes in block (a)
    :param n_b: number of nodes in block (b)
    :return: (2m, 2m) excitation matrix
    """
    alpha_s_ab, alpha_r_ab, alpha_tc_ab, alpha_gr_ab, alpha_al_ab, alpha_alr_ab = alphas_ab
    alpha_s_ba, alpha_r_ba, alpha_tc_ba, alpha_gr_ba, alpha_al_ba, alpha_alr_ba = alphas_ba
    M_ab = n_a * n_b
    # alpha_matrix (2M_ab , 2M_ab)
    alpha_matrix = np.zeros((2*M_ab , 2*M_ab))
    np_list = get_np_off_list(n_a, n_b)
    # loop through node pairs in block pair (a, b)
    for from_idx, (i, j) in enumerate(np_list[:M_ab]):
        # loop through all node pairs
        for to_idx, (x, y) in enumerate(np_list):
            # alpha_s
            if (i, j) == (x, y):
                alpha_matrix[from_idx, to_idx] = alpha_s_ab
            # alpha_r
            elif (i, j) == (y, x):
                alpha_matrix[from_idx, to_idx] = alpha_r_ab
            # alpha_tc
            elif i==x:
                alpha_matrix[from_idx, to_idx] = alpha_tc_ab
            # alpha_gr
            elif i==y:
                alpha_matrix[from_idx, to_idx] = alpha_gr_ab
            # alpha_al
            elif j==y:
                alpha_matrix[from_idx, to_idx] = alpha_al_ab
            # alpha_alr
            elif j==x:
                alpha_matrix[from_idx, to_idx] = alpha_alr_ab
    # loop through node pairs in block pair (b, a)
    for from_idx, (i, j) in enumerate(np_list[M_ab:], M_ab):
        # loop through all node pairs
        for to_idx, (x, y) in enumerate(np_list):
            # alpha_s
            if (i, j) == (x, y):
                alpha_matrix[from_idx, to_idx] = alpha_s_ba
            # alpha_r
            elif (i, j) == (y, x):
                alpha_matrix[from_idx, to_idx] = alpha_r_ba
            # alpha_tc
            elif i==x:
                alpha_matrix[from_idx, to_idx] = alpha_tc_ba
            # alpha_gr
            elif i==y:
                alpha_matrix[from_idx, to_idx] = alpha_gr_ba
            # alpha_al
            elif j==y:
                alpha_matrix[from_idx, to_idx] = alpha_al_ba
            # alpha_alr
            elif j==x:
                alpha_matrix[from_idx, to_idx] = alpha_alr_ba
    return alpha_matrix


def get_4_alphas_matrix_dia_bp(alphas, n_nodes):
    """Get (m, m) excitation matrix for on diagonal block pair, m = number of node pair in block pair

    :param alphas: (4,) array of values of excitations of block pair (a, a)
    :param n_a: number of nodes in block (a)
    :return: (m, m) excitation matrix
    """

    alpha_s, alpha_r, alpha_tc, alpha_gr = alphas
    nodes_set = set(np.arange(n_nodes))  # set of nodes
    block = (np.ones((n_nodes - 1, n_nodes - 1)) - np.identity(n_nodes - 1)) * alpha_tc + np.identity(n_nodes - 1) * alpha_s
    alpha_matrix = np.kron(np.eye(n_nodes), block)
    # add alpha_r , alpha_gr parameters assuming node are ordered
    for u in range(n_nodes):
        for v in range(n_nodes):
            if u != v:
                alpha_matrix[node_pair_index((u, v), n_nodes), node_pair_index((v, u), n_nodes)] = alpha_r
                nodes_minus = nodes_set - {v} - {u}
                for i in nodes_minus:
                    alpha_matrix[node_pair_index((u, v), n_nodes), node_pair_index((i, u), n_nodes)] = alpha_gr
    return alpha_matrix
def get_4_alphas_matrix_off_bp(alphas_ab, alphas_ba, n_nodes_a, n_nodes_b):
    """Get (2m, 2m) excitation matrix for two off-diagonal block pair, m = number of node pair in block pair (a,b)

    :param alphas_ab: (4,) array of values of excitations of block pair (a, b)
    :param alphas_ba: (4,) array of values of excitations of block pair (b, a)
    :param n_a: number of nodes in block (a)
    :param n_b: number of nodes in block (b)
    :return: (2m, 2m) excitation matrix
    """
    M = n_nodes_a * n_nodes_b  # number of nodes pair per block pair
    alpha_s_ab, alpha_r_ab, alpha_tc_ab, alpha_gr_ab = alphas_ab
    alpha_s_ba, alpha_r_ba, alpha_tc_ba, alpha_gr_ba = alphas_ba
    ### alpha matrix (2M,2M)
    # alpha ab-ab
    block = (np.ones((n_nodes_b, n_nodes_b)) - np.identity(n_nodes_b)) * alpha_tc_ab + np.identity(
        n_nodes_b) * alpha_s_ab
    alpha_matrix_ab_ab = np.kron(np.eye(n_nodes_a), block)
    # alpha ab-ba
    alpha_matrix_ab_ba = np.zeros((M, M))
    for a in range(n_nodes_b):
        col = [alpha_gr_ab] * n_nodes_b
        col[a] = alpha_r_ab
        for b in range(n_nodes_a):
            alpha_matrix_ab_ba[b * n_nodes_b:(b + 1) * n_nodes_b, b + a * n_nodes_a] = col
    # alpha ba-ab
    alpha_matrix_ba_ab = np.zeros((M, M))
    for b in range(n_nodes_a):
        col = [alpha_gr_ba] * n_nodes_a
        col[b] = alpha_r_ba
        for a in range(n_nodes_b):
            alpha_matrix_ba_ab[a * n_nodes_a:(a + 1) * n_nodes_a, a + b * n_nodes_b] = col
    # alpha ba-ba
    block = (np.ones((n_nodes_a, n_nodes_a)) - np.identity(n_nodes_a)) * alpha_tc_ba + np.identity(
        n_nodes_a) * alpha_s_ba
    alpha_matrix_ba_ba = np.kron(np.eye(n_nodes_b), block)
    alpha_matrix = np.vstack(
        (np.hstack((alpha_matrix_ab_ab, alpha_matrix_ab_ba)), np.hstack((alpha_matrix_ba_ab, alpha_matrix_ba_ba))))
    return alpha_matrix


def get_2_alphas_matrix_dia_bp(alphas, n_nodes):
    """Get (m, m) excitation matrix for on diagonal block pair, m = number of node pair in block pair

    :param alphas: (2,) array of values of excitations of block pair (a, a)
    :param n_a: number of nodes in block (a)
    :return: (m, m) excitation matrix
    """

    alpha_s, alpha_r = alphas
    block = np.identity(n_nodes - 1) * alpha_s
    alpha_matrix = np.kron(np.eye(n_nodes), block)
    # add alpha_r assuming node are ordered
    for u in range(n_nodes):
        for v in range(n_nodes):
            if u != v:
                alpha_matrix[node_pair_index((u, v), n_nodes), node_pair_index((v, u), n_nodes)] = alpha_r
    return alpha_matrix
def get_2_alphas_matrix_off_bp(alphas_ab, alphas_ba, n_nodes_a, n_nodes_b):
    """Get (2m, 2m) excitation matrix for two off-diagonal block pair, m = number of node pair in block pair (a,b)

    :param alphas_ab: (2,) array of values of excitations of block pair (a, b)
    :param alphas_ba: (2,) array of values of excitations of block pair (b, a)
    :param n_a: number of nodes in block (a)
    :param n_b: number of nodes in block (b)
    :return: (2m, 2m) excitation matrix
    """
    M = n_nodes_a * n_nodes_b  # number of processes
    alpha_s_ab, alpha_r_ab = alphas_ab
    alpha_s_ba, alpha_r_ba = alphas_ba

    ### alpha matrix (2M,2M)
    # alpha ab-ab
    alpha_matrix_ab_ab = np.identity(M) * alpha_s_ab
    # alpha ab-ba
    alpha_matrix_ab_ba = np.zeros((M, M))
    for a in range(n_nodes_b):
        col = [0] * n_nodes_b
        col[a] = alpha_r_ab
        for b in range(n_nodes_a):
            alpha_matrix_ab_ba[b * n_nodes_b:(b + 1) * n_nodes_b, b + a * n_nodes_a] = col
    # alpha ba-ab
    alpha_matrix_ba_ab = np.zeros((M, M))
    for b in range(n_nodes_a):
        col = [0] * n_nodes_a
        col[b] = alpha_r_ba
        for a in range(n_nodes_b):
            alpha_matrix_ba_ab[a * n_nodes_a:(a + 1) * n_nodes_a, a + b * n_nodes_b] = col
    # alpha ba-ba
    alpha_matrix_ba_ba = np.identity(M) * alpha_s_ba
    # combine four alpha_matrix
    alpha_matrix = np.vstack(
        (np.hstack((alpha_matrix_ab_ab, alpha_matrix_ab_ba)), np.hstack((alpha_matrix_ba_ab, alpha_matrix_ba_ba))))
    return alpha_matrix

def get_mu_array_alpha_matrix_dia_bp(mu, alphas, n_a):
    """Get baseline array and excitation matrix for on diagonal block pair.

    m = number of node pair in block pair
    baseline array (mu_array) = (m,) array of diagonal block pair mu parameter
    excitation matrix = (m, m) array

    :param mu: diagonal block pair mu parameter
    :param alphas: array of values of excitations of block pair (a, a)
    :param n_a: number of nodes in block (a)
    :return: (m,) baseline array, (m, m) excitation matrix
    """

    n_alphas = len(alphas)
    M = n_a * (n_a - 1)  # number of node pairs in block pair
    # excitation matrix
    if n_alphas == 6:
        alpha_matrix = get_6_alphas_matrix_dia_bp(alphas, n_a)
    elif n_alphas == 4:
        alpha_matrix = get_4_alphas_matrix_dia_bp(alphas, n_a)
    else:
        alpha_matrix = get_2_alphas_matrix_dia_bp(alphas, n_a)
    # mu array (M, 1)
    mu_array = np.ones(M) * mu
    return mu_array, alpha_matrix


def get_mu_array_alpha_matrix_off_bp(mu_ab, alphas_ab, mu_ba, alphas_ba, n_a, n_b):
    """Get baseline array and excitation matrix for two off-diagonal block pairs.

    m = number of node pair in block pair (a, b) = (b, a)
    baseline array (mu_array) = (2m,) array of both off-diagonal block pairs mu's parameter
    excitation matrix = (2m, 2m) array

    :param mu_ab: (a, b) block pair mu parameter
    :param mu_ba: (b, a) block pair mu parameter
    :param alphas_ab: array of values of excitations of block pair (a, b)
    :param alphas_ab: array of values of excitations of block pair (b, a)
    :param n_a: number of nodes in block (a)
    :param n_a: number of nodes in block (b)
    :return: (2m,) baseline array, (2m, 2m) excitation matrix
    """
    n_alphas = len(alphas_ab)
    # excitation matrix
    if n_alphas == 6:
        alpha_matrix = get_6_alphas_matrix_off_bp(alphas_ab, alphas_ba, n_a, n_b)
    elif n_alphas == 4:
        alpha_matrix = get_4_alphas_matrix_off_bp(alphas_ab, alphas_ba, n_a, n_b)
    else:
        alpha_matrix = get_2_alphas_matrix_off_bp(alphas_ab, alphas_ba, n_a, n_b)
    M_ab = n_a * n_b

    # mu array (2*M_ab,1)
    mu_array = np.array([mu_ab] * M_ab + [mu_ba] * M_ab)
    return mu_array, alpha_matrix

#%% Other helper function

def get_np_dia_list(n_nodes):
    """
    generate list of all possible node pairs in a diagonal block pair

    :param n_nodes: (int) number of nodes in the block
    :return: (list) list of node pairs (u, v)
    """
    nodes_list = list(range(n_nodes))
    np_list = []
    for i in nodes_list:
        for j in nodes_list:
            if i!=j: np_list.append((i,j))
    return np_list


def get_np_off_list(n_nodes_a, n_nodes_b):
    """
    generate list of all possible node pairs in an off-diagonal block pair

    :param n_nodes_a: number of nodes in block a
    :param n_nodes_b: number of nodes in block b
    :return: (list) list of node pairs (u, v)
    """
    N_a_list = list(range(n_nodes_a))
    N_b_list = list(range(n_nodes_a, n_nodes_a + n_nodes_b))
    np_list = []
    for i in N_a_list:
        for j in N_b_list:
            np_list.append((i,j))
    for i in N_b_list:
        for j in N_a_list:
            np_list.append((i,j))
    return np_list


def node_pair_index(node_pair, n_nodes):
    """
    index of a node pair in a list of all possible node pairs in a block pair

    nodes ids from [0: n_nodes-1]
    assume that list is ordered ascending with respect to u then v [(0,1), (0,2), .., (1,0), (1,2), ...]

    :param node_pair: (tuple) tuple of a node pair (u, v)
    :param n_nodes: (int) number of nodes in a diagonal block pair
    :return: (int) index of node pair
    """
    u, v = node_pair
    if v > u:
        return (n_nodes - 1) * u + v - 1
    else:
        return (n_nodes - 1) * u + v


def events_list_to_events_dict_remove_empty_np(events_list, a_nodes):
    events_dict = {}
    for u, i in zip(a_nodes, range(len(a_nodes))):
        for v, j in zip(a_nodes, range(len(a_nodes))):
            index = node_pair_index((i, j), len(a_nodes))
            if i != j and len(events_list[index]) > 0:
                events_dict[(u, v)] = events_list[index]
    return events_dict

def events_list_to_events_dict_remove_empty_np_off(events_list, a_nodes, b_nodes):
    if len(events_list) == 0:
        return {}
    events_dict = {}
    i = 0
    for u in a_nodes:
        for v in b_nodes:
            if u!=v:
                if len(events_list[i]) != 0:
                    events_dict[(u, v)] = events_list[i]
                i += 1
    return events_dict

