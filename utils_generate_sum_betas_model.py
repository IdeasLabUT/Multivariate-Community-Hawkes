import random
import numpy as np
from hawkes.MHP_Kernels import MHP_Kernels_2

""" parameters in matrix format for simulation and detailed function"""


#%% mulch sum of kernel simulation function
def simulate_sum_kernel_model(sim_param, n_nodes, n_classes, p, duration):
    if len(sim_param) == 5:
        mu_sim, alpha_n_sim, alpha_r_sim, C_sim, betas_sim = sim_param
    elif len(sim_param) == 7:
        mu_sim, alpha_n_sim, alpha_r_sim, alpha_br_sim, alpha_gr_sim, C_sim , betas_sim = sim_param
    elif len(sim_param) == 9:
        mu_sim, alpha_n_sim, alpha_r_sim, alpha_br_sim, alpha_gr_sim, alpha_al_sim, alpha_alr_sim, C_sim , betas_sim = sim_param
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
                        par = (mu_sim[i, j], alpha_n_sim[i, j], alpha_r_sim[i, j], C_sim[i, j], betas_sim)
                    elif len(sim_param) == 7:
                        par = (mu_sim[i, j], alpha_n_sim[i, j], alpha_r_sim[i, j], alpha_br_sim[i, j], alpha_gr_sim[i, j],
                           np.array(C_sim[i, j]), betas_sim)
                    elif len(sim_param) == 9:
                        par = (mu_sim[i, j], alpha_n_sim[i, j], alpha_r_sim[i, j], alpha_br_sim[i, j], alpha_gr_sim[i, j],
                           alpha_al_sim[i, j], alpha_alr_sim[i, j], np.array(C_sim[i, j]), betas_sim)
                    events_dict = simulate_sum_betas_dia_bp(par, list(class_nodes_list[i]), duration)
                    events_dict_all.update(events_dict)
            elif i < j:
                if len(sim_param) == 5:
                    par_ab = (mu_sim[i, j], alpha_n_sim[i, j], alpha_r_sim[i, j], C_sim[i, j], betas_sim)
                    par_ba = (mu_sim[j, i], alpha_n_sim[j, i], alpha_r_sim[j, i], C_sim[j, i], betas_sim)
                elif len(sim_param) == 7:
                    par_ab = (mu_sim[i, j], alpha_n_sim[i, j], alpha_r_sim[i, j], alpha_br_sim[i, j], alpha_gr_sim[i, j],
                              np.array(C_sim[i, j]), betas_sim)
                    par_ba = (mu_sim[j, i], alpha_n_sim[j, i], alpha_r_sim[j, i], alpha_br_sim[j, i], alpha_gr_sim[j, i],
                              np.array(C_sim[j, i]) ,betas_sim)
                elif len(sim_param) == 9:
                    par_ab = (mu_sim[i, j], alpha_n_sim[i, j], alpha_r_sim[i, j], alpha_br_sim[i, j], alpha_gr_sim[i, j],
                              alpha_al_sim[i, j], alpha_alr_sim[i, j], np.array(C_sim[i, j]), betas_sim)
                    par_ba = (mu_sim[j, i], alpha_n_sim[j, i], alpha_r_sim[j, i], alpha_br_sim[j, i], alpha_gr_sim[j, i],
                              alpha_al_sim[j, i], alpha_alr_sim[j, i], np.array(C_sim[j, i]) ,betas_sim)
                d_ab, d_ba = simulate_sum_betas_off_bp(par_ab, par_ba, list(class_nodes_list[i]), list(class_nodes_list[j]),
                                                                 duration)
                events_dict_all.update(d_ab)
                events_dict_all.update(d_ba)
    return events_dict_all, node_mem_actual
#%% sum of betas block pair simulation functions (diagonal & off-diagonal)

def simulate_sum_betas_dia_bp(par, a_nodes, end_time, return_list=False):
    # kernel: alpha*beta*exp(-beta*t)
    if len(par) == 5:
        mu, alpha_n, alpha_r, C, betas = par
        alphas = alpha_n, alpha_r
        mu_array, alpha_matrix = get_mu_array_alpha_matrix_dia_bp(mu, alphas, len(a_nodes))
    elif len(par) == 7:
        mu, alpha_n, alpha_r, alpha_br, alpha_gr, C, betas = par
        alphas = alpha_n, alpha_r, alpha_br, alpha_gr
        mu_array, alpha_matrix= get_mu_array_alpha_matrix_dia_bp(mu, alphas, len(a_nodes))
    else:
        mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, C, betas = par
        alphas = alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr
        mu_array, alpha_matrix = get_mu_array_alpha_matrix_dia_bp(mu, alphas, len(a_nodes))
    P = MHP_Kernels_2(mu=mu_array, alpha=alpha_matrix, C=C, betas=betas)
    P.generate_seq(end_time)
    # assume that timestamps list is ordered ascending with respect to u then v [(0,1), (0,2), .., (1,0), (1,2), ...]
    events_list = []
    for m in range(len(mu_array)):
        i = np.nonzero(P.data[:, 1] == m)[0]
        events_list.append(P.data[i, 0])
    events_dict = events_list_to_events_dict_remove_empty_np(events_list, a_nodes)
    if return_list:
        return events_list, events_dict
    return events_dict

def simulate_sum_betas_off_bp(par_ab, par_ba, a_nodes, b_nodes, end_time, return_list=False):

    # parameters = (mu, alphas, C, betas)
    if len(par_ba) == 5: # 2-alpha model
        mu_array, alpha_matrix = get_mu_array_alpha_matrix_off_bp(par_ab[0], par_ab[1:3], par_ba[0], par_ba[1:3],
                                                                  len(a_nodes), len(b_nodes))
    elif len(par_ba) == 7: # 4-alpha model
        mu_array, alpha_matrix = get_mu_array_alpha_matrix_off_bp(par_ab[0], par_ab[1:5], par_ba[0], par_ba[1:5],
                                                                  len(a_nodes), len(b_nodes))
    elif len(par_ba) == 9: # 6-alpha model
        mu_array, alpha_matrix = get_mu_array_alpha_matrix_off_bp(par_ab[0], par_ab[1:7], par_ba[0], par_ba[1:7],
                                                                  len(a_nodes), len(b_nodes))
    P = MHP_Kernels_2(mu=mu_array, alpha=alpha_matrix, C=par_ab[-2], C_r=par_ba[-2], betas=par_ab[-1])
    P.generate_seq(end_time)

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

def get_6_alphas_matrix_dia_bp(alphas, n_nodes):
    alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr = alphas
    # add alpha_n, alpha_br to alpha_matrix
    block = (np.ones((n_nodes - 1, n_nodes - 1)) - np.identity(n_nodes - 1)) * alpha_br + np.identity(n_nodes - 1) * alpha_n
    alpha_matrix = np.kron(np.eye(n_nodes), block)
    np_list = get_np_dia_list(n_nodes)
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
def get_6_alphas_matrix_off_bp(alphas_ab, alphas_ba, N_a, N_b):
    alpha_n_ab, alpha_r_ab, alpha_br_ab, alpha_gr_ab, alpha_al_ab, alpha_alr_ab = alphas_ab
    alpha_n_ba, alpha_r_ba, alpha_br_ba, alpha_gr_ba, alpha_al_ba, alpha_alr_ba = alphas_ba
    M_ab = N_a * N_b
    # alpha_matrix (2M_ab , 2M_ab)
    alpha_matrix = np.zeros((2*M_ab , 2*M_ab))
    np_list = get_np_off_list(N_a, N_b)
    # loop through node pairs in block pair (a, b)
    for from_idx, (i, j) in enumerate(np_list[:M_ab]):
        # loop through all node pairs
        for to_idx, (x, y) in enumerate(np_list):
            # alpha_n
            if (i, j) == (x, y):
                alpha_matrix[from_idx, to_idx] = alpha_n_ab
            # alpha_r
            elif (i, j) == (y, x):
                alpha_matrix[from_idx, to_idx] = alpha_r_ab
            # alpha_br
            elif i==x:
                alpha_matrix[from_idx, to_idx] = alpha_br_ab
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
            # alpha_n
            if (i, j) == (x, y):
                alpha_matrix[from_idx, to_idx] = alpha_n_ba
            # alpha_r
            elif (i, j) == (y, x):
                alpha_matrix[from_idx, to_idx] = alpha_r_ba
            # alpha_br
            elif i==x:
                alpha_matrix[from_idx, to_idx] = alpha_br_ba
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

# model (n, r, br, gr) array parameters
def get_4_alphas_matrix_dia_bp(alphas, n_nodes):
    alpha_n, alpha_r, alpha_br, alpha_gr = alphas
    nodes_set = set(np.arange(n_nodes))  # set of nodes
    block = (np.ones((n_nodes - 1, n_nodes - 1)) - np.identity(n_nodes - 1)) * alpha_br + np.identity(n_nodes - 1) * alpha_n
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
    M = n_nodes_a * n_nodes_b  # number of nodes pair per block pair
    alpha_n_ab, alpha_r_ab, alpha_br_ab, alpha_gr_ab = alphas_ab
    alpha_n_ba, alpha_r_ba, alpha_br_ba, alpha_gr_ba = alphas_ba
    ### alpha matrix (2M,2M)
    # alpha ab-ab
    block = (np.ones((n_nodes_b, n_nodes_b)) - np.identity(n_nodes_b)) * alpha_br_ab + np.identity(
        n_nodes_b) * alpha_n_ab
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
    block = (np.ones((n_nodes_a, n_nodes_a)) - np.identity(n_nodes_a)) * alpha_br_ba + np.identity(
        n_nodes_a) * alpha_n_ba
    alpha_matrix_ba_ba = np.kron(np.eye(n_nodes_b), block)
    alpha_matrix = np.vstack(
        (np.hstack((alpha_matrix_ab_ab, alpha_matrix_ab_ba)), np.hstack((alpha_matrix_ba_ab, alpha_matrix_ba_ba))))
    return alpha_matrix


def get_2_alphas_matrix_dia_bp(alphas, n_nodes):
    alpha_n, alpha_r = alphas
    block = np.identity(n_nodes - 1) * alpha_n
    alpha_matrix = np.kron(np.eye(n_nodes), block)
    # add alpha_r assuming node are ordered
    for u in range(n_nodes):
        for v in range(n_nodes):
            if u != v:
                alpha_matrix[node_pair_index((u, v), n_nodes), node_pair_index((v, u), n_nodes)] = alpha_r
    return alpha_matrix
def get_2_alphas_matrix_off_bp(alphas_ab, alphas_ba, n_nodes_a, n_nodes_b):
    M = n_nodes_a * n_nodes_b  # number of processes
    alpha_n_ab, alpha_r_ab = alphas_ab
    alpha_n_ba, alpha_r_ba = alphas_ba

    ### alpha matrix (2M,2M)
    # alpha ab-ab
    alpha_matrix_ab_ab = np.identity(M) * alpha_n_ab
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
    alpha_matrix_ba_ba = np.identity(M) * alpha_n_ba
    # combine four alpha_matrix
    alpha_matrix = np.vstack(
        (np.hstack((alpha_matrix_ab_ab, alpha_matrix_ab_ba)), np.hstack((alpha_matrix_ba_ab, alpha_matrix_ba_ba))))
    return alpha_matrix

def get_mu_array_alpha_matrix_dia_bp(mu, alphas, n_nodes):
    n_alphas = len(alphas)
    M = n_nodes * (n_nodes - 1)  # number of node pairs in block pair
    # excitation matrix
    if n_alphas == 6:
        alpha_matrix = get_6_alphas_matrix_dia_bp(alphas, n_nodes)
    elif n_alphas == 4:
        alpha_matrix = get_4_alphas_matrix_dia_bp(alphas, n_nodes)
    else:
        alpha_matrix = get_2_alphas_matrix_dia_bp(alphas, n_nodes)
    # mu array (M, 1)
    mu_array = np.ones(M) * mu
    return mu_array, alpha_matrix


def get_mu_array_alpha_matrix_off_bp(mu_ab, alphas_ab, mu_ba, alphas_ba, n_nodes_a, n_nodes_b):
    n_alphas = len(alphas_ab)
    # excitation matrix
    if n_alphas == 6:
        alpha_matrix = get_6_alphas_matrix_off_bp(alphas_ab, alphas_ba, n_nodes_a, n_nodes_b)
    elif n_alphas == 4:
        alpha_matrix = get_4_alphas_matrix_off_bp(alphas_ab, alphas_ba, n_nodes_a, n_nodes_b)
    else:
        alpha_matrix = get_2_alphas_matrix_off_bp(alphas_ab, alphas_ba, n_nodes_a, n_nodes_b)
    M_ab = n_nodes_a* n_nodes_b

    # mu array (2*M_ab,1)
    mu_array = np.array([mu_ab] * M_ab + [mu_ba] * M_ab)
    return mu_array, alpha_matrix

#%% helper function

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

