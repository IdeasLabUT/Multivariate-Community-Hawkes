import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, approx_fprime
from bisect import bisect_left
import time
import random
import sys

from os import path, getcwd
sys.path.append(path.join(getcwd(), "hawkes"))
from MHP import MHP
from MHP_Kernels_2 import MHP_Kernels_2


# %% one block_pair simulation functions
"""
Simulation function using new kernel: beta*alpha*exp(-beta*t)
"""
# single beta model - diagonal and off-diagonal block pairs
def simulate_one_beta_dia_2(params_sim, a_nodes, end_time, return_list=False):
    if len(params_sim) == 4:
        mu_array, alpha_matrix, beta = get_array_param_n_r_dia(params_sim, len(a_nodes))
    elif len(params_sim) == 6:
        mu_array, alpha_matrix, beta = get_array_param_n_r_br_gr_dia(params_sim, len(a_nodes))
    elif len(params_sim) == 8:
        mu_array, alpha_matrix, beta = get_array_param_n_r_br_gr_al_alr_dia(params_sim, len(a_nodes))
    # multivariate hawkes process object [NOTE: alpha=jump_size/beta, omega=beta]
    P = MHP(mu=mu_array, alpha=alpha_matrix, omega=beta)
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
def simulate_one_beta_off_2(param_ab, param_ba, a_nodes, b_nodes, end_time, return_list=False):
    if len(param_ab) == 4:
        mu_array, alpha_matrix, beta = get_array_param_n_r_off(param_ab, param_ba, len(a_nodes), len(b_nodes))
    elif len(param_ab) == 6:
        mu_array, alpha_matrix, beta = get_array_param_n_r_br_gr_off(param_ab, param_ba, len(a_nodes), len(b_nodes))
    if len(param_ab) == 8:
        mu_array, alpha_matrix, beta = get_array_param_n_r_br_gr_al_alr_off(param_ab, param_ba, len(a_nodes), len(b_nodes))
    # multivariate hawkes process object [NOTE: alpha=jump_size/beta, omega=beta]
    P = MHP(mu=mu_array, alpha=alpha_matrix, omega=beta)
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
# sum of kernels model - diagonal and off-diagonal block pairs
def simulate_kernel_sum_dia_2(param, a_nodes, end_time, return_list=False):
    # kernel: alpha*beta*exp(-beta*t)
    if len(param) == 5:
        mu, alpha_n, alpha_r, C, betas = param
        params_sim = (mu, alpha_n, alpha_r, 1)
        mu_array, alpha_matrix, _ = get_array_param_n_r_dia(params_sim, len(a_nodes))
    elif len(param) == 7:
        mu, alpha_n, alpha_r, alpha_br, alpha_gr, C, betas = param
        params_sim = (mu, alpha_n, alpha_r, alpha_br, alpha_gr, 1)
        mu_array, alpha_matrix, _ = get_array_param_n_r_br_gr_dia(params_sim, len(a_nodes))
    elif len(param) == 9:
        mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, C, betas = param
        params_sim = (mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, 1)
        mu_array, alpha_matrix, _ = get_array_param_n_r_br_gr_al_alr_dia(params_sim, len(a_nodes))
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
def simulate_kernel_sum_off_2(p_sum_ab, p_sum_ba, a_nodes, b_nodes, end_time, return_list=False):
    if len(p_sum_ba) == 5:
        mu, alpha_n, alpha_r, C, betas = p_sum_ab
        param_ab = (mu, alpha_n, alpha_r, 1)
        mu, alpha_n, alpha_r, C_r, betas = p_sum_ba
        param_ba = (mu, alpha_n, alpha_r, 1)
        mu_array, alpha_matrix, _ = get_array_param_n_r_off(param_ab, param_ba, len(a_nodes), len(b_nodes))
    elif len(p_sum_ba) == 7:
        mu, alpha_n, alpha_r, alpha_br, alpha_gr, C, betas = p_sum_ab
        param_ab = (mu, alpha_n, alpha_r, alpha_br, alpha_gr, 1)
        mu, alpha_n, alpha_r, alpha_br, alpha_gr, C_r, betas = p_sum_ba
        param_ba = (mu, alpha_n, alpha_r, alpha_br, alpha_gr, 1)
        mu_array, alpha_matrix, _ = get_array_param_n_r_br_gr_off(param_ab, param_ba, len(a_nodes), len(b_nodes))
    elif len(p_sum_ba) == 9:
        mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, C, betas = p_sum_ab
        param_ab = (mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, 1)
        mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, C_r, betas = p_sum_ba
        param_ba = (mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, 1)
        mu_array, alpha_matrix, _ = get_array_param_n_r_br_gr_al_alr_off(param_ab, param_ba, len(a_nodes), len(b_nodes))
    P = MHP_Kernels_2(mu=mu_array, alpha=alpha_matrix, C=C, C_r=C_r, betas=betas)
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

# %% helper functions

def plot_kernel(alpha, betas, C, time_range):
    lambda_sum = []
    for t in time_range:
        lambda_sum.append( alpha*np.sum(betas * C * np.exp(-t*betas)))
    plt.figure()
    plt.plot(time_range, lambda_sum, color='red', label=f"betas1={betas}")
    plt.xlabel("t(s)")
    plt.ylabel("lambda(t)")
    plt.yscale('log')
    plt.title('sum of kernels C=[0.33, 0.33, 0.34] - different betas')
    plt.legend()
    plt.grid(True)
    plt.show()

def print_param(param):
    if len(param)==4:
        print(f"mu={param[0]:.7f}, alpha_n={param[1]:.4f}, alpha_r={param[2]:.4f}, beta={param[3]:.3f}")
    elif len(param)==6:
        print(f"mu={param[0]:.7f}, alpha_n={param[1]:.4f}, alpha_r={param[2]:.4f}, alpha_br={param[3]:.4f},"
          f" alpha_gr={param[4]:.4f}, beta={param[5]:.3f}")
    elif len(param)==8:
        print(f"mu={param[0]:.7f}, alpha_n={param[1]:.2f}, alpha_r={param[2]:.2f}, alpha_br={param[3]:.5f},"
              f" alpha_gr={param[4]:.5f}, alpha_al={param[5]:.4f}, alpha_alr={param[6]:.4f}, beta={param[7]:.3f}")

def print_param_kernels(param):
    if len(param) == 9:
        print(f"C = {param[7]}")
        print(f"mu={param[0]:.7f}, alpha_n={param[1]:.4f}, alpha_r={param[2]:.4f}, alpha_br={param[3]:.4f},"
              f" alpha_gr={param[4]:.4f}, alpha_al={param[5]:.4f}, alpha_alr={param[6]:.4f}")
    elif len(param) == 7:
        print(f"C = {param[5]}")
        print(f"mu={param[0]:.7f}, alpha_n={param[1]:.4f}, alpha_r={param[2]:.4f}, alpha_br={param[3]:.4f},"
              f" alpha_gr={param[4]:.4f}")
    elif len(param) == 5:
        print(f"C = {param[3]}")
        print(f"mu={param[0]:.7f}, alpha_n={param[1]:.4f}, alpha_r={param[2]:.4f}")

def cal_num_events_2(events_dict):
    num_events = 0
    for events_array in events_dict.values():
        num_events += len(events_array)
    return num_events

def pair_index(node_pair, n_nodes):
    # nodes numbering start from 0 to n_nodes-1
    # assume that events_list is ordered ascending with respect to u then v [(0,1), (0,2), .., (1,0), (1,2), ...]
    u, v = node_pair
    if v > u:
        return (n_nodes - 1) * u + v - 1
    else:
        return (n_nodes - 1) * u + v

def events_list_to_events_dict_remove_empty_np(events_list, a_nodes):
    events_dict = {}
    for u, i in zip(a_nodes, range(len(a_nodes))):
        for v, j in zip(a_nodes, range(len(a_nodes))):
            index = pair_index((i, j), len(a_nodes))
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

def get_Ri_temp_Q(uv_events, e_intertimes_Q, xy_events, betas):
    Q = len(betas)
    num_events_uv = len(uv_events)
    Ri_temp = np.zeros((num_events_uv, Q))
    prev_index = 0
    for k in range(0, num_events_uv):
        # return index below which t(x,y) < kth event of (u,v)
        # if no events exists returns len(events(x,y))
        index = bisect_left(xy_events, uv_events[k], lo=prev_index)
        # no events found
        if index == prev_index:
            if k == 0: continue
            for q in range(Q):
                Ri_temp[k, q] = e_intertimes_Q[k - 1, q] * Ri_temp[k - 1, q]
        else:
            diff_times = uv_events[k] - xy_events[prev_index:index]
            if k == 0:
                for q in range(Q):
                    Ri_temp[k, q] = np.sum(np.exp(-betas[q] * diff_times))
            else:
                for q in range(Q):
                    Ri_temp[k, q] = e_intertimes_Q[k - 1, q] * Ri_temp[k - 1, q] + np.sum(np.exp(-betas[q] * diff_times))
            prev_index = index
    return Ri_temp

""" parameters in matrix format for simulation and detailed function"""
# model (n, r, br, gr, al, alr) array parameters
def get_np_dia_list(N):
    nodes_list = list(range(N))
    np_list = []
    for i in nodes_list:
        for j in nodes_list:
            if i!=j: np_list.append((i,j))
    return np_list
def get_alpha_n_r_br_gr_al_alr_dia(alphas, n_nodes):
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
def get_array_param_n_r_br_gr_al_alr_dia(param, n_nodes):
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, beta = param
    nodes_set = set(np.arange(n_nodes))  # set of nodes
    M = n_nodes * (n_nodes - 1)  # number of processes
    alpha_matrix = get_alpha_n_r_br_gr_al_alr_dia((alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr), n_nodes)
    mu_array = np.ones(M) * mu
    return mu_array, alpha_matrix, beta
def get_np_off_list(N_a, N_b):
    N_a_list = list(range(N_a))
    N_b_list = list(range(N_a, N_a + N_b))
    np_list = []
    for i in N_a_list:
        for j in N_b_list:
            np_list.append((i,j))
    for i in N_b_list:
        for j in N_a_list:
            np_list.append((i,j))
    return np_list
def get_alpha_n_r_br_gr_al_alr_off(alphas_ab, alphas_ba, N_a, N_b):
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
def get_array_param_n_r_br_gr_al_alr_off(p_ab, p_ba, N_a, N_b):
    alpha_matrix = get_alpha_n_r_br_gr_al_alr_off(p_ab[1:-1], p_ba[1:-1], N_a, N_b)
    M_ab = N_a* N_b
    ### mu array (2*M_ab,1)
    mu_array = np.array([p_ab[0]] * M_ab + [p_ba[0]] * M_ab)
    return mu_array, alpha_matrix, p_ab[-1]

# model (n, r, br, gr) array parameters
def get_alpha_n_r_br_gr_dia(alphas, n_nodes):
    alpha_n, alpha_r, alpha_br, alpha_gr = alphas
    nodes_set = set(np.arange(n_nodes))  # set of nodes
    block = (np.ones((n_nodes - 1, n_nodes - 1)) - np.identity(n_nodes - 1)) * alpha_br + np.identity(n_nodes - 1) * alpha_n
    alpha_matrix = np.kron(np.eye(n_nodes), block)
    # add alpha_r , alpha_gr parameters assuming node are ordered
    for u in range(n_nodes):
        for v in range(n_nodes):
            if u != v:
                alpha_matrix[pair_index((u, v), n_nodes), pair_index((v, u), n_nodes)] = alpha_r
                nodes_minus = nodes_set - {v} - {u}
                for i in nodes_minus:
                    alpha_matrix[pair_index((u, v), n_nodes), pair_index((i, u), n_nodes)] = alpha_gr
    return alpha_matrix
def get_array_param_n_r_br_gr_dia(param_r, n_nodes):
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, beta = param_r
    nodes_set = set(np.arange(n_nodes))  # set of nodes
    M = n_nodes * (n_nodes - 1)  # number of processes
    alpha_matrix = get_alpha_n_r_br_gr_dia((alpha_n, alpha_r, alpha_br, alpha_gr), n_nodes)
    mu_array = np.ones(M) * mu
    return mu_array, alpha_matrix, beta
def get_array_param_n_r_br_gr_off(param_ab, param_ba, n_nodes_a, n_nodes_b):
    M = n_nodes_a * n_nodes_b  # number of processes
    mu_ab, alpha_n_ab, alpha_r_ab, alpha_br_ab, alpha_gr_ab, beta_ab = param_ab
    mu_ba, alpha_n_ba, alpha_r_ba, alpha_br_ba, alpha_gr_ba, beta_ba = param_ba

    ### alpha matrix (2M,2M)
    # alpha ab-ab
    block = (np.ones((n_nodes_b, n_nodes_b)) - np.identity(n_nodes_b)) * alpha_br_ab + np.identity(n_nodes_b) * alpha_n_ab
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
    block = (np.ones((n_nodes_a, n_nodes_a)) - np.identity(n_nodes_a)) * alpha_br_ba + np.identity(n_nodes_a) * alpha_n_ba
    alpha_matrix_ba_ba = np.kron(np.eye(n_nodes_b), block)
    alpha_matrix = np.vstack((np.hstack((alpha_matrix_ab_ab, alpha_matrix_ab_ba)), np.hstack((alpha_matrix_ba_ab, alpha_matrix_ba_ba))))
    ### mu array (2M,1)
    mu_array = np.array([mu_ab] * M + [mu_ba] * M)
    beta = beta_ab
    return mu_array, alpha_matrix, beta

# model (n, r) array parameters
def get_array_param_n_r_dia(param, n_nodes):
    mu, alpha_n, alpha_r, beta = param
    M = n_nodes * (n_nodes - 1)  # number of processes
    block = np.identity(n_nodes - 1) * alpha_n
    alpha_matrix = np.kron(np.eye(n_nodes), block)
    # add alpha_r assuming node are ordered
    for u in range(n_nodes):
        for v in range(n_nodes):
            if u != v:
                alpha_matrix[pair_index((u, v), n_nodes), pair_index((v, u), n_nodes)] = alpha_r
    mu_array = np.ones(M) * mu
    return mu_array, alpha_matrix, beta
def get_array_param_n_r_off(param_ab, param_ba, n_nodes_a, n_nodes_b):
    M = n_nodes_a * n_nodes_b  # number of processes
    mu_ab, alpha_n_ab, alpha_r_ab, beta_ab = param_ab
    mu_ba, alpha_n_ba, alpha_r_ba, beta_ba = param_ba

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
    alpha_matrix = np.vstack((np.hstack((alpha_matrix_ab_ab, alpha_matrix_ab_ba)), np.hstack((alpha_matrix_ba_ab, alpha_matrix_ba_ba))))

    ### mu array (2M,1)
    mu_array = np.array([mu_ab] * M + [mu_ba] * M)
    beta = beta_ab
    return mu_array, alpha_matrix, beta

#%% single beta (n, r, br, gr, al, alr) model --- beta*alpha*exp(-beta*t)
""" helper functions"""
def cal_diff_sum(events_dict, end_time, beta):
    events_array = list(events_dict.values())
    if len(events_array) !=0 :
        return np.sum(1 - np.exp(-beta * (end_time - np.concatenate(events_array))))
    else:
        return 0
def cal_Ri_temp(n_uv_events, uv_events, uv_inter_t, xy_events, beta):
    Ri_temp = np.zeros((n_uv_events, 1))
    prev_index = 0
    for k in range(0, n_uv_events):
        # return index below which t(x,y) < kth event of (u,v)
        # if no events exists returns len(events(x,y))
        index = bisect_left(xy_events, uv_events[k], lo=prev_index)
        # no events found
        if index == prev_index:
            if k == 0: continue
            Ri_temp[k, 0] = uv_inter_t[k - 1] * Ri_temp[k - 1, 0]
        else:
            if k == 0:
                Ri_temp[k, 0] = np.sum(np.exp(-beta * (uv_events[k] - xy_events[prev_index:index])))
            else:
                Ri_temp[k, 0] = uv_inter_t[k - 1] * Ri_temp[k - 1, 0] + np.sum(
                    np.exp(-beta * (uv_events[k] - xy_events[prev_index:index])))
            prev_index = index
    return Ri_temp
def cal_R_n_r_br_gr_al_alr_dia(events_dict, beta):
    Ris = []
    for (u, v) in events_dict:
        # array of events of node pair (u,v)
        uv_events = events_dict[(u, v)]
        num_events_uv = len(uv_events)
        if num_events_uv == 0:
            Ris.append(np.array([0]))  # R=0 if node_pair (u,v) has no events
        else:
            # 6 columns (alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr)
            Ri = np.zeros((num_events_uv, 6))
            intertimes = np.exp(-beta * (uv_events[1:] - uv_events[:-1]))
            for (x, y) in events_dict:
                # same node_pair events (alpha_n)
                if (u, v) == (x, y):
                    for k in range(1, num_events_uv):
                        Ri[k, 0] = intertimes[k - 1] * (1 + Ri[k - 1, 0])
                # reciprocal node_pair events (alpha_r)
                elif (v, u) == (x, y):
                    Ri_temp = cal_Ri_temp(num_events_uv, uv_events, intertimes, events_dict[(x, y)], beta)
                    Ri[:, 1] = Ri_temp[:, 0]
                # broadcast node_pairs events (alpha_br)
                elif u == x and v != y:
                    Ri_temp = cal_Ri_temp(num_events_uv, uv_events, intertimes, events_dict[(x, y)], beta)
                    Ri[:, 2] += Ri_temp[:, 0]
                # gr node_pairs events (alpha_gr)
                elif u == y and v != x:
                    Ri_temp = cal_Ri_temp(num_events_uv, uv_events, intertimes, events_dict[(x, y)], beta)
                    Ri[:, 3] += Ri_temp[:, 0]
                # alliance np (alpha_al)
                elif v == y and u != x:
                    Ri_temp = cal_Ri_temp(num_events_uv, uv_events, intertimes, events_dict[(x, y)], beta)
                    Ri[:, 4] += Ri_temp[:, 0]
                # alliance reciprocal np (alpha_alr)
                elif v == x and u != y:
                    Ri_temp = cal_Ri_temp(num_events_uv, uv_events, intertimes, events_dict[(x, y)], beta)
                    Ri[:, 5] += Ri_temp[:, 0]
            Ris.append(Ri)
    # return list of arrays - list size = #node_pairs_events_in block_pair
    return Ris
def cal_R_n_r_br_gr_al_alr_off(events_dict, events_dict_r, beta):
    Ris = []
    for (u, v) in events_dict:
        uv_events = events_dict[(u, v)]  # array of events of node pair (u,v)
        num_events_uv = len(uv_events)  # check if node pair (u,v) has no events
        if num_events_uv == 0:
            Ris.append(np.array([0]))  # R=0 if node_pair (u,v) has no events
        else:
            Ri = np.zeros((num_events_uv, 6))  # 6 columns for [alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr]
            intertimes = np.exp(-beta * (uv_events[1:] - uv_events[:-1]))
            # loop through node pairs in block pair ab
            for (x, y) in events_dict:
                # same node_pair events (alpha_n)
                if (u, v) == (x, y):
                    for k in range(1, num_events_uv):
                        Ri[k, 0] = intertimes[k - 1] * (1 + Ri[k - 1, 0]) # reciprocal node_pair events (alpha_r)
                # broadcast alpha_br
                elif u == x:
                    Ri_temp = cal_Ri_temp(num_events_uv, uv_events, intertimes, events_dict[(x, y)], beta)
                    Ri[:, 2] += Ri_temp[:, 0] # alliance np (alpha_al)
                # alliance alpha_al
                elif v == y:
                    Ri_temp = cal_Ri_temp(num_events_uv, uv_events, intertimes, events_dict[(x, y)], beta)
                    Ri[:, 4] += Ri_temp[:, 0]
            # loop through node pairs in reciprocal block pair ba
            for (x, y) in events_dict_r:
                # reciprocal node_pair events (alpha_r)
                if (v, u) == (x, y):
                    Ri_temp = cal_Ri_temp(num_events_uv, uv_events, intertimes, events_dict_r[(x, y)], beta)
                    Ri[:, 1] = Ri_temp[:, 0]
                # gr node_pairs events (alpha_gr)
                elif u == y and v != x:
                    Ri_temp = cal_Ri_temp(num_events_uv, uv_events, intertimes, events_dict_r[(x, y)], beta)
                    Ri[:, 3] += Ri_temp[:, 0]
                # alliance reciprocal np (alpha_alr)
                elif v == x and u != y:
                    Ri_temp = cal_Ri_temp(num_events_uv, uv_events, intertimes, events_dict_r[(x, y)], beta)
                    Ri[:, 5] += Ri_temp[:, 0]
            Ris.append(Ri)
    # return list of arrays - list size = #node_pairs_events_in block_pair
    return Ris

""" diagonal block pairs"""
def LL_n_r_br_gr_al_alr_one_dia(params, events_dict, end_time, n_nodes, M, Ris=None, diff_sum=None):
    # events_dict of node_pairs with events
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, beta = params
    ##first term
    first = -M * mu * end_time
    ##second term
    second = 0
    if diff_sum is None:
        diff_sum = cal_diff_sum(events_dict, end_time, beta)
    second = -((alpha_n + alpha_r + (alpha_br + alpha_gr + alpha_al + alpha_alr) * (n_nodes - 2)) * diff_sum)
    ##third term
    if Ris is None:  # only calculate Ris if beta is not fixed (Ris is not passed)
        Ris = cal_R_n_r_br_gr_al_alr_dia(events_dict, beta)  # list of M_np elements, each is (n_events_np, 6) array
    third = 0
    for i in range(len(Ris)):
        third += np.sum(np.log(mu + beta*(alpha_n * Ris[i][:, 0] + alpha_r * Ris[i][:, 1] +
                                          alpha_br * Ris[i][:, 2] + alpha_gr * Ris[i][:, 3] +
                                          alpha_al * Ris[i][:, 4] + alpha_alr * Ris[i][:, 5])))
    log_likelihood_value = first + second + third
    return log_likelihood_value
def NLL_n_r_br_gr_al_alr_one_dia(params, events_dict, end_time, n_nodes, M, beta, Ris, diff_sum):
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr = params
    params_fixed_b = mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, beta
    return -LL_n_r_br_gr_al_alr_one_dia(params_fixed_b, events_dict, end_time, n_nodes, M, Ris, diff_sum)
def NLL_n_r_br_gr_al_alr_one_dia_jac(params, events_dict, end_time, n_nodes, M, beta, Ris, diff_sum):
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr = params
    d_params = np.zeros(len(params))
    d_params[0] = M * end_time
    d_params[1:3] = diff_sum
    d_params[3:] = (n_nodes - 2) * diff_sum
    for i in range(len(Ris)):
        denominator = mu + beta * (alpha_n * Ris[i][:, 0] + alpha_r * Ris[i][:, 1] +
                                   alpha_br * Ris[i][:, 2] + alpha_gr * Ris[i][:, 3]+
                                   alpha_al * Ris[i][:, 4] + alpha_alr * Ris[i][:, 5])
        d_params[0] -= np.sum(1 / (denominator))
        for p in range(1,len(params)):
            d_params[p] -= np.sum(beta*Ris[i][:, p-1] / (denominator))
    return d_params
def fit_n_r_br_gr_al_alr_one_dia(events_dict, end_time, n_nodes, M, beta):
    # events_dict : (u,v):array_of_events
    if len(events_dict) == 0:  # handling empty block pair with no events
        return (1e-10, 0, 0, 0, 0, 0, 0, beta)
    mu_i = np.random.uniform(1e-6, 1e-2)
    alpha_n_i, alpha_r_i = np.random.uniform(0.1, 0.5, 2)
    alpha_br_i, alpha_gr_i, alpha_al_i, alpha_alr_i = np.random.uniform(1e-5, 0.1, 4)
    init_param = tuple([mu_i, alpha_n_i, alpha_r_i, alpha_br_i, alpha_gr_i, alpha_al_i, alpha_alr_i]) # <-- random initialization
    # init_param = (1e-2, 2e-2, 2e-2, 1e-2, 1e-2, 1e-2, 1e-2)
    Ris = cal_R_n_r_br_gr_al_alr_dia(events_dict, beta)
    diff_sum = cal_diff_sum(events_dict, end_time, beta)
    res = minimize(NLL_n_r_br_gr_al_alr_one_dia, init_param, method='L-BFGS-B',
                   bounds=((1e-7, None), (1e-7, None), (1e-7, None), (1e-7, None), (1e-7, None), (1e-7, None), (1e-7, None)),
                   jac=NLL_n_r_br_gr_al_alr_one_dia_jac,
                   args=(events_dict, end_time, n_nodes, M, beta, Ris, diff_sum), tol=1e-12)
    results = res.x
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr = results[0:]
    return (mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, beta)

""" off-diagonal block pairs"""
def LL_n_r_br_gr_al_alr_one_off(params, events_dict, events_dict_r, end_time, N_b, M_ab, Ris=None, diff_sum=None, diff_sum_r=None):
    N_a = M_ab//N_b # number of nodes in block a
    # events_dict of node_pairs with events
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, beta = params
    ##first term
    first = -M_ab * mu * end_time
    ##second term (block pair (ab) + block pair (ba))
    second = 0
    if diff_sum is None:
        diff_sum = cal_diff_sum(events_dict, end_time, beta)
        diff_sum_r = cal_diff_sum(events_dict_r, end_time, beta)
    second -= (alpha_n + alpha_br * (N_b - 1) + alpha_al  * (N_a - 1)) * diff_sum
    second -= (alpha_r + alpha_gr * (N_b - 1) + alpha_alr * (N_a - 1)) * diff_sum_r
    ##third term
    if Ris is None:  # only calculate Ris if beta is a variable
        Ris = cal_R_n_r_br_gr_al_alr_off(events_dict, events_dict_r, beta)  # list of M_np elements, each is (n_events_np,6) array
    third = 0
    for i in range(len(Ris)):
        third += np.sum(np.log(mu + beta * (alpha_n * Ris[i][:, 0] + alpha_r * Ris[i][:, 1]
                                            + alpha_br * Ris[i][:, 2] + alpha_gr * Ris[i][:, 3]
                                            + alpha_al * Ris[i][:, 4] + alpha_alr * Ris[i][:, 5])))
    log_likelihood_value = first + second + third
    # print("LL ", first, second, third)
    return log_likelihood_value
def NLL_n_r_br_gr_al_alr_one_off(p, d_ab, d_ba, end_time, n_nodes_to, M, beta, Ris, diff_sum, diff_sum_r):
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr = p
    param = mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, beta
    return -LL_n_r_br_gr_al_alr_one_off(param, d_ab, d_ba, end_time, n_nodes_to, M, Ris, diff_sum, diff_sum_r)
def NLL_n_r_br_gr_al_alr_one_off_jac(params, events_dict, events_dict_r, end_time, N_b, M_ab, beta, Ris, diff_sum, diff_sum_r):
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr = params
    N_a = M_ab//N_b
    d_params = np.zeros(len(params))
    d_params[0] = M_ab * end_time # d_mu
    d_params[1] = diff_sum # d_alpha_n
    d_params[2] = diff_sum_r  # d_alpha_r
    d_params[3] = (N_b - 1) * diff_sum # d_alpha_br
    d_params[4] = (N_b - 1) * diff_sum_r # d_alpha_gr
    d_params[5] = (N_a - 1) * diff_sum # d_alpha_al
    d_params[6] = (N_a - 1) * diff_sum_r # d_alpha_alr
    for i in range(len(Ris)):
        denominator = mu + beta * (alpha_n  * Ris[i][:, 0] + alpha_r   * Ris[i][:, 1] +
                                   alpha_br * Ris[i][:, 2] + alpha_gr  * Ris[i][:, 3] +
                                   alpha_al * Ris[i][:, 4] + alpha_alr * Ris[i][:, 5])
        d_params[0] -= np.sum(1 / denominator)
        for p in range(1, len(params)):
            d_params[p] -= np.sum(beta * Ris[i][:, p - 1] / denominator)
    return d_params
def fit_n_r_br_gr_al_alr_one_off(events_dict, events_dict_r, end_time, n_nodes_to, M, beta):
    # events_dict : (u,v):array_of_events
    if len(events_dict) == 0:  # handling empty block pair with no events
        return (1e-10, 0, 0, 0, 0, 0, 0, beta)
    mu_i = np.random.uniform(1e-6, 1e-2)
    alpha_n_i, alpha_r_i = np.random.uniform(0.1, 0.5, 2)
    alpha_br_i, alpha_gr_i, alpha_al_i, alpha_alr_i = np.random.uniform(1e-5, 0.1, 4)
    init_param = tuple([mu_i, alpha_n_i, alpha_r_i, alpha_br_i, alpha_gr_i, alpha_al_i, alpha_alr_i])  # <-- random initialization
    # init_param = (1e-2, 2e-2, 2e-2, 1e-2, 1e-2, 1e-2, 1e-2)
    Ris = cal_R_n_r_br_gr_al_alr_off(events_dict, events_dict_r, beta)
    diff_sum = cal_diff_sum(events_dict, end_time, beta)
    diff_sum_r = cal_diff_sum(events_dict_r, end_time, beta)
    res = minimize(NLL_n_r_br_gr_al_alr_one_off, init_param, method='L-BFGS-B',
                   bounds=((1e-7, None), (1e-7, None), (1e-7, None), (1e-7, None), (1e-7, None), (1e-7, None), (1e-7, None))
                   , jac=NLL_n_r_br_gr_al_alr_one_off_jac,
                   args=(events_dict, events_dict_r, end_time, n_nodes_to, M, beta, Ris, diff_sum, diff_sum_r), tol=1e-12)
    results = res.x
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr = results
    return (mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, beta)
#%% single beta (n, r, br, gr) model --- beta*alpha*exp(-beta*t)

""" diagonal block pairs"""
def cal_R_n_r_br_gr_dia(events_dict, beta):
    Ris = []
    for (u, v) in events_dict:
        # array of events of node pair (u,v)
        uv_events = events_dict[(u, v)]
        num_events_uv = len(uv_events)
        if num_events_uv == 0:
            Ris.append(np.array([0]))  # R=0 if node_pair (u,v) has no events
        else:
            # 4 columns (alpha_n, alpha_r, alpha_br, alpha_gr)
            Ri = np.zeros((num_events_uv, 4))
            intertimes = np.exp(-beta * (uv_events[1:] - uv_events[:-1]))
            for (x, y) in events_dict:
                if x == u or y == u:
                    prev_index = 0
                    # same node_pair events (alpha_n)
                    if (u, v) == (x, y):
                        for k in range(1, num_events_uv):
                            Ri[k, 0] = intertimes[k - 1] * (1 + Ri[k - 1, 0])
                    # reciprocal node_pair events (alpha_r)
                    elif (v, u) == (x, y):
                        for k in range(0, num_events_uv):
                            # return index below which t(x,y) < kth event of (u,v)
                            # if no events exists returns len(events(x,y))
                            index = bisect_left(events_dict[(x, y)], uv_events[k], lo=prev_index)
                            # no events found
                            if index == prev_index:
                                if k == 0: continue
                                Ri[k, 1] = intertimes[k - 1] * Ri[k - 1, 1]
                            else:
                                if k == 0:
                                    Ri[k, 1] = np.sum(np.exp(-beta * (uv_events[k] - events_dict[(x, y)][prev_index:index])))
                                else:
                                    Ri[k, 1] = intertimes[k - 1] * Ri[k - 1, 1] + np.sum(
                                        np.exp(-beta * (uv_events[k] - events_dict[(x, y)][prev_index:index])))
                                prev_index = index
                    # br node_pairs events (alpha_br)
                    elif u == x and v != y:
                        Ri_temp = np.zeros((num_events_uv, 1))
                        for k in range(0, num_events_uv):
                            # return index below which t(x,y) < kth event of (u,v)
                            # if no events exists returns len(events(x,y))
                            index = bisect_left(events_dict[(x, y)], uv_events[k], lo=prev_index)
                            # no events found
                            if index == prev_index:
                                if k == 0: continue
                                Ri_temp[k, 0] = intertimes[k - 1] * Ri_temp[k - 1, 0]
                            else:
                                if k == 0:
                                    Ri_temp[k, 0] = np.sum(np.exp(-beta * (uv_events[k] - events_dict[(x, y)][prev_index:index])))
                                else:
                                    Ri_temp[k, 0] = intertimes[k - 1] * Ri_temp[k - 1, 0] + np.sum(
                                        np.exp(-beta * (uv_events[k] - events_dict[(x, y)][prev_index:index])))
                                prev_index = index
                        Ri[:, 2] += Ri_temp[:, 0]
                    # gr node_pairs events (alpha_gr)
                    elif u == y and v != x:
                        Ri_temp = np.zeros((num_events_uv, 1))
                        for k in range(0, num_events_uv):
                            # return index below which t(x,y) < kth event of (u,v)
                            # if no events exists returns len(events(x,y))
                            index = bisect_left(events_dict[(x, y)], uv_events[k], lo=prev_index)
                            # no events found
                            if index == prev_index:
                                if k == 0: continue
                                Ri_temp[k, 0] = intertimes[k - 1] * Ri_temp[k - 1, 0]
                            else:
                                if k == 0:
                                    Ri_temp[k, 0] = np.sum(np.exp(-beta * (uv_events[k] - events_dict[(x, y)][prev_index:index])))
                                else:
                                    Ri_temp[k, 0] = intertimes[k - 1] * Ri_temp[k - 1, 0] + np.sum(
                                        np.exp(-beta * (uv_events[k] - events_dict[(x, y)][prev_index:index])))
                                prev_index = index
                        Ri[:, 3] += Ri_temp[:, 0]
            Ris.append(Ri)
    # return list of arrays - list size = #node_pairs_events_in block_pair
    return Ris

def LL_n_r_br_gr_one_dia_2(params, events_dict, end_time, n_nodes, M, Ris=None, diff_sum=None):
    # events_dict of node_pairs with events
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, beta = params
    ##first term
    first = -M * mu * end_time
    ##second term
    second = 0
    if diff_sum is None:
        events_array = list(events_dict.values())
        if events_array:  # block pair ab is not empty
            second = -((alpha_n + alpha_r + (alpha_br + alpha_gr) * (n_nodes - 2)) * np.sum(
                1 - np.exp(-beta * (end_time - np.concatenate(events_array)))))
    else:
        second = -((alpha_n + alpha_r + (alpha_br + alpha_gr) * (n_nodes - 2)) * diff_sum)
    ##third term
    if Ris is None:  # only calculate Ris if beta is not fixed (Ris is not passed)
        Ris = cal_R_n_r_br_gr_dia(events_dict, beta)  # list of M_np elements, each is (n_events_np,4) array
    # print("time after R()  = ", time.time())
    third = 0
    for i in range(len(Ris)):
        third += np.sum(np.log(mu + beta*(alpha_n * Ris[i][:, 0] + alpha_r * Ris[i][:, 1] +
                                          alpha_br * Ris[i][:, 2] + alpha_gr * Ris[i][:, 3])))
    # print("third r = ", third)
    log_likelihood_value = first + second + third
    return log_likelihood_value

def NLL_n_r_br_gr_one_dia_2(params, events_dict, end_time, n_nodes, M, beta, Ris, diff_sum):
    mu, alpha_n, alpha_r, alpha_br, alpha_gr = params
    params_fixed_b = mu, alpha_n, alpha_r, alpha_br, alpha_gr, beta
    return -LL_n_r_br_gr_one_dia_2(params_fixed_b, events_dict, end_time, n_nodes, M, Ris, diff_sum)

def NLL_n_r_br_gr_one_dia_jac_2(params, events_dict, end_time, n_nodes, M, beta, Ris, diff_sum):
    mu, alpha_n, alpha_r, alpha_br, alpha_gr = params
    d_mu = M * end_time
    d_alpha_n = diff_sum
    d_alpha_r = diff_sum
    d_alpha_br = (n_nodes - 2) * diff_sum
    d_alpha_gr = (n_nodes - 2) * diff_sum
    for i in range(len(Ris)):
        denominator = mu + beta * (alpha_n * Ris[i][:, 0] + alpha_r * Ris[i][:, 1] +
                                   alpha_br * Ris[i][:, 2] + alpha_gr * Ris[i][:, 3])
        d_mu -= np.sum(1 / (denominator))
        d_alpha_n -= np.sum(beta*Ris[i][:, 0] / (denominator))
        d_alpha_r -= np.sum(beta*Ris[i][:, 1] / (denominator))
        d_alpha_br -= np.sum(beta*Ris[i][:, 2] / (denominator))
        d_alpha_gr -= np.sum(beta*Ris[i][:, 3] / (denominator))
    return np.array([d_mu, d_alpha_n, d_alpha_r, d_alpha_br, d_alpha_gr])

def fit_n_r_br_gr_one_dia_2(events_dict, end_time, n_nodes, M, beta):
    # events_dict : (u,v):array_of_events
    if len(events_dict) == 0:  # handling empty block pair with no events
        return (1e-10, 0, 0, 0, 0, beta)
    mu_i = np.random.uniform(1e-6, 1e-2)
    alpha_n_i, alpha_r_i = np.random.uniform(0.1, 0.5, 2)
    alpha_br_i, alpha_gr_i = np.random.uniform(1e-5, 0.1, 2)
    init_param = tuple([mu_i, alpha_n_i, alpha_r_i, alpha_br_i, alpha_gr_i]) # <-- random initialization
    # init_param = (1e-2, 2e-2, 2e-2, 1e-2, 1e-2)
    Ris = cal_R_n_r_br_gr_dia(events_dict, beta)
    events_array = list(events_dict.values())
    diff_sum = np.sum(1 - np.exp(-beta * (end_time - np.concatenate(events_array))))
    res = minimize(NLL_n_r_br_gr_one_dia_2, init_param, method='L-BFGS-B',
                   bounds=((1e-7, None), (1e-7, None), (1e-7, None), (1e-7, None), (1e-7, None)),
                   jac=NLL_n_r_br_gr_one_dia_jac_2,
                   args=(events_dict, end_time, n_nodes, M, beta, Ris, diff_sum), tol=1e-12)
    results = res.x
    mu = results[0]
    alpha_n = results[1]
    alpha_r = results[2]
    alpha_br = results[3]
    alpha_gr = results[4]
    return (mu, alpha_n, alpha_r, alpha_br, alpha_gr, beta)

""" off-diagonal block pairs"""
def cal_R_n_r_br_gr_off(events_dict, events_dict_r, beta):
    Ris = []
    for (u, v) in events_dict:
        uv_events = events_dict[(u, v)]  # array of events of node pair (u,v)
        num_events_uv = len(uv_events)  # check if node pair (u,v) has no events
        if num_events_uv == 0:
            Ris.append(np.array([0]))  # R=0 if node_pair (u,v) has no events
        else:
            Ri = np.zeros((num_events_uv, 4))  # 4 columns for [alpha_n, alpha_r, alpha_br, alpha_gr]
            intertimes = np.exp(-beta * (uv_events[1:] - uv_events[:-1]))
            # loop through node pairs in block pair ab
            for (x, y) in events_dict:
                if x == u:
                    prev_index = 0
                    # same node_pair events (alpha_n)
                    if (u, v) == (x, y):
                        for k in range(1, num_events_uv):
                            Ri[k, 0] = intertimes[k - 1] * (1 + Ri[k - 1, 0])
                    # br node_pairs events (alpha_br)
                    else:
                        Ri_temp = np.zeros((num_events_uv, 1))
                        for k in range(0, num_events_uv):
                            # return index below which t(x,y) < kth event of (u,v)
                            # if no events exists returns len(events(x,y))
                            index = bisect_left(events_dict[(x, y)], uv_events[k], lo=prev_index)
                            if index == prev_index:  # no events found
                                if k == 0: continue
                                Ri_temp[k, 0] = intertimes[k - 1] * Ri_temp[k - 1, 0]
                            else:  # events less than kth event of (u,v) were found
                                if k == 0:
                                    Ri_temp[k, 0] = np.sum(np.exp(-beta * (uv_events[k] - events_dict[(x, y)][prev_index:index])))
                                else:
                                    Ri_temp[k, 0] = intertimes[k - 1] * Ri_temp[k - 1, 0] + np.sum(
                                        np.exp(-beta * (uv_events[k] - events_dict[(x, y)][prev_index:index])))
                                prev_index = index
                        Ri[:, 2] += Ri_temp[:, 0]  # third column for alpha_br
            # loop through node pairs in reciprocal block pair ba
            for (x, y) in events_dict_r:
                if y == u:
                    prev_index = 0
                    # reciprocal node_pair events (alpha_r)
                    if (v, u) == (x, y):
                        for k in range(0, num_events_uv):
                            # return index below which t(x,y) < kth event of (u,v)
                            # if no events exists returns len(events(x,y))
                            index = bisect_left(events_dict_r[(x, y)], uv_events[k], lo=prev_index)
                            if index == prev_index:  # no events found
                                if k == 0: continue
                                Ri[k, 1] = intertimes[k - 1] * Ri[k - 1, 1]
                            else:  # events less than kth event of (u,v) were found
                                if k == 0:
                                    Ri[k, 1] = np.sum(np.exp(-beta * (uv_events[k] - events_dict_r[(x, y)][prev_index:index])))
                                else:
                                    Ri[k, 1] = intertimes[k - 1] * Ri[k - 1, 1] + np.sum(
                                        np.exp(-beta * (uv_events[k] - events_dict_r[(x, y)][prev_index:index])))
                                prev_index = index
                    # gr node_pairs events (alpha_gr)
                    else:
                        Ri_temp = np.zeros((num_events_uv, 1))
                        for k in range(0, num_events_uv):
                            # return index below which t(x,y) < kth event of (u,v)
                            # if no events exists returns len(events(x,y))
                            index = bisect_left(events_dict_r[(x, y)], uv_events[k], lo=prev_index)
                            # no events found
                            if index == prev_index:
                                if k == 0: continue
                                Ri_temp[k, 0] = intertimes[k - 1] * Ri_temp[k - 1, 0]
                            else:
                                if k == 0:
                                    Ri_temp[k, 0] = np.sum(np.exp(-beta * (uv_events[k] - events_dict_r[(x, y)][prev_index:index])))
                                else:
                                    Ri_temp[k, 0] = intertimes[k - 1] * Ri_temp[k - 1, 0] + np.sum(
                                        np.exp(-beta * (uv_events[k] - events_dict_r[(x, y)][prev_index:index])))
                                prev_index = index
                        Ri[:, 3] += Ri_temp[:, 0]  # forth column for alpha_gr
            Ris.append(Ri)
    # return list of arrays - list size = #node_pairs_events_in block_pair
    return Ris

def LL_n_r_br_gr_one_off_2(params, events_dict, events_dict_r, end_time, n_nodes_to, M, Ris=None, diff_sum=None, diff_sum_r=None):
    # events_dict of node_pairs with events
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, beta = params
    ##first term
    first = -M * mu * end_time
    ##second term (block pair (ab) + block pair (ba))
    second = 0
    if diff_sum is None:
        events_array = list(events_dict.values())
        if events_array:  # block pair ab is not empty
            second -= ((alpha_n + alpha_br * (n_nodes_to - 1)) *
                       np.sum(1 - np.exp(-beta * (end_time - np.concatenate(events_array)))))
        events_array_r = list(events_dict_r.values())
        if events_dict_r:  # block pair ba is not empty
            second -= ((alpha_r + alpha_gr * (n_nodes_to - 1)) *
                       np.sum(1 - np.exp(-beta * (end_time - np.concatenate(events_array_r)))))
    else:
        second -= ((alpha_n + alpha_br * (n_nodes_to - 1)) * diff_sum)
        second -= ((alpha_r + alpha_gr * (n_nodes_to - 1)) * diff_sum_r)
    # print("second r br = ", second)
    ##third term
    if Ris is None:  # only calculate Ris if beta is a variable
        Ris = cal_R_n_r_br_gr_off(events_dict, events_dict_r, beta)  # list of M_np elements, each is (n_events_np,4) array
    third = 0
    for i in range(len(Ris)):
        third += np.sum(np.log(mu + beta * (alpha_n * Ris[i][:, 0] + alpha_r * Ris[i][:, 1]
                                            + alpha_br * Ris[i][:, 2] + alpha_gr * Ris[i][:, 3])))
    # print("third r = ", third)
    log_likelihood_value = first + second + third
    return log_likelihood_value

def NLL_n_r_br_gr_one_off_2(params, d_ab, d_ba, end_time, n_nodes_to, M, beta, Ris, diff_sum, diff_sum_r):
    mu, alpha_n, alpha_r, alpha_br, alpha_gr = params
    params_fixed_b = mu, alpha_n, alpha_r, alpha_br, alpha_gr, beta
    return -LL_n_r_br_gr_one_off_2(params_fixed_b, d_ab, d_ba, end_time, n_nodes_to, M, Ris, diff_sum, diff_sum_r)

def NLL_n_r_br_gr_one_off_jac_2(params, events_dict, events_dict_r, end_time, n_nodes_to, M, beta, Ris, diff_sum, diff_sum_r):
    mu, alpha_n, alpha_r, alpha_br, alpha_gr = params
    d_mu = M * end_time
    d_alpha_n = diff_sum
    d_alpha_r = diff_sum_r
    d_alpha_br = (n_nodes_to - 1) * diff_sum
    d_alpha_gr = (n_nodes_to - 1) * diff_sum_r
    for i in range(len(Ris)):
        denominator = mu + beta * (alpha_n * Ris[i][:, 0] + alpha_r * Ris[i][:, 1]
                                   + alpha_br * Ris[i][:, 2] + alpha_gr * Ris[i][:, 3])
        d_mu -= np.sum(1 / denominator)
        d_alpha_n -= np.sum(beta *Ris[i][:, 0] / (denominator))
        d_alpha_r -= np.sum(beta *Ris[i][:, 1] / (denominator))
        d_alpha_br -= np.sum(beta *Ris[i][:, 2] / (denominator))
        d_alpha_gr -= np.sum(beta *Ris[i][:, 3] / (denominator))
    return np.array([d_mu, d_alpha_n, d_alpha_r, d_alpha_br, d_alpha_gr])

def fit_n_r_br_gr_one_off_2(events_dict, events_dict_r, end_time, n_nodes_to, M, beta):
    # events_dict : (u,v):array_of_events
    if len(events_dict) == 0:  # handling empty block pair with no events
        return (1e-10, 0, 0, 0, 0, beta)
    mu_i = np.random.uniform(1e-6, 1e-2)
    alpha_n_i, alpha_r_i = np.random.uniform(0.1, 0.5, 2)
    alpha_br_i, alpha_gr_i = np.random.uniform(1e-5, 0.1, 2)
    init_param = tuple([mu_i, alpha_n_i, alpha_r_i, alpha_br_i, alpha_gr_i]) # <-- random initialization
    # init_param = (1e-2, 2e-2, 2e-2, 1e-2, 1e-2)
    Ris = cal_R_n_r_br_gr_off(events_dict, events_dict_r, beta)
    events_array = list(events_dict.values())
    events_array_r = list(events_dict_r.values())
    diff_sum = np.sum(1 - np.exp(-beta * (end_time - np.concatenate(events_array))))
    if len(events_array_r) == 0:
        diff_sum_r = 0
    else:
        diff_sum_r = np.sum(1 - np.exp(-beta * (end_time - np.concatenate(events_array_r))))
    res = minimize(NLL_n_r_br_gr_one_off_2, init_param, method='L-BFGS-B',
                   bounds=((1e-7, None), (1e-7, None), (1e-7, None), (1e-7, None), (1e-7, None)), jac=NLL_n_r_br_gr_one_off_jac_2,
                   args=(events_dict, events_dict_r, end_time, n_nodes_to, M, beta, Ris, diff_sum, diff_sum_r), tol=1e-12)
    results = res.x
    mu = results[0]
    alpha_n = results[1]
    alpha_r = results[2]
    alpha_br = results[3]
    alpha_gr = results[4]
    return (mu, alpha_n, alpha_r, alpha_br, alpha_gr, beta)


#%% single beta (n, r) model --- beta*alpha*exp(-beta*t)

"""diagonal"""
def cal_R_n_r_dia(events_dict, beta):
    Ris = []
    for (u, v) in events_dict:
        # array of events of node pair (u,v)
        uv_events = events_dict[(u, v)]
        num_events_uv = len(uv_events)
        if num_events_uv == 0:
            Ris.append(np.array([0]))  # R=0 if node_pair (u,v) has no events
        else:
            # 2 columns (alpha_n, alpha_r)
            Ri = np.zeros((num_events_uv, 2))
            intertimes = np.exp(-beta * (uv_events[1:] - uv_events[:-1]))
            for (x, y) in events_dict:
                if x == u or y == u:
                    prev_index = 0
                    # same node_pair events (alpha_n)
                    if (u, v) == (x, y):
                        for k in range(1, num_events_uv):
                            Ri[k, 0] = intertimes[k - 1] * (1 + Ri[k - 1, 0])
                    # reciprocal node_pair events (alpha_r)
                    elif (v, u) == (x, y):
                        for k in range(0, num_events_uv):
                            # return index below which t(x,y) < kth event of (u,v)
                            # if no events exists returns len(events(x,y))
                            index = bisect_left(events_dict[(x, y)], uv_events[k], lo=prev_index)
                            # no events found
                            if index == prev_index:
                                if k == 0: continue
                                Ri[k, 1] = intertimes[k - 1] * Ri[k - 1, 1]
                            else:
                                if k == 0:
                                    Ri[k, 1] = np.sum(np.exp(-beta * (uv_events[k] - events_dict[(x, y)][prev_index:index])))
                                else:
                                    Ri[k, 1] = intertimes[k - 1] * Ri[k - 1, 1] + np.sum(
                                        np.exp(-beta * (uv_events[k] - events_dict[(x, y)][prev_index:index])))
                                prev_index = index
            Ris.append(Ri)
    # return list of arrays - list size = #node_pairs_events_in block_pair
    return Ris
def LL_n_r_one_dia_2(params, events_dict, end_time, n_nodes, M, Ris=None, diff_sum=None):
    # events_dict of node_pairs with events
    mu, alpha_n, alpha_r, beta = params
    ##first term
    first = - M * mu * end_time
    ##second term
    second = 0
    events_array = list(events_dict.values())
    if diff_sum is None:
        if len(events_array) != 0:  # block pair ab is not empty
            second = - ((alpha_n + alpha_r) * np.sum(1 - np.exp(-beta * (end_time - np.concatenate(events_array)))))
    else:
        second = - ((alpha_n + alpha_r) * diff_sum)
    ##third term
    if Ris is None:
        Ris = cal_R_n_r_dia(events_dict, beta)  # list of M_np elements, each is (n_events_np,2) array
    third = 0
    for i in range(len(Ris)):
        third += np.sum(np.log(mu + beta*(alpha_n * Ris[i][:, 0] + alpha_r * Ris[i][:, 1])))
    log_likelihood_value = first + second + third
    return log_likelihood_value
def NLL_n_r_one_dia_2(params, events_dict, end_time, n_nodes, M, beta, Ris, diff_sum):
    mu, alpha_n, alpha_r = params
    params_fixed_b = mu, alpha_n, alpha_r, beta
    return - LL_n_r_one_dia_2(params_fixed_b, events_dict, end_time, n_nodes, M, Ris, diff_sum)
def NLL_n_r_one_dia_jac_2(params, events_dict, end_time, n_nodes, M, beta, Ris, diff_sum):
    mu, alpha_n, alpha_r = params
    d_mu = M * end_time
    d_alpha_n = diff_sum
    d_alpha_r = diff_sum
    for i in range(len(Ris)):
        denominator = mu + beta *(alpha_n * Ris[i][:, 0] + alpha_r * Ris[i][:, 1])
        d_mu -= np.sum(1 / (denominator))
        d_alpha_n -= np.sum(beta * Ris[i][:, 0] / (denominator))
        d_alpha_r -= np.sum(beta * Ris[i][:, 1] / (denominator))
    return np.array([d_mu, d_alpha_n, d_alpha_r])
def fit_n_r_one_dia_2(events_dict, end_time, n_nodes, M, beta):
    # events_dict : (u,v):array_of_events
    if len(events_dict) == 0:  # handling empty block pair with no events
        return (1e-10, 0, 0, beta)
    # parameters initialization
    mu_i = np.random.uniform(1e-6, 1e-2)
    alpha_n_i, alpha_r_i = np.random.uniform(0.1, 0.5, 2)
    init_param = tuple([mu_i, alpha_n_i, alpha_r_i]) # <-- random initialization
    # init_param = (1e-2, 2e-2, 2e-2)
    Ris = cal_R_n_r_dia(events_dict, beta)
    events_array = list(events_dict.values())
    diff_sum = np.sum(1 - np.exp(-beta * (end_time - np.concatenate(events_array))))
    res = minimize(NLL_n_r_one_dia_2, init_param, method='L-BFGS-B', bounds=((1e-7, None), (1e-7, None), (1e-7, None)),
                   jac=NLL_n_r_one_dia_jac_2, args=(events_dict, end_time, n_nodes, M, beta, Ris, diff_sum), tol=1e-12)
    results = res.x
    mu = results[0]
    alpha_n = results[1]
    alpha_r = results[2]
    return (mu, alpha_n, alpha_r, beta)

"""off-diagonal"""
def cal_R_n_r_off(events_dict, events_dict_r, beta):
    Ris = []
    for (u, v) in events_dict:
        uv_events = events_dict[(u, v)]  # array of events of node pair (u,v)
        num_events_uv = len(uv_events)  # check if node pair (u,v) has no events
        if num_events_uv == 0:
            Ris.append(np.array([0]))  # R=0 if node_pair (u,v) has no events
        else:
            Ri = np.zeros((num_events_uv, 2))  # 2 columns for [alpha_n, alpha_r]
            intertimes = np.exp(-beta * (uv_events[1:] - uv_events[:-1]))
            # loop through node pairs in block pair ab
            for (x, y) in events_dict:
                # same node_pair events (alpha_n)
                if (u, v) == (x, y):
                    for k in range(1, num_events_uv):
                        Ri[k, 0] = intertimes[k - 1] * (1 + Ri[k - 1, 0])
            # loop through node pairs in reciprocal block pair ba
            for (x, y) in events_dict_r:
                prev_index = 0
                # reciprocal node_pair events (alpha_r)
                if (v, u) == (x, y):
                    for k in range(0, num_events_uv):
                        # return index below which t(x,y) < kth event of (u,v)
                        # if no events exists returns len(events(x,y))
                        index = bisect_left(events_dict_r[(x, y)], uv_events[k], lo=prev_index)
                        if index == prev_index:  # no events found
                            if k == 0: continue
                            Ri[k, 1] = intertimes[k - 1] * Ri[k - 1, 1]
                        else:  # events less than kth event of (u,v) were found
                            if k == 0:
                                Ri[k, 1] = np.sum(np.exp(-beta * (uv_events[k] - events_dict_r[(x, y)][prev_index:index])))
                            else:
                                Ri[k, 1] = intertimes[k - 1] * Ri[k - 1, 1] + np.sum(
                                    np.exp(-beta * (uv_events[k] - events_dict_r[(x, y)][prev_index:index])))
                            prev_index = index
            Ris.append(Ri)
    # return list of arrays - list size = #node_pairs_events_in block_pair
    return Ris

def LL_n_r_one_off_2(params, events_dict, events_dict_r, end_time, n_nodes_to, M, Ris=None, diff_sum=None, diff_sum_r=None):
    # events_dict of node_pairs with events
    mu, alpha_n, alpha_r, beta = params
    ##first term
    first = - M * mu * end_time

    ##second term (block pair (ab) + reciprocal block pair (ba))
    second = 0
    if diff_sum is None:
        events_array = list(events_dict.values())
        if len(events_array) != 0:  # block pair ab is not empty
            second -= (alpha_n * np.sum(1 - np.exp(-beta * (end_time - np.concatenate(events_array)))))
        events_array_r = list(events_dict_r.values())
        if len(events_dict_r) != 0:  # block pair ba is not empty
            second -= (alpha_r * np.sum(1 - np.exp(-beta * (end_time - np.concatenate(events_array_r)))))
    else:
        second -= (alpha_n * diff_sum)
        second -= (alpha_r * diff_sum_r)

    ##third term
    if Ris is None:
        Ris = cal_R_n_r_off(events_dict, events_dict_r, beta)  # list of M_np elements, each is (n_events_np,4) array
    third = 0
    for i in range(len(Ris)):
        third += np.sum(np.log(mu + beta * (alpha_n * Ris[i][:, 0] + alpha_r * Ris[i][:, 1])))
    log_likelihood_value = first + second + third
    # print(f"ll = {first} + {second} + {third} = {first + second + third}")
    return log_likelihood_value
def NLL_n_r_one_off_2(params, events_dict, events_dict_r, end_time, n_nodes_to, M, beta, Ris, diff_sum, diff_sum_r):
    mu, alpha_n, alpha_r = params
    params_fixed_b = mu, alpha_n, alpha_r, beta
    return -LL_n_r_one_off_2(params_fixed_b, events_dict, events_dict_r, end_time, n_nodes_to, M, Ris, diff_sum, diff_sum_r)
def NLL_n_r_one_off_jac_2(params, events_dict, events_dict_r, end_time, n_nodes_to, M, beta, Ris, diff_sum, diff_sum_r):
    mu, alpha_n, alpha_r = params
    d_mu = M * end_time
    d_alpha_n = diff_sum
    d_alpha_r = diff_sum_r
    for i in range(len(Ris)):
        denominator = mu + beta * (alpha_n * Ris[i][:, 0] + alpha_r * Ris[i][:, 1])
        d_mu -= np.sum(1 / (denominator))
        d_alpha_n -= np.sum(beta * Ris[i][:, 0] / (denominator))
        d_alpha_r -= np.sum(beta * Ris[i][:, 1] / (denominator))
    return np.array([d_mu, d_alpha_n, d_alpha_r])
def fit_n_r_one_off_2(events_dict, events_dict_r, end_time, n_nodes_to, M, beta):
    # events_dict : (u,v):array_of_events
    if len(events_dict) == 0:  # handling empty block pair with no events
        return (1e-10, 0, 0, beta)
    mu_i = np.random.uniform(1e-6, 1e-2)
    alpha_n_i, alpha_r_i = np.random.uniform(0.1, 0.5, 2)
    init_param = tuple([mu_i, alpha_n_i, alpha_r_i]) # <-- random initialization
    # init_param = (1e-2, 2e-2, 2e-2) # <-- fixed initialization
    Ris = cal_R_n_r_off(events_dict, events_dict_r, beta)
    events_array = list(events_dict.values())
    events_array_r = list(events_dict_r.values())
    diff_sum = np.sum(1 - np.exp(-beta * (end_time - np.concatenate(events_array))))
    if len(events_array_r) == 0:
        diff_sum_r = 0
    else:
        diff_sum_r = np.sum(1 - np.exp(-beta * (end_time - np.concatenate(events_array_r))))
    res = minimize(NLL_n_r_one_off_2, init_param, method='L-BFGS-B', bounds=((1e-7, None), (1e-7, None), (1e-7, None)),
                   jac=NLL_n_r_one_off_jac_2, args=(events_dict, events_dict_r, end_time, n_nodes_to, M, beta, Ris, diff_sum, diff_sum_r),
                   tol=1e-12)
    results = res.x
    mu = results[0]
    alpha_n = results[1]
    alpha_r = results[2]
    return (mu, alpha_n, alpha_r, beta)

"""off-diagonal restricted rho"""
def LL_n_r_one_off_rho(params, events_ab, events_ba, end_time, N_a, N_b,
                       Ris_ab=None, Ris_ba=None, diff_sum_ab=None, diff_sum_ba=None):
    # events_dict of node_pairs with events
    mu_ab, mu_ba, alpha_ab, alpha_ba, rho, beta = params
    ##first term
    first = - N_a * N_b * (mu_ab + mu_ba) * end_time
    ##second term (block pair (ab) + reciprocal block pair (ba))
    second = 0
    if diff_sum_ab is None:
        diff_sum_ab, diff_sum_ba = 0, 0
        events_array_ab = list(events_ab.values())
        if len(events_array_ab) != 0:  # block pair ab is not empty
            diff_sum_ab = np.sum(1 - np.exp(-beta * (end_time - np.concatenate(events_array_ab))))
        events_array_ba = list(events_ba.values())
        if len(events_array_ba) != 0:  # block pair ba is not empty
            diff_sum_ba = np.sum(1 - np.exp(-beta * (end_time - np.concatenate(events_array_ba))))
    second -= (alpha_ab + rho * alpha_ba) * diff_sum_ab
    second -= (alpha_ba + rho * alpha_ab) * diff_sum_ba

    ##third term
    if Ris_ab is None:
        Ris_ab = cal_R_n_r_off(events_ab, events_ba, beta)  # list of np with events in (a, b), each is (n_events_np,2) array
        Ris_ba = cal_R_n_r_off(events_ba, events_ab, beta)  # list of np with events in (b, a), each is (n_events_np,2) array
    third = 0
    for i in range(len(Ris_ab)):
        third += np.sum(np.log(mu_ab + beta * (alpha_ab * Ris_ab[i][:, 0] + rho * alpha_ab * Ris_ab[i][:, 1])))
    for i in range(len(Ris_ba)):
        third += np.sum(np.log(mu_ba + beta * (alpha_ba * Ris_ba[i][:, 0] + rho * alpha_ba * Ris_ba[i][:, 1])))
    log_likelihood_value = first + second + third
    # print(f"ll_rho = {first} + {second} + {third} = {first + second + third}")
    return log_likelihood_value
def NLL_n_r_one_off_rho(params, events_ab, events_ba, end_time, N_a, N_b, beta,
                        Ris_ab, Ris_ba, diff_sum_ab, diff_sum_ba):
    mu_ab, mu_ba, alpha_ab, alpha_ba, rho = params
    params_beta = mu_ab, mu_ba, alpha_ab, alpha_ba, rho, beta
    return -LL_n_r_one_off_rho(params_beta, events_ab, events_ba, end_time, N_a, N_b,
                               Ris_ab, Ris_ba, diff_sum_ab, diff_sum_ba)
def NLL_n_r_one_off_jac_rho(params, events_ab, events_ba, end_time, N_a, N_b, beta,
                        Ris_ab, Ris_ba, diff_sum_ab, diff_sum_ba):
    mu_ab, mu_ba, alpha_ab, alpha_ba, rho = params
    d_mu_ab = N_a * N_b * end_time
    d_mu_ba = N_a * N_b * end_time
    d_alpha_ab = diff_sum_ab + rho * diff_sum_ba
    d_alpha_ba = diff_sum_ba + rho * diff_sum_ab
    d_rho = alpha_ba * diff_sum_ab + alpha_ab * diff_sum_ba
    # block pair (a, b)
    for i in range(len(Ris_ab)):
        denominator = mu_ab + beta * (alpha_ab * Ris_ab[i][:, 0] + rho* alpha_ab * Ris_ab[i][:, 1])
        d_mu_ab -= np.sum(1 / (denominator))
        d_alpha_ab -= np.sum(beta * (Ris_ab[i][:, 0] + rho * Ris_ab[i][:, 1]) / (denominator))
        d_rho -= np.sum(beta * alpha_ab * Ris_ab[i][:, 1] / (denominator))
    # block pair (b, a)
    for i in range(len(Ris_ba)):
        denominator = mu_ba + beta * (alpha_ba * Ris_ba[i][:, 0] + rho * alpha_ba * Ris_ba[i][:, 1])
        d_mu_ba -= np.sum(1 / (denominator))
        d_alpha_ba -= np.sum(beta * (Ris_ba[i][:, 0] + rho * Ris_ba[i][:, 1]) / (denominator))
        d_rho -= np.sum(beta * alpha_ba * Ris_ba[i][:, 1] / (denominator))
    return np.array([d_mu_ab, d_mu_ba, d_alpha_ab, d_alpha_ba, d_rho])
def fit_n_r_one_off_rho(events_ab, events_ba, end_time, N_a, N_b, beta):
    # if both bp's have no events
    if len(events_ab) == 0 and len(events_ba)==0:
        return (1e-10, 1e-10, 0, 0, 0, beta)    #(mu_ab, mu_ba, alpha_ab, alpha_ba, rho, beta)
    # if only bp(b,a) has events
    elif len(events_ab) == 0:
        mu_ba, alpha_ba, rho_alpha_ba, beta = fit_n_r_one_off_2(events_ba, events_ab, end_time, N_a, N_a*N_b, beta)
        rho = rho_alpha_ba/ alpha_ba
        return (1e-10, mu_ba, 0, alpha_ba, rho, beta)
    # if only bp(a, b) has events
    elif len(events_ba) == 0:
        mu_ab, alpha_ab, rho_alpha_ab, beta = fit_n_r_one_off_2(events_ab, events_ba, end_time, N_b, N_a*N_b, beta)
        rho = rho_alpha_ab / alpha_ab
        return (mu_ab, 1e-10, alpha_ab, 0, rho, beta)

    # initialize parameter randomly
    mu_ab_i, mu_ba_i = np.random.uniform(1e-6, 1e-2, 2)
    alpha_ab_i, alpha_ba_i = np.random.uniform(0.1, 0.5, 2)
    rho_i = np.random.uniform(0.1, 0.5)
    param_i = tuple([mu_ab_i, mu_ba_i, alpha_ab_i, alpha_ba_i, rho_i]) # <-- random initialization

    Ris_ab = cal_R_n_r_off(events_ab, events_ba, beta)  # list of np with events in (a, b), each is (n_events_np,2) array
    Ris_ba = cal_R_n_r_off(events_ba, events_ab, beta)  # list of np with events in (b, a), each is (n_events_np,2) array

    diff_sum_ab, diff_sum_ba = 0, 0
    events_array_ab = list(events_ab.values())
    if len(events_array_ab) != 0:  # block pair ab is not empty
        diff_sum_ab = np.sum(1 - np.exp(-beta * (end_time - np.concatenate(events_array_ab))))
    events_array_ba = list(events_ba.values())
    if len(events_array_ba) != 0:  # block pair ba is not empty
        diff_sum_ba = np.sum(1 - np.exp(-beta * (end_time - np.concatenate(events_array_ba))))

    res = minimize(NLL_n_r_one_off_rho, param_i, method='L-BFGS-B', bounds=tuple([(1e-7, None)] * 4 + [(0, None)]),
                   jac=NLL_n_r_one_off_jac_rho,
                   args=(events_ab, events_ba, end_time, N_a, N_b, beta, Ris_ab, Ris_ba, diff_sum_ab, diff_sum_ba), tol=1e-12)
    results = res.x
    mu_ab, mu_ba, alpha_ab, alpha_ba, rho = results
    return (mu_ab, mu_ba, alpha_ab, alpha_ba, rho, beta)
#%% sum of kernles (n, r, br, gr, al, alr) model --- beta*alpha*exp(-beta*t)
""" helper functions"""
def cal_diff_sums_Q(events_dict, end_time, betas):
    Q = len(betas)
    T_diff_sums = np.zeros(Q, )
    if len(events_dict) == 0:
        return T_diff_sums
    events_array = list(events_dict.values())
    T_diff = end_time - np.concatenate(events_array)
    for q in range(Q):
        T_diff_sums[q] = np.sum(1 - np.exp(-betas[q] * T_diff))
    return T_diff_sums

def cal_R_6_alpha_kernel_sum_dia(events_dict, betas):
    # betas: array of fixed decays
    Q = len(betas)
    Ris = []
    for (u, v) in events_dict:
        # array of events of node pair (u,v)
        uv_events = events_dict[(u, v)]
        num_events_uv = len(uv_events)
        if num_events_uv == 0:
            Ris.append(np.array([0]))  # R=0 if node_pair (u,v) has no events
        else:
            # 6*Q columns (alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr)*Q
            Ri = np.zeros((num_events_uv, 6 * Q))
            uv_intertimes = (uv_events[1:] - uv_events[:-1])
            # (#_uv_events-1, Q) array
            e_intertimes_Q = np.zeros((len(uv_intertimes), Q))
            for q in range(Q):
                e_intertimes_Q[:, q] = np.exp(-betas[q] * uv_intertimes)
            for (x, y) in events_dict:
                # same node_pair events (alpha_n)
                if (u, v) == (x, y):
                    for k in range(1, num_events_uv):
                        for q in range(Q):
                            Ri[k, 0 + q * 6] = e_intertimes_Q[k - 1, q] * (1 + Ri[k - 1, 0 + q * 6])
                # reciprocal node_pair events (alpha_r)
                elif (v, u) == (x, y):
                    Ri_temp = get_Ri_temp_Q(uv_events, e_intertimes_Q, events_dict[(x, y)], betas)
                    for q in range(Q):
                        Ri[:, 1 + q * 6] = Ri_temp[:, q]
                # br node_pairs events (alpha_br)
                elif u == x and v != y:
                    Ri_temp = get_Ri_temp_Q(uv_events, e_intertimes_Q, events_dict[(x, y)], betas)
                    for q in range(Q):
                        Ri[:, 2 + q * 6] += Ri_temp[:, q]
                # gr node_pairs events (alpha_gr)
                elif u == y and v != x:
                    Ri_temp = get_Ri_temp_Q(uv_events, e_intertimes_Q, events_dict[(x, y)], betas)
                    for q in range(Q):
                        Ri[:, 3 + q * 6] += Ri_temp[:, q]
                # alliance np (alpha_al)
                elif v == y and u != x:
                    Ri_temp = get_Ri_temp_Q(uv_events, e_intertimes_Q, events_dict[(x, y)], betas)
                    for q in range(Q):
                        Ri[:, 4 + q * 6] += Ri_temp[:, q]
                # alliance reciprocal np (alpha_alr)
                elif v == x and u != y:
                    Ri_temp = get_Ri_temp_Q(uv_events, e_intertimes_Q, events_dict[(x, y)], betas)
                    for q in range(Q):
                        Ri[:, 5 + q * 6] += Ri_temp[:, q]
            Ris.append(Ri)
    # return list of arrays - list size = #node_pairs_events_in block_pair
    return Ris
def cal_R_6_alpha_kernel_sum_off(events_dict, events_dict_r, betas):
    # betas: array of fixed decays
    Q = len(betas)
    Ris = []
    for (u, v) in events_dict:
        # array of events of node pair (u,v)
        uv_events = events_dict[(u, v)]
        num_events_uv = len(uv_events)
        if num_events_uv == 0:
            Ris.append(np.array([0]))  # R=0 if node_pair (u,v) has no events
        else:
            # 6*Q columns (alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr)*Q
            Ri = np.zeros((num_events_uv, 6 * Q))
            uv_intertimes = (uv_events[1:] - uv_events[:-1])
            # (#_uv_events-1, Q) array
            e_intertimes_Q = np.zeros((len(uv_intertimes), Q))
            for q in range(Q):
                e_intertimes_Q[:, q] = np.exp(-betas[q] * uv_intertimes)

            # loop through node pairs in block pair ab
            for (x, y) in events_dict:
                # same node_pair events (alpha_n)
                if (u, v) == (x, y):
                    for k in range(1, num_events_uv):
                        for q in range(Q):
                            Ri[k, 0 + q * 6] = e_intertimes_Q[k - 1, q] * (1 + Ri[k - 1, 0 + q * 6])
                # br node_pairs events (alpha_br)
                elif u == x and v != y:
                    Ri_temp = get_Ri_temp_Q(uv_events, e_intertimes_Q, events_dict[(x, y)], betas)
                    for q in range(Q):
                        Ri[:, 2 + q * 6] += Ri_temp[:, q]
                # alliance np (alpha_al)
                elif v == y and u != x:
                    Ri_temp = get_Ri_temp_Q(uv_events, e_intertimes_Q, events_dict[(x, y)], betas)
                    for q in range(Q):
                        Ri[:, 4 + q * 6] += Ri_temp[:, q]
            # loop through node pairs in reciprocal block pair ba
            for (x, y) in events_dict_r:
                # reciprocal node_pair events (alpha_r)
                if (v, u) == (x, y):
                    Ri_temp = get_Ri_temp_Q(uv_events, e_intertimes_Q, events_dict_r[(x, y)], betas)
                    for q in range(Q):
                        Ri[:, 1 + q * 6] = Ri_temp[:, q]
                # gr node_pairs events (alpha_gr)
                elif u == y and v != x:
                    Ri_temp = get_Ri_temp_Q(uv_events, e_intertimes_Q, events_dict_r[(x, y)], betas)
                    for q in range(Q):
                        Ri[:, 3 + q * 6] += Ri_temp[:, q]
                # alliance reciprocal np (alpha_alr)
                elif v == x and u != y:
                    Ri_temp = get_Ri_temp_Q(uv_events, e_intertimes_Q, events_dict_r[(x, y)], betas)
                    for q in range(Q):
                        Ri[:, 5 + q * 6] += Ri_temp[:, q]
            Ris.append(Ri)
    # return list of arrays - list size = #node_pairs_events_in block_pair
    return Ris

""" diagonal block pair"""
def LL_6_alpha_kernel_sum_dia(params, events_dict, end_time, n_a, M, T_diff_sums=None, Ris=None):
    # events_dict of node_pairs with events
    # C: scaling parameters - same length as betas
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, C, betas = params
    Q = len(betas)
    ##first term
    first = -M * mu * end_time
    ### block pair has no events (empty)
    if len(events_dict) == 0:
        return first
    ##second term
    if T_diff_sums is None:
        T_diff_sums = cal_diff_sums_Q(events_dict, end_time, betas)
    second = -(alpha_n + alpha_r + (alpha_br + alpha_gr + alpha_al + alpha_alr) * (n_a - 2)) * C @ T_diff_sums
    ##third term
    if Ris is None:
        Ris = cal_R_6_alpha_kernel_sum_dia(events_dict, betas)
    third = 0
    for i in range(len(Ris)):
        col_sum = np.zeros(Ris[i].shape[0])
        for q in range(Q):
            col_sum[:] += C[q]*betas[q] * (alpha_n * Ris[i][:, 0 + q * 6]  + alpha_r * Ris[i][:, 1 + q * 6] +
                                           alpha_br * Ris[i][:, 2 + q * 6] + alpha_gr * Ris[i][:,3 + q * 6] +
                                           alpha_al * Ris[i][:, 4 + q * 6] + alpha_alr * Ris[i][:,5 + q * 6])
        col_sum += mu
        third += np.sum(np.log(col_sum))
    log_likelihood_value = first + second + third
    # print("ll: ", first, second, third)
    return log_likelihood_value
def NLL_6_alpha_kernel_sum_dia(p, betas, events_dict, end_time, n_nodes, M, T_diff_sums, Ris):
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr = p[:7]
    C = np.array(p[7:])
    # scaling step - constaint C sums to 1
    C_sum = np.sum(C)
    if (C_sum != 0):
        C = C / C_sum
        alpha_n = alpha_n * C_sum
        alpha_r = alpha_r * C_sum
        alpha_br = alpha_br * C_sum
        alpha_gr = alpha_gr * C_sum
        alpha_al = alpha_al * C_sum
        alpha_alr = alpha_alr * C_sum
    params = mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, C, betas
    return -LL_6_alpha_kernel_sum_dia(params, events_dict, end_time, n_nodes, M, T_diff_sums, Ris)
def NLL_6_alpha_kernel_sum_dia_jac(p, betas, events_dict, end_time, n_nodes, M, T_diff_sums, Ris):
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr = p[:7]
    C = np.array(p[7:])
    Q = len(C)
    # derivatives of second term
    d_mu = M * end_time
    d_alphas = np.zeros(6)
    d_alphas[:2] = C @ T_diff_sums
    d_alphas[2:] = (n_nodes - 2) * C @ T_diff_sums
    d_C = (alpha_n + alpha_r + (n_nodes - 2) * (alpha_br + alpha_gr + alpha_al + alpha_alr)) * T_diff_sums
    # derivatives of third term
    for i in range(len(Ris)):
        denominator = np.zeros(Ris[i].shape[0])
        # one column for each alpha_j
        numerator_alphas = np.zeros((Ris[i].shape[0], 6))
        numerator_C = []
        for q in range(Q):
            numerator_C.append(betas[q]* (alpha_n * Ris[i][:, 0 + q * 6] + alpha_r * Ris[i][:, 1 + q * 6] +
                                          alpha_br * Ris[i][:, 2 + q * 6] + alpha_gr * Ris[i][:,3 + q * 6] +
                                          alpha_al * Ris[i][:, 4 + q * 6] + alpha_alr * Ris[i][:,5 + q * 6]))
            denominator += C[q] * numerator_C[q]
            for j in range(6):
                numerator_alphas[:, j] += C[q] * betas[q] * Ris[i][:, j + q * 6]
        denominator += mu
        d_mu -= np.sum(1 / denominator)
        for a in range(6):
            d_alphas[a] -= np.sum(numerator_alphas[:, a] / denominator)
        for q in range(Q):
            d_C[q] -= np.sum(numerator_C[q] / denominator)
    return np.hstack((d_mu, d_alphas, d_C))

def fit_6_alpha_kernel_sum_dia(events_dict, end_time, n_nodes, M, betas):
    Q = len(betas)
    # events_dict : (u,v):array_of_events
    if len(events_dict) == 0:  # handling empty block pair with no events
        # (mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, C, betas)
        C = np.zeros(Q)
        return (1e-10, 0, 0, 0, 0, 0, 0, C, betas)

    # calculate fixed terms in log-likelihood
    Ris = cal_R_6_alpha_kernel_sum_dia(events_dict, betas)
    T_diff_sums = cal_diff_sums_Q(events_dict, end_time, betas)

    # initialize parameters (mu, alpha_n, alpha_r, alpha_br, alpha_gr, c1, ..., cQ)
    mu_i = np.random.uniform(1e-6, 1e-2)
    alpha_n_i, alpha_r_i = np.random.uniform(0.1, 0.5, 2)
    alpha_br_i, alpha_gr_i, alpha_al_i, alpha_alr_i = np.random.uniform(1e-5, 0.1, 4)
    mu_alpha_init = [mu_i, alpha_n_i, alpha_r_i, alpha_br_i, alpha_gr_i, alpha_al_i, alpha_alr_i] # <-- random initialization
    # mu_alpha_init = [1e-2, 2e-2, 2e-2, 1e-2, 1e-2, 1e-2, 1e-2]  # <-- fixed initialization
    C = [1 / Q] * Q  # <-- fixed initialization
    init_param = tuple(mu_alpha_init + C)
    # define bounds
    mu_alpha_bo = [(1e-7, None)] * 7
    C_bo = [(0, 1)] * Q
    bounds = tuple(mu_alpha_bo + C_bo)

    # minimize function
    # to print optimization details , options={'disp': True}
    res = minimize(NLL_6_alpha_kernel_sum_dia, init_param, method='L-BFGS-B', bounds=bounds, jac=NLL_6_alpha_kernel_sum_dia_jac,
                   args=(betas, events_dict, end_time, n_nodes, M, T_diff_sums, Ris), tol=1e-12)
    results = res.x
    # print("success ", res.success, ", status ", res.status, ", fun value ", res.fun)
    # print("message ", res.message)
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr = results[:7]
    C = np.array(results[7:])

    # scaling step
    C_sum = np.sum(C)
    if C_sum != 0:
        C = C / C_sum
        alpha_n = alpha_n * C_sum
        alpha_r = alpha_r * C_sum
        alpha_br = alpha_br * C_sum
        alpha_gr = alpha_gr * C_sum
        alpha_al = alpha_al * C_sum
        alpha_alr = alpha_alr * C_sum

    return (mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, C, betas)

""" off-diagonal """
def LL_6_alpha_kernel_sum_off(params, ed, ed_r, end_time, N_b, M_ab, T_diff_sums=None, T_diff_sums_r=None, Ris=None):
    # ed: dictionary {key:node_pair, value: array of events} <- block pair (a, b)
    # ed_r: dictionary {key:node_pair, value: array of events} <- block pair (b, a)
    # C: scaling parameters - same length as betas
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, C, betas = params
    Q = len(betas)
    N_a = M_ab // N_b
    ##first term
    first = -M_ab * mu * end_time
    ##second term
    if T_diff_sums is None:
        T_diff_sums = cal_diff_sums_Q(ed, end_time, betas)
        T_diff_sums_r = cal_diff_sums_Q(ed_r, end_time, betas)
    second = -(alpha_n + alpha_br * (N_b - 1) + alpha_al  * (N_a - 1)) * C @ T_diff_sums
    second -= (alpha_r + alpha_gr * (N_b - 1) + alpha_alr * (N_a - 1)) * C @ T_diff_sums_r
    ##third term
    if Ris is None:
        Ris = cal_R_6_alpha_kernel_sum_off(ed, ed_r, betas)
    third = 0
    for i in range(len(Ris)):
        col_sum = np.zeros(Ris[i].shape[0])
        for q in range(Q):
            col_sum[:] += C[q] * betas[q]  * (alpha_n * Ris[i][:, 0 + q * 6] + alpha_r * Ris[i][:, 1 + q * 6] +
                                              alpha_br * Ris[i][:, 2 + q * 6] + alpha_gr * Ris[i][:, 3 + q * 6] +
                                              alpha_al * Ris[i][:, 4 + q * 6] + alpha_alr * Ris[i][:,5 + q * 6])
        col_sum += mu
        third += np.sum(np.log(col_sum))
    log_likelihood_value = first + second + third
    return log_likelihood_value
def NLL_6_alpha_kernel_sum_off(p, betas, ed, ed_r, end_time, N_b, M_ab, T_diff_sums, T_diff_sums_r, Ris):
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr = p[:7]
    C = np.array(p[7:])
    # scaling step - constaint C sums to 1
    C_sum = np.sum(C)
    if (C_sum != 0):
        C = C / C_sum
        alpha_n = alpha_n * C_sum
        alpha_r = alpha_r * C_sum
        alpha_br = alpha_br * C_sum
        alpha_gr = alpha_gr * C_sum
        alpha_al = alpha_al * C_sum
        alpha_alr = alpha_alr * C_sum
    params = mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, C, betas
    return -LL_6_alpha_kernel_sum_off(params, ed, ed_r, end_time, N_b, M_ab, T_diff_sums, T_diff_sums_r, Ris)
def NLL_6_alpha_kernel_sum_off_jac(p, betas, ed, ed_r, end_time, N_b, M_ab, T_diff_sums, T_diff_sums_r, Ris):
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr = p[:7]
    C = np.array(p[7:])
    Q = len(C)
    N_a = M_ab // N_b
    # derivatives of second term
    d_mu = M_ab * end_time
    d_alphas = np.zeros(6)
    d_alphas[0] = C @ T_diff_sums
    d_alphas[1] = C @ T_diff_sums_r
    d_alphas[2] = (N_b - 1) * C @ T_diff_sums
    d_alphas[3] = (N_b - 1) * C @ T_diff_sums_r
    d_alphas[4] = (N_a - 1) * C @ T_diff_sums
    d_alphas[5] = (N_a - 1) * C @ T_diff_sums_r
    d_C = (alpha_n + alpha_br * (N_b - 1) + alpha_al  * (N_a - 1)) * T_diff_sums \
          + (alpha_r + alpha_gr * (N_b - 1) + alpha_alr * (N_a - 1)) * T_diff_sums_r
    # derivatives of third term
    for i in range(len(Ris)):
        denominator = np.zeros(Ris[i].shape[0])
        # one column for each alpha_j
        numerator_alphas = np.zeros((Ris[i].shape[0], 6))
        numerator_C = []
        for q in range(Q):
            numerator_C.append(betas[q]*( alpha_n * Ris[i][:, 0 + q * 6] + alpha_r * Ris[i][:, 1 + q * 6] +
                                          alpha_br * Ris[i][:, 2 + q * 6] + alpha_gr * Ris[i][:,3 + q * 6] +
                                          alpha_al * Ris[i][:, 4 + q * 6] + alpha_alr * Ris[i][:,5 + q * 6]))
            denominator += C[q] * numerator_C[q]
            for j in range(6):
                numerator_alphas[:, j] += C[q] * betas[q] * Ris[i][:, j + q * 6]
        denominator += mu
        d_mu -= np.sum(1 / denominator)
        for a in range(6):
            d_alphas[a] -= np.sum(numerator_alphas[:, a] / denominator)
        for q in range(Q):
            d_C[q] -= np.sum(numerator_C[q] / denominator)
    return np.hstack((d_mu, d_alphas, d_C))
def fit_6_alpha_kernel_sum_off(ed, ed_r, end_time, n_nodes_to, M, betas):
    Q = len(betas)
    # events_dict : (u,v):array_of_events
    if len(ed) == 0:  # handling empty block pair with no events
        # (mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, C, betas)
        C = np.zeros(Q)
        return (1e-10, 0, 0, 0, 0, 0, 0, C, betas)

    # calculate fixed terms in log-likelihood
    Ris = cal_R_6_alpha_kernel_sum_off(ed, ed_r, betas)
    T_diff_sums = cal_diff_sums_Q(ed, end_time, betas)
    T_diff_sums_r = cal_diff_sums_Q(ed_r, end_time, betas)

    # initialize parameters
    # mu_alpha_init = [1e-2, 2e-2, 2e-2, 1e-2, 1e-2]  # <-- fixed initialization
    mu_i = np.random.uniform(1e-6, 1e-2)
    alpha_n_i, alpha_r_i = np.random.uniform(0.1, 0.5, 2)
    alpha_br_i, alpha_gr_i, alpha_al_i, alpha_alr_i = np.random.uniform(1e-5, 0.1, 4)
    mu_alpha_init = [mu_i, alpha_n_i, alpha_r_i, alpha_br_i, alpha_gr_i, alpha_al_i, alpha_alr_i] # <-- random initialization
    C = [1 / Q] * Q  # <-- fixed initialization
    init_param = tuple(mu_alpha_init + C)
    # define bounds
    mu_alpha_bo = [(1e-7, None)] * 7
    C_bo = [(0, 1)] * Q
    bounds = tuple(mu_alpha_bo + C_bo)

    # minimize function
    # options = {'disp': True}
    res = minimize(NLL_6_alpha_kernel_sum_off, init_param, method='L-BFGS-B', bounds=bounds, jac=NLL_6_alpha_kernel_sum_off_jac,
                   args=(betas, ed, ed_r, end_time, n_nodes_to, M, T_diff_sums, T_diff_sums_r, Ris), tol=1e-12)
    results = res.x
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr = results[:7]
    C = np.array(results[7:])

    # scaling step -
    C_sum = np.sum(C)
    if C_sum != 0:
        C = C / C_sum
        alpha_n = alpha_n * C_sum
        alpha_r = alpha_r * C_sum
        alpha_br = alpha_br * C_sum
        alpha_gr = alpha_gr * C_sum
        alpha_al = alpha_al * C_sum
        alpha_alr = alpha_alr * C_sum
    return (mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, C, betas)
#%% sum of kernles (n, r, br, gr) model --- beta*alpha*exp(-beta*t)
def cal_R_4_alpha_kernel_sum_dia(events_dict, betas):
    # betas: array of fixed decays
    Q = len(betas)
    Ris = []
    for (u, v) in events_dict:
        # array of events of node pair (u,v)
        uv_events = events_dict[(u, v)]
        num_events_uv = len(uv_events)
        if num_events_uv == 0:
            Ris.append(np.array([0]))  # R=0 if node_pair (u,v) has no events
        else:
            # 4*Q columns (alpha_n, alpha_r, alpha_br, alpha_gr)*Q
            Ri = np.zeros((num_events_uv, 4 * Q))
            uv_intertimes = (uv_events[1:] - uv_events[:-1])
            # (#_uv_events-1, Q) array
            e_intertimes_Q = np.zeros((len(uv_intertimes), Q))
            for q in range(Q):
                e_intertimes_Q[:, q] = np.exp(-betas[q] * uv_intertimes)
            for (x, y) in events_dict:
                if x == u or y == u:
                    prev_index = 0
                    # same node_pair events (alpha_n)
                    if (u, v) == (x, y):
                        for k in range(1, num_events_uv):
                            for q in range(Q):
                                Ri[k, 0 + q * 4] = e_intertimes_Q[k - 1, q] * (1 + Ri[k - 1, 0 + q * 4])
                    # reciprocal node_pair events (alpha_r)
                    elif (v, u) == (x, y):
                        Ri_temp = get_Ri_temp_Q(uv_events, e_intertimes_Q, events_dict[(x, y)], betas)
                        for q in range(Q):
                            Ri[:, 1 + q * 4] = Ri_temp[:, q]
                    # br node_pairs events (alpha_br)
                    elif u == x and v != y:
                        Ri_temp = get_Ri_temp_Q(uv_events, e_intertimes_Q, events_dict[(x, y)], betas)
                        for q in range(Q):
                            Ri[:, 2 + q * 4] += Ri_temp[:, q]
                    # gr node_pairs events (alpha_gr)
                    elif u == y and v != x:
                        Ri_temp = get_Ri_temp_Q(uv_events, e_intertimes_Q, events_dict[(x, y)], betas)
                        for q in range(Q):
                            Ri[:, 3 + q * 4] += Ri_temp[:, q]
            Ris.append(Ri)
    # return list of arrays - list size = #node_pairs_events_in block_pair
    return Ris
def LL_4_alpha_kernel_sum_dia(params, events_dict, end_time, n_nodes, M, T_diff_sums=None, Ris=None):
    # events_dict of node_pairs with events
    # C: scaling parameters - same length as betas
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, C, betas = params
    Q = len(betas)
    ##first term
    first = -M * mu * end_time
    ##second term
    if T_diff_sums is None:
        events_array = list(events_dict.values())
        # if block pair has no events
        if len(events_array) == 0:
            return first
        T_diff = end_time - np.concatenate(events_array)
        T_diff_sums = np.zeros(Q, )
        for q in range(Q):
            T_diff_sums[q] = np.sum(1 - np.exp(-betas[q] * T_diff))
    second = -(alpha_n + alpha_r + (alpha_br + alpha_gr) * (n_nodes - 2)) * C @ T_diff_sums
    ##third term
    if Ris is None:
        Ris = cal_R_4_alpha_kernel_sum_dia(events_dict, betas)
    third = 0
    for i in range(len(Ris)):
        col_sum = np.zeros(Ris[i].shape[0])
        for q in range(Q):
            col_sum[:] += C[q]*betas[q] * (alpha_n * Ris[i][:, 0 + q * 4] + alpha_r * Ris[i][:, 1 + q * 4] +
                                  alpha_br * Ris[i][:, 2 + q * 4] + alpha_gr * Ris[i][:,3 + q * 4])
        col_sum += mu
        third += np.sum(np.log(col_sum))
    log_likelihood_value = first + second + third
    return log_likelihood_value
def NLL_4_alpha_kernel_sum_dia(p, betas, events_dict, end_time, n_nodes, M, T_diff_sums, Ris):
    mu, alpha_n, alpha_r, alpha_br, alpha_gr = p[:5]
    C = np.array(p[5:])

    # scaling step - constaint C sums to 1
    C_sum = np.sum(C)
    if (C_sum != 0):
        C = C / C_sum
        alpha_n = alpha_n * C_sum
        alpha_r = alpha_r * C_sum
        alpha_br = alpha_br * C_sum
        alpha_gr = alpha_gr * C_sum

    params = mu, alpha_n, alpha_r, alpha_br, alpha_gr, C, betas
    return -LL_4_alpha_kernel_sum_dia(params, events_dict, end_time, n_nodes, M, T_diff_sums, Ris)
def NLL_4_alpha_kernel_sum_dia_jac(p, betas, events_dict, end_time, n_nodes, M, T_diff_sums, Ris):
    mu, alpha_n, alpha_r, alpha_br, alpha_gr = p[:5]
    C = np.array(p[5:])
    Q = len(C)
    # derivatives of second term
    d_mu = M * end_time
    d_alpha_n = C @ T_diff_sums
    d_alpha_r = C @ T_diff_sums
    d_alpha_br = (n_nodes - 2) * C @ T_diff_sums
    d_alpha_gr = (n_nodes - 2) * C @ T_diff_sums
    d_C = (alpha_n + alpha_r + (n_nodes - 2) * (alpha_br + alpha_gr)) * T_diff_sums
    # derivatives of third term
    for i in range(len(Ris)):
        denominator = np.zeros(Ris[i].shape[0])
        # one column for each alpha_j
        numerator_alphas = np.zeros((Ris[i].shape[0], 4))
        numerator_C = []
        for q in range(Q):
            numerator_C.append(betas[q]* (alpha_n * Ris[i][:, 0 + q * 4] + alpha_r * Ris[i][:, 1 + q * 4] +
                               alpha_br * Ris[i][:, 2 + q * 4] + alpha_gr * Ris[i][:,3 + q * 4]))
            denominator += C[q] * numerator_C[q]
            for j in range(4):
                numerator_alphas[:, j] += C[q] * betas[q] * Ris[i][:, j + q * 4]
        denominator += mu
        d_mu -= np.sum(1 / denominator)
        d_alpha_n -= np.sum(numerator_alphas[:, 0] / denominator)
        d_alpha_r -= np.sum(numerator_alphas[:, 1] / denominator)
        d_alpha_br -= np.sum(numerator_alphas[:, 2] / denominator)
        d_alpha_gr -= np.sum(numerator_alphas[:, 3] / denominator)
        for q in range(Q):
            d_C[q] -= np.sum(numerator_C[q] / denominator)
    return np.hstack((d_mu, d_alpha_n, d_alpha_r, d_alpha_br, d_alpha_gr, d_C))
def fit_4_alpha_kernel_sum_dia(events_dict, end_time, n_nodes, M, betas):
    Q = len(betas)
    # events_dict : (u,v):array_of_events
    if len(events_dict) == 0:  # handling empty block pair with no events
        # (mu, alpha_n, alpha_r, alpha_br, alpha_gr, C, betas)
        C = np.zeros(Q)
        return (1e-10, 0, 0, 0, 0, C, betas)

    # calculate fixed terms in log-likelihood
    Ris = cal_R_4_alpha_kernel_sum_dia(events_dict, betas)
    events_array = list(events_dict.values())
    T_diff = end_time - np.concatenate(events_array)
    T_diff_sums = np.zeros(Q, )
    for q in range(Q):
        T_diff_sums[q] = np.sum(1 - np.exp(-betas[q] * T_diff))

    # initialize parameters (mu, alpha_n, alpha_r, alpha_br, alpha_gr, c1, ..., cQ)
    mu_i = np.random.uniform(1e-6, 1e-2)
    alpha_n_i, alpha_r_i = np.random.uniform(0.1, 0.5, 2)
    alpha_br_i, alpha_gr_i = np.random.uniform(1e-5, 0.1, 2)
    mu_alpha_init = [mu_i, alpha_n_i, alpha_r_i, alpha_br_i, alpha_gr_i] # <-- random initialization
    # mu_alpha_init = [1e-2, 2e-2, 2e-2, 1e-2, 1e-2]  # <-- fixed initialization
    C = [1 / Q] * Q  # <-- fixed initialization
    init_param = tuple(mu_alpha_init + C)
    # define bounds
    mu_alpha_bo = [(1e-7, None)] * 5
    C_bo = [(0, 1)] * Q
    bounds = tuple(mu_alpha_bo + C_bo)

    # minimize function
    res = minimize(NLL_4_alpha_kernel_sum_dia, init_param, method='L-BFGS-B', bounds=bounds, jac=NLL_4_alpha_kernel_sum_dia_jac,
                   args=(betas, events_dict, end_time, n_nodes, M, T_diff_sums, Ris), tol=1e-12)
    results = res.x
    mu, alpha_n, alpha_r, alpha_br, alpha_gr = results[:5]
    C = np.array(results[5:])

    # scaling step
    C_sum = np.sum(C)
    if C_sum != 0:
        C = C / C_sum
        alpha_n = alpha_n * C_sum
        alpha_r = alpha_r * C_sum
        alpha_br = alpha_br * C_sum
        alpha_gr = alpha_gr * C_sum

    return (mu, alpha_n, alpha_r, alpha_br, alpha_gr, C, betas)

""" off-diagonal """
def cal_R_4_alpha_kernel_sum_off(events_dict, events_dict_r, betas):
    # betas: array of fixed decays
    Q = len(betas)
    Ris = []
    for (u, v) in events_dict:
        # array of events of node pair (u,v)
        uv_events = events_dict[(u, v)]
        num_events_uv = len(uv_events)
        if num_events_uv == 0:
            Ris.append(np.array([0]))  # R=0 if node_pair (u,v) has no events
        else:
            # 4*Q columns (alpha_n, alpha_r, alpha_br, alpha_gr)*Q
            Ri = np.zeros((num_events_uv, 4 * Q))
            uv_intertimes = (uv_events[1:] - uv_events[:-1])
            # (#_uv_events-1, Q) array
            e_intertimes_Q = np.zeros((len(uv_intertimes), Q))
            for q in range(Q):
                e_intertimes_Q[:, q] = np.exp(-betas[q] * uv_intertimes)

            # loop through node pairs in block pair ab
            for (x, y) in events_dict:
                if x == u:
                    prev_index = 0
                    # same node_pair events (alpha_n)
                    if (u, v) == (x, y):
                        for k in range(1, num_events_uv):
                            for q in range(Q):
                                Ri[k, 0 + q * 4] = e_intertimes_Q[k - 1, q] * (1 + Ri[k - 1, 0 + q * 4])
                    # br node_pairs events (alpha_br)
                    else:
                        Ri_temp = get_Ri_temp_Q(uv_events, e_intertimes_Q, events_dict[(x, y)], betas)
                        for q in range(Q):
                            Ri[:, 2 + q * 4] += Ri_temp[:, q]

            # loop through node pairs in reciprocal block pair ba
            for (x, y) in events_dict_r:
                if y == u:
                    # reciprocal node_pair events (alpha_r)
                    if (v, u) == (x, y):
                        Ri_temp = get_Ri_temp_Q(uv_events, e_intertimes_Q, events_dict_r[(x, y)], betas)
                        for q in range(Q):
                            Ri[:, 1 + q * 4] = Ri_temp[:, q]
                    # gr node_pairs events (alpha_gr)
                    else:
                        Ri_temp = get_Ri_temp_Q(uv_events, e_intertimes_Q, events_dict_r[(x, y)], betas)
                        for q in range(Q):
                            Ri[:, 3 + q * 4] += Ri_temp[:, q]

            Ris.append(Ri)
    # return list of arrays - list size = #node_pairs_events_in block_pair
    return Ris
def LL_4_alpha_kernel_sum_off(params, ed, ed_r, end_time, n_nodes_to, M, T_diff_sums=None, T_diff_sums_r=None, Ris=None):
    # ed: dictionary {key:node_pair, value: array of events} <- block pair (a, b)
    # ed_r: dictionary {key:node_pair, value: array of events} <- block pair (b, a)
    # C: scaling parameters - same length as betas
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, C, betas = params
    Q = len(betas)
    ##first term
    first = -M * mu * end_time
    ##second term
    if T_diff_sums is None:
        events_array = list(ed.values())
        events_array_r = list(ed_r.values())
        T_diff_sums = np.zeros(Q, )
        T_diff_sums_r = np.zeros(Q, )
        if len(events_array) != 0:
            T_diff = end_time - np.concatenate(events_array)
            for q in range(Q):
                T_diff_sums[q] = np.sum(1 - np.exp(-betas[q] * T_diff))
        if len(events_array_r) != 0:
            T_diff = end_time - np.concatenate(events_array_r)
            for q in range(Q):
                T_diff_sums_r[q] = np.sum(1 - np.exp(-betas[q] * T_diff))
    second = -(alpha_n + alpha_br * (n_nodes_to - 1)) * C @ T_diff_sums
    second -= (alpha_r + alpha_gr * (n_nodes_to - 1)) * C @ T_diff_sums_r
    ##third term
    if Ris is None:
        Ris = cal_R_4_alpha_kernel_sum_off(ed, ed_r, betas)
    third = 0
    for i in range(len(Ris)):
        col_sum = np.zeros(Ris[i].shape[0])
        for q in range(Q):
            col_sum[:] += C[q] * betas[q]  * (alpha_n * Ris[i][:, 0 + q * 4] + alpha_r * Ris[i][:, 1 + q * 4] +
                                              alpha_br * Ris[i][:, 2 + q * 4] + alpha_gr * Ris[i][:, 3 + q * 4])
        col_sum += mu
        third += np.sum(np.log(col_sum))
    log_likelihood_value = first + second + third
    return log_likelihood_value
def NLL_4_alpha_kernel_sum_off(p, betas, ed, ed_r, end_time, n_nodes_to, M, T_diff_sums, T_diff_sums_r, Ris):
    mu, alpha_n, alpha_r, alpha_br, alpha_gr = p[:5]
    C = np.array(p[5:])

    # scaling step - constaint C sums to 1
    C_sum = np.sum(C)
    if (C_sum != 0):
        C = C / C_sum
        alpha_n = alpha_n * C_sum
        alpha_r = alpha_r * C_sum
        alpha_br = alpha_br * C_sum
        alpha_gr = alpha_gr * C_sum

    params = mu, alpha_n, alpha_r, alpha_br, alpha_gr, C, betas
    return -LL_4_alpha_kernel_sum_off(params, ed, ed_r, end_time, n_nodes_to, M, T_diff_sums, T_diff_sums_r, Ris)
def NLL_4_alpha_kernel_sum_off_jac(p, betas, ed, ed_r, end_time, n_nodes_to, M, T_diff_sums, T_diff_sums_r, Ris):
    mu, alpha_n, alpha_r, alpha_br, alpha_gr = p[:5]
    C = np.array(p[5:])
    Q = len(C)
    # derivatives of second term
    d_mu = M * end_time
    d_alpha_n = C @ T_diff_sums
    d_alpha_r = C @ T_diff_sums_r
    d_alpha_br = (n_nodes_to - 1) * C @ T_diff_sums
    d_alpha_gr = (n_nodes_to - 1) * C @ T_diff_sums_r
    d_C = (alpha_n + (n_nodes_to - 1) * alpha_br) * T_diff_sums + (alpha_r + (n_nodes_to - 1) * alpha_gr) * T_diff_sums_r
    # derivatives of third term
    for i in range(len(Ris)):
        denominator = np.zeros(Ris[i].shape[0])
        # one column for each alpha_j
        numerator_alphas = np.zeros((Ris[i].shape[0], 4))
        numerator_C = []
        for q in range(Q):
            numerator_C.append(betas[q]*( alpha_n * Ris[i][:, 0 + q * 4] + alpha_r * Ris[i][:, 1 + q * 4] +
                                alpha_br * Ris[i][:, 2 + q * 4] + alpha_gr * Ris[i][:,3 + q * 4]))
            denominator += C[q] * numerator_C[q]
            for j in range(4):
                numerator_alphas[:, j] += C[q] * betas[q] * Ris[i][:, j + q * 4]
        denominator += mu
        d_mu -= np.sum(1 / denominator)
        d_alpha_n -= np.sum(numerator_alphas[:, 0] / denominator)
        d_alpha_r -= np.sum(numerator_alphas[:, 1] / denominator)
        d_alpha_br -= np.sum(numerator_alphas[:, 2] / denominator)
        d_alpha_gr -= np.sum(numerator_alphas[:, 3] / denominator)
        for q in range(Q):
            d_C[q] -= np.sum(numerator_C[q] / denominator)
    return np.hstack((d_mu, d_alpha_n, d_alpha_r, d_alpha_br, d_alpha_gr, d_C))
def fit_4_alpha_kernel_sum_off(ed, ed_r, end_time, n_nodes_to, M, betas):
    Q = len(betas)
    # events_dict : (u,v):array_of_events
    if len(ed) == 0:  # handling empty block pair with no events
        # (mu, alpha_n, alpha_r, alpha_br, alpha_gr, C, betas)
        C = np.zeros(Q)
        return (1e-10, 0, 0, 0, 0, C, betas)

    # calculate fixed terms in log-likelihood
    Ris = cal_R_4_alpha_kernel_sum_off(ed, ed_r, betas)
    events_array = list(ed.values())
    events_array_r = list(ed_r.values())
    T_diff_sums = np.zeros(Q, )
    T_diff_sums_r = np.zeros(Q, )
    if len(events_array) != 0:
        T_diff = end_time - np.concatenate(events_array)
        for q in range(Q):
            T_diff_sums[q] = np.sum(1 - np.exp(-betas[q] * T_diff))
    if len(events_array_r) != 0:
        T_diff = end_time - np.concatenate(events_array_r)
        for q in range(Q):
            T_diff_sums_r[q] = np.sum(1 - np.exp(-betas[q] * T_diff))

    # initialize parameters
    # mu_alpha_init = [1e-2, 2e-2, 2e-2, 1e-2, 1e-2]  # <-- fixed initialization
    mu_i = np.random.uniform(1e-6, 1e-2)
    alpha_n_i, alpha_r_i = np.random.uniform(0.1, 0.5, 2)
    alpha_br_i, alpha_gr_i = np.random.uniform(1e-5, 0.1, 2)
    mu_alpha_init = [mu_i, alpha_n_i, alpha_r_i, alpha_br_i, alpha_gr_i] # <-- random initialization
    C = [1 / Q] * Q  # <-- fixed initialization
    init_param = tuple(mu_alpha_init + C)
    # define bounds
    mu_alpha_bo = [(1e-7, None)] * 5
    C_bo = [(0, 1)] * Q
    bounds = tuple(mu_alpha_bo + C_bo)

    # minimize function
    res = minimize(NLL_4_alpha_kernel_sum_off, init_param, method='L-BFGS-B', bounds=bounds, jac=NLL_4_alpha_kernel_sum_off_jac,
                   args=(betas, ed, ed_r, end_time, n_nodes_to, M, T_diff_sums, T_diff_sums_r, Ris), tol=1e-12)
    results = res.x
    mu, alpha_n, alpha_r, alpha_br, alpha_gr = results[:5]
    C = np.array(results[5:])

    # scaling step -
    C_sum = np.sum(C)
    if C_sum != 0:
        C = C / C_sum
        alpha_n = alpha_n * C_sum
        alpha_r = alpha_r * C_sum
        alpha_br = alpha_br * C_sum
        alpha_gr = alpha_gr * C_sum

    return (mu, alpha_n, alpha_r, alpha_br, alpha_gr, C, betas)


#%% sum of kernles (n, r) model --- beta*alpha*exp(-beta*t)

""" diagonal """
def cal_R_2_alpha_kernel_sum_dia(events_dict, betas):
    # betas: array of Q fixed decays
    Q = len(betas)
    Ris = []
    for (u, v) in events_dict:
        # array of events of node pair (u,v)
        uv_events = events_dict[(u, v)]
        num_events_uv = len(uv_events)
        if num_events_uv == 0:
            Ris.append(np.array([0]))  # R=0 if node_pair (u,v) has no events
        else:
            # 2*Q columns (alpha_n, alpha_r)*Q
            Ri = np.zeros((num_events_uv, 2 * Q))
            uv_intertimes = (uv_events[1:] - uv_events[:-1])
            # (#_uv_events-1, Q) array
            e_intertimes_Q = np.zeros((len(uv_intertimes), Q))
            for q in range(Q):
                e_intertimes_Q[:, q] = np.exp(-betas[q] * uv_intertimes)
            for (x, y) in events_dict:
                if x == u or y == u:
                    prev_index = 0
                    # same node_pair events (alpha_n)
                    if (u, v) == (x, y):
                        for k in range(1, num_events_uv):
                            for q in range(Q):
                                Ri[k, 0 + q * 2] = e_intertimes_Q[k - 1, q] * (1 + Ri[k - 1, 0 + q * 2])
                    # reciprocal node_pair events (alpha_r)
                    elif (v, u) == (x, y):
                        for k in range(0, num_events_uv):
                            # return index below which t(x,y) < kth event of (u,v)
                            # if no events exists returns len(events(x,y))
                            index = bisect_left(events_dict[(x, y)], uv_events[k], lo=prev_index)
                            # no events found
                            if index == prev_index:
                                if k == 0: continue
                                for q in range(Q):
                                    Ri[k, 1 + q * 2] = e_intertimes_Q[k - 1, q] * Ri[k - 1, 1 + q * 2]
                            else:
                                diff_times = uv_events[k] - events_dict[(x, y)][prev_index:index]
                                if k == 0:
                                    for q in range(Q):
                                        Ri[k, 1 + q * 2] = np.sum(np.exp(-betas[q] * diff_times))
                                else:
                                    for q in range(Q):
                                        Ri[k, 1 + q * 2] = e_intertimes_Q[k - 1, q] * Ri[k - 1, 1 + q * 2] + np.sum(
                                            np.exp(-betas[q] * diff_times))
                                prev_index = index
            Ris.append(Ri)
    # return list of arrays - list size = #node_pairs_events_in block_pair
    return Ris
def LL_2_alpha_kernel_sum_dia(params, events_dict, end_time, n_nodes, M, T_diff_sums=None, Ris=None):
    # events_dict of node_pairs with events
    # C: scaling parameters - same length as betas
    mu, alpha_n, alpha_r, C, betas = params
    Q = len(betas)
    ##first term
    first = -M * mu * end_time
    ##second term
    if T_diff_sums is None:
        events_array = list(events_dict.values())
        # if block pair has no events
        if len(events_array) == 0:
            return first
        T_diff = end_time - np.concatenate(events_array)
        T_diff_sums = np.zeros(Q, )
        for q in range(Q):
            T_diff_sums[q] = np.sum(1 - np.exp(-betas[q] * T_diff))
    second = -(alpha_n + alpha_r) * C @ T_diff_sums
    ##third term
    if Ris is None:
        Ris = cal_R_2_alpha_kernel_sum_dia(events_dict, betas)
    third = 0
    for i in range(len(Ris)):
        col_sum = np.zeros(Ris[i].shape[0])
        for q in range(Q):
            col_sum[:] += C[q] * betas[q] * (alpha_n * Ris[i][:, 0 + q * 2] + alpha_r * Ris[i][:, 1 + q * 2])
        col_sum += mu
        third += np.sum(np.log(col_sum))
    log_likelihood_value = first + second + third
    return log_likelihood_value
def NLL_2_alpha_kernel_sum_dia(p, betas, events_dict, end_time, n_nodes, M, T_diff_sums, Ris):
    mu, alpha_n, alpha_r = p[:3]
    C = np.array(p[3:])

    # scaling step - constaint C sums to 1
    C_sum = np.sum(C)
    if (C_sum != 0):
        C = C / C_sum
        alpha_n = alpha_n * C_sum
        alpha_r = alpha_r * C_sum

    params = mu, alpha_n, alpha_r, C, betas
    return -LL_2_alpha_kernel_sum_dia(params, events_dict, end_time, n_nodes, M, T_diff_sums, Ris)
def NLL_2_alpha_kernel_sum_dia_jac(p, betas, events_dict, end_time, n_nodes, M, T_diff_sums, Ris):
    mu, alpha_n, alpha_r = p[:3]
    C = np.array(p[3:])
    Q = len(C)
    # derivatives of second term
    d_mu = M * end_time
    d_alpha_n = C @ T_diff_sums
    d_alpha_r = C @ T_diff_sums
    d_C = (alpha_n + alpha_r) * T_diff_sums
    # derivatives of third term
    for i in range(len(Ris)):
        denominator = np.zeros(Ris[i].shape[0])
        # one column for each alpha_j
        numerator_alphas = np.zeros((Ris[i].shape[0], 2))
        numerator_C = []
        for q in range(Q):
            numerator_C.append(betas[q] * (alpha_n * Ris[i][:, 0 + q * 2] + alpha_r * Ris[i][:, 1 + q * 2]))
            denominator += C[q] * numerator_C[q]
            for j in range(2):
                numerator_alphas[:, j] += C[q] * betas[q] * Ris[i][:, j + q * 2]
        denominator += mu
        d_mu -= np.sum(1 / denominator)
        d_alpha_n -= np.sum(numerator_alphas[:, 0] / denominator)
        d_alpha_r -= np.sum(numerator_alphas[:, 1] / denominator)
        for q in range(Q):
            d_C[q] -= np.sum(numerator_C[q] / denominator)
    return np.hstack((d_mu, d_alpha_n, d_alpha_r, d_C))
def fit_2_alpha_kernel_sum_dia(events_dict, end_time, n_nodes, M, betas):
    Q = len(betas)
    # events_dict : (u,v):array_of_events
    if len(events_dict) == 0:  # handling empty block pair with no events
        # (mu, alpha_n, alpha_r, C, betas)
        C = np.zeros(Q)
        return (1e-10, 0, 0, C, betas)

    # calculate fixed terms in log-likelihood
    Ris = cal_R_2_alpha_kernel_sum_dia(events_dict, betas)
    events_array = list(events_dict.values())
    T_diff = end_time - np.concatenate(events_array)
    T_diff_sums = np.zeros(Q, )
    for q in range(Q):
        T_diff_sums[q] = np.sum(1 - np.exp(-betas[q] * T_diff))

    # initialize parameters (mu, alpha_n, alpha_r, c1, ..., cQ)
    mu_i = np.random.uniform(1e-6, 1e-2)
    alpha_n_i, alpha_r_i = np.random.uniform(0.1, 0.5, 2)
    mu_alpha_init = [mu_i, alpha_n_i, alpha_r_i] # <-- random initialization
    # mu_alpha_init = [1e-2, 2e-2, 2e-2]  # <-- fixed initialization
    C = [1 / Q] * Q  # <-- fixed initialization
    init_param = tuple(mu_alpha_init + C)
    # define bounds
    mu_alpha_bo = [(1e-7, None)] * 3
    C_bo = [(0, 1)] * Q
    bounds = tuple(mu_alpha_bo + C_bo)

    # minimize function
    res = minimize(NLL_2_alpha_kernel_sum_dia, init_param, method='L-BFGS-B', bounds=bounds, jac=NLL_2_alpha_kernel_sum_dia_jac,
                   args=(betas, events_dict, end_time, n_nodes, M, T_diff_sums, Ris), tol=1e-12)
    results = res.x
    mu, alpha_n, alpha_r = results[:3]
    C = np.array(results[3:])

    # scaling step
    C_sum = np.sum(C)
    if C_sum != 0:
        C = C / C_sum
        alpha_n = alpha_n * C_sum
        alpha_r = alpha_r * C_sum

    return (mu, alpha_n, alpha_r, C, betas)

""" off-diagonal """
def cal_R_2_alpha_kernel_sum_off(events_dict, events_dict_r, betas):
    # betas: array of fixed decays
    Q = len(betas)
    Ris = []
    for (u, v) in events_dict:
        # array of events of node pair (u,v)
        uv_events = events_dict[(u, v)]
        num_events_uv = len(uv_events)
        if num_events_uv == 0:
            Ris.append(np.array([0]))  # R=0 if node_pair (u,v) has no events
        else:
            # 2*Q columns (alpha_n, alpha_r)*Q
            Ri = np.zeros((num_events_uv, 2 * Q))
            uv_intertimes = (uv_events[1:] - uv_events[:-1])
            # (#_uv_events-1, Q) array
            e_intertimes_Q = np.zeros((len(uv_intertimes), Q))
            for q in range(Q):
                e_intertimes_Q[:, q] = np.exp(-betas[q] * uv_intertimes)

            # loop through node pairs in block pair ab
            for (x, y) in events_dict:
                # same node_pair events (alpha_n)
                if (u, v) == (x, y):
                    for k in range(1, num_events_uv):
                        for q in range(Q):
                            Ri[k, 0 + q * 2] = e_intertimes_Q[k - 1, q] * (1 + Ri[k - 1, 0 + q * 2])
            # loop through node pairs in reciprocal block pair ba
            for (x, y) in events_dict_r:
                # reciprocal node_pair events (alpha_r)
                if (v, u) == (x, y):
                    prev_index = 0
                    for k in range(0, num_events_uv):
                        # return index below which t(x,y) < kth event of (u,v)
                        # if no events exists returns len(events(x,y))
                        index = bisect_left(events_dict_r[(x, y)], uv_events[k], lo=prev_index)
                        # no events found
                        if index == prev_index:
                            if k == 0: continue
                            for q in range(Q):
                                Ri[k, 1 + q * 2] = e_intertimes_Q[k - 1, q] * Ri[k - 1, 1 + q * 2]
                        else:
                            diff_times = uv_events[k] - events_dict_r[(x, y)][prev_index:index]
                            if k == 0:
                                for q in range(Q):
                                    Ri[k, 1 + q * 2] = np.sum(np.exp(-betas[q] * diff_times))
                            else:
                                for q in range(Q):
                                    Ri[k, 1 + q * 2] = e_intertimes_Q[k - 1, q] * Ri[k - 1, 1 + q * 2] + np.sum(
                                        np.exp(-betas[q] * diff_times))
                            prev_index = index
            Ris.append(Ri)
    # return list of arrays - list size = #node_pairs_events_in block_pair
    return Ris
def LL_2_alpha_kernel_sum_off(params, ed, ed_r, end_time, n_nodes_to, M, T_diff_sums=None, T_diff_sums_r=None, Ris=None):
    # ed: dictionary {key:node_pair, value: array of events} <- block pair (a, b)
    # ed_r: dictionary {key:node_pair, value: array of events} <- block pair (b, a)
    # C: scaling parameters - same length as betas
    mu, alpha_n, alpha_r, C, betas = params
    Q = len(betas)
    ##first term
    first = -M * mu * end_time
    ##second term
    if T_diff_sums is None:
        events_array = list(ed.values())
        events_array_r = list(ed_r.values())
        T_diff_sums = np.zeros(Q, )
        T_diff_sums_r = np.zeros(Q, )
        if len(events_array) != 0:
            T_diff = end_time - np.concatenate(events_array)
            for q in range(Q):
                T_diff_sums[q] = np.sum(1 - np.exp(-betas[q] * T_diff))
        if len(events_array_r) != 0:
            T_diff = end_time - np.concatenate(events_array_r)
            for q in range(Q):
                T_diff_sums_r[q] = np.sum(1 - np.exp(-betas[q] * T_diff))
    second = -alpha_n * C @ T_diff_sums
    second -= alpha_r * C @ T_diff_sums_r
    ##third term
    if Ris is None:
        Ris = cal_R_2_alpha_kernel_sum_off(ed, ed_r, betas)
    third = 0
    for i in range(len(Ris)):
        col_sum = np.zeros(Ris[i].shape[0])
        for q in range(Q):
            col_sum[:] += C[q] * betas[q] * (alpha_n * Ris[i][:, 0 + q * 2] + alpha_r * Ris[i][:, 1 + q * 2])
        col_sum += mu
        third += np.sum(np.log(col_sum))
    log_likelihood_value = first + second + third
    return log_likelihood_value
def NLL_2_alpha_kernel_sum_off(p, betas, ed, ed_r, end_time, n_nodes_to, M, T_diff_sums, T_diff_sums_r, Ris):
    mu, alpha_n, alpha_r = p[:3]
    C = np.array(p[3:])

    # scaling step - constaint C sums to 1
    C_sum = np.sum(C)
    if (C_sum != 0):
        C = C / C_sum
        alpha_n = alpha_n * C_sum
        alpha_r = alpha_r * C_sum

    params = mu, alpha_n, alpha_r, C, betas
    return -LL_2_alpha_kernel_sum_off(params, ed, ed_r, end_time, n_nodes_to, M, T_diff_sums, T_diff_sums_r, Ris)
def NLL_2_alpha_kernel_sum_off_jac(p, betas, ed, ed_r, end_time, n_nodes_to, M, T_diff_sums, T_diff_sums_r, Ris):
    mu, alpha_n, alpha_r = p[:3]
    C = np.array(p[3:])
    Q = len(C)
    # derivatives of second term
    d_mu = M * end_time
    d_alpha_n = C @ T_diff_sums
    d_alpha_r = C @ T_diff_sums_r
    d_C = alpha_n * T_diff_sums + alpha_r * T_diff_sums_r
    # derivatives of third term
    for i in range(len(Ris)):
        denominator = np.zeros(Ris[i].shape[0])
        # one column for each alpha_j
        numerator_alphas = np.zeros((Ris[i].shape[0], 2))
        numerator_C = []
        for q in range(Q):
            numerator_C.append(betas[q] * (alpha_n * Ris[i][:, 0 + q * 2] + alpha_r * Ris[i][:, 1 + q * 2]))
            denominator += C[q] * numerator_C[q]
            for j in range(2):
                numerator_alphas[:, j] += C[q] * betas[q] * Ris[i][:, j + q * 2]
        denominator += mu
        d_mu -= np.sum(1 / denominator)
        d_alpha_n -= np.sum(numerator_alphas[:, 0] / denominator)
        d_alpha_r -= np.sum(numerator_alphas[:, 1] / denominator)
        for q in range(Q):
            d_C[q] -= np.sum(numerator_C[q] / denominator)
    return np.hstack((d_mu, d_alpha_n, d_alpha_r, d_C))
def fit_2_alpha_kernel_sum_off(ed, ed_r, end_time, n_nodes_to, M, betas):
    Q = len(betas)
    # events_dict : (u,v):array_of_events
    if len(ed) == 0:  # handling empty block pair with no events
        # (mu, alpha_n, alpha_r, C, betas)
        C = np.zeros(Q)
        return (1e-10, 0, 0, C, betas)

    # calculate fixed terms in log-likelihood
    Ris = cal_R_2_alpha_kernel_sum_off(ed, ed_r, betas)
    events_array = list(ed.values())
    events_array_r = list(ed_r.values())
    T_diff_sums = np.zeros(Q, )
    T_diff_sums_r = np.zeros(Q, )
    if len(events_array) != 0:
        T_diff = end_time - np.concatenate(events_array)
        for q in range(Q):
            T_diff_sums[q] = np.sum(1 - np.exp(-betas[q] * T_diff))
    if len(events_array_r) != 0:
        T_diff = end_time - np.concatenate(events_array_r)
        for q in range(Q):
            T_diff_sums_r[q] = np.sum(1 - np.exp(-betas[q] * T_diff))

    # initialize parameters (mu, alpha_n, alpha_r, c1, ..., cQ)
    mu_i = np.random.uniform(1e-6, 1e-2)
    alpha_n_i, alpha_r_i = np.random.uniform(0.1, 0.5, 2)
    mu_alpha_init = [mu_i, alpha_n_i, alpha_r_i] # <-- random initialization
    # mu_alpha_init = [1e-2, 2e-2, 2e-2]  # <-- fixed initialization
    C = [1 / Q] * Q  # <-- fixed initialization
    init_param = tuple(mu_alpha_init + C)
    # define bounds
    mu_alpha_bo = [(1e-7, None)] * 3
    C_bo = [(0, 1)] * Q
    bounds = tuple(mu_alpha_bo + C_bo)

    # minimize function
    res = minimize(NLL_2_alpha_kernel_sum_off, init_param, method='L-BFGS-B', bounds=bounds, jac=NLL_2_alpha_kernel_sum_off_jac,
                   args=(betas, ed, ed_r, end_time, n_nodes_to, M, T_diff_sums, T_diff_sums_r, Ris), tol=1e-12)
    results = res.x
    mu, alpha_n, alpha_r = results[:3]
    C = np.array(results[3:])

    # scaling step -
    C_sum = np.sum(C)
    if C_sum != 0:
        C = C / C_sum
        alpha_n = alpha_n * C_sum
        alpha_r = alpha_r * C_sum

    return (mu, alpha_n, alpha_r, C, betas)


#%% detailed log-likelihood functions
""" only for checking code """
# single beta - beta*alpha*exp(-beta*t) kernel
def log_likelihood_detailed_2(params_array, events_list, end_time, M):
    """ mu_array : (M,) array : baseline intensity of each process
        alpoha_array: (M,M) narray: adjacency*beta (actual jumbs) """
    mu_array, alpha_array, beta = params_array
    # set all mu=0 to mu=1e-10/end_time
    # mu_array[mu_array==0] = 1e-10/end_time

    # first term
    first = -np.sum(mu_array) * end_time
    # second term
    second = 0
    for m in range(M):
        for v in range(M):
            if len(events_list[v]) != 0:
                second -= alpha_array[m, v] * np.sum(1 - np.exp(-beta * (end_time - events_list[v])))
    # third term
    third = 0
    for m in range(M):
        for k in range(len(events_list[m])):
            tmk = events_list[m][k]
            inter_sum = 0
            for v in range(M):
                v_less = events_list[v][events_list[v] < tmk]
                Rmvk = np.sum(np.exp(-beta * (tmk - v_less)))
                inter_sum += beta * alpha_array[m, v] * Rmvk
            third += np.log(mu_array[m] + inter_sum)
    print(f"detailed: {first} , {second} , {third}")
    return first + second + third

# sum of kernels - beta*alpha*exp(-beta*t) kernel
def detailed_LL_kernel_sum_2(params_array, C, betas, events_list, end_time, M, C_r=None):
    mu_array, alpha_matrix, _ = params_array
    Q = len(betas)
    # log-likelihood terms
    # first term
    first = -np.sum(mu_array) * end_time
    second = 0
    for m in range(M):
        for v in range(M):
            for q in range(len(betas)):
                T_diff_sum = np.sum(1 - np.exp(-betas[q] * (end_time - events_list[v])))
                if C_r is None:
                    second -= C[q] * alpha_matrix[m, v] * T_diff_sum
                else:
                    if m < (M / 2):
                        second -= C[q] * alpha_matrix[m, v] * T_diff_sum
                    else:
                        second -= C_r[q] * alpha_matrix[m, v] * T_diff_sum
    # third term
    third = 0
    for m in range(M):
        for k in range(len(events_list[m])):
            tmk = events_list[m][k]
            inter_sum = 0
            for v in range(M):
                v_less = events_list[v][events_list[v] < tmk]
                for q in range(Q):
                    Rmvk = np.sum(np.exp(-betas[q] * (tmk - v_less)))
                    if C_r is None:
                        inter_sum += C[q] * betas[q] * alpha_matrix[m, v] * Rmvk
                    else:
                        if m < (M / 2):
                            inter_sum += C[q] * betas[q] * alpha_matrix[m, v] * Rmvk
                        else:
                            inter_sum += C_r[q] * betas[q] * alpha_matrix[m, v] * Rmvk
            third += np.log(mu_array[m] + inter_sum)
    # print("detailed: ", first, second, third)
    return first + second + third

# %% Testing functions

def plot_alpha_lambda_matrix():
    # plot alpha matrix
    Na = 5
    nodes_a = list(range(Na))
    Nb = 3
    N = Na + Nb
    nodes_b = list(range(Na, Nb + Na))
    alphas_aa = (30, 35, 7, 5, 3, 3)
    alphas_ab = (40, 45, 20, 17, 15, 12)
    mu_a = 0.07
    mu_b = 0.05
    Maa = Na * (Na - 1)
    Mab2 = Na * Nb * 2
    Mbb = Nb * (Nb - 1)
    M = Maa + Mab2 + Mbb
    alpha = np.zeros((M, M))
    alpha_ab_ba = get_alpha_n_r_br_gr_al_alr_off(alphas_ab, alphas_ab, Na, Nb)
    alpha_aa = get_alpha_n_r_br_gr_al_alr_dia(alphas_aa, Na)
    alpha_bb = get_alpha_n_r_br_gr_al_alr_dia(alphas_aa, Nb)
    alpha[0:Maa, 0:Maa] = alpha_aa
    alpha[Maa:Mab2 + Maa, Maa:Mab2 + Maa] = alpha_ab_ba
    alpha[Mab2 + Maa:, Mab2 + Maa:] = alpha_bb
    mu = np.array([mu_a] * Maa + [mu_b] * Mab2 + [mu_a] * Mbb)
    inverse = np.linalg.inv(np.eye(M) - alpha)
    lambda_vector = alpha @ mu
    node_pairs = np_list(nodes_a, nodes_a) + np_list(nodes_a, nodes_b) + np_list(nodes_b, nodes_a) + np_list(nodes_b, nodes_b)
    lambda_matrix = np.zeros((N, N))
    for idx, (u, v) in enumerate(node_pairs):
        lambda_matrix[u, v] = lambda_vector[idx]

    # alpha matrix plot
    fig, ax = plt.subplots(figsize=(6, 4))
    plot = ax.pcolor(alpha, cmap='Greys')
    ax.set_xticks([Maa / 2, Maa + Mab2 / 4, Maa + Mab2 / 2 + Mab2 / 4, Maa + Mab2 + Mbb / 2])
    ax.set_xticklabels(['bp(a,a)', 'bp(a,b)', 'bp(b,a)', 'bp(b,b)'])
    ax.set_yticks([Maa / 2, Maa + Mab2 / 4, Maa + Mab2 / 2 + Mab2 / 4, Maa + Mab2 + Mbb / 2])
    ax.set_yticklabels(['bp(a,a)', 'bp(a,b)', 'bp(b,a)', 'bp(b,b)'])
    ax.invert_yaxis()
    fig.colorbar(plot, ax=ax)
    fig.tight_layout()
    fig.savefig(f"/shared/Results/MultiBlockHawkesModel/excitation_matrix.pdf")
    plt.show()

    fig1, ax1 = plt.subplots(figsize=(5, 4))
    plot = ax1.pcolor(lambda_matrix, cmap='Greys')
    ax1.set_xticks(list(np.arange(N) + 0.5), minor=False)
    ax1.set_xticklabels(list(np.arange(N) + 1))
    ax1.set_yticks(list(np.arange(N) + 0.5))
    ax1.set_yticklabels(list(np.arange(N) + 1))
    ax1.set_xticks(list(np.arange(N)), minor=True)
    ax1.set_yticks(list(np.arange(N)), minor=True)
    ax1.grid(which='minor', color="w", linestyle='-', linewidth=2)
    ax1.set_xlabel('nodes')
    ax1.set_ylabel('nodes')
    ax1.invert_yaxis()
    fig1.colorbar(plot, ax=ax1)
    fig1.tight_layout()
    fig1.savefig(f"/shared/Results/MultiBlockHawkesModel/lambda_matrix.pdf")
    plt.show()

def test_simulate_fit_one_beta_rho():
    param_ab = (0.001, .2, 0.6, 1)  # assuming that beta_ab = beta_ba
    beta = param_ab[3]
    rho_sim = param_ab[2] / param_ab[1]
    # param_ba = (0.0009, .1, 0.1*rho_sim, beta)  # assuming that beta_ab = beta_ba
    param_ba = (0, 0, 0*rho_sim, beta)  # assuming that beta_ab = beta_ba
    param_rho = (param_ab[0], param_ba[0], param_ab[1], param_ba[1], rho_sim, beta)
    a_nodes = list(np.arange(0, 30))
    b_nodes = list(np.arange(30, 60))
    end_time_sim = 2000
    M = len(a_nodes) * len(b_nodes)
    print("Block pair ab parameters:")
    print(f"mu={param_ab[0]}, alpha_n={param_ab[1]}, alpha_r={param_ab[2]}, beta={param_ab[3]}")
    print("Block pair ba parameters:")
    print(f"mu={param_ba[0]}, alpha_n={param_ba[1]}, alpha_r={param_ba[2]}, beta={param_ba[3]}")
    print("#nodes_a = ", len(a_nodes), ",\t#nodes_b = ", len(b_nodes), "\tsim time = ", end_time_sim)
    l, d_ab, d_ba = simulate_one_beta_off_2(param_ab, param_ba, a_nodes, b_nodes, end_time_sim, return_list=True)
    print(f"number of events = {cal_num_events_2(d_ab)} + {cal_num_events_2(d_ba)} = {cal_num_events_2(d_ba)+cal_num_events_2(d_ab)}")
    # true parameters log-likelihood
    ll_ab = LL_n_r_one_off_2(param_ab, d_ab, d_ba, end_time_sim, len(b_nodes), M)
    ll_ba = LL_n_r_one_off_2(param_ba, d_ba, d_ab, end_time_sim, len(a_nodes), M)
    # param_array_actual = get_array_param_n_r_off(param_ab, param_ba, len(a_nodes), len(b_nodes))
    # ll_detailed = log_likelihood_detailed_2(param_array_actual, l, end_time_sim, 2 * M)
    ll_rho = LL_n_r_one_off_rho(param_rho, d_ab, d_ba, end_time_sim, len(a_nodes), len(b_nodes))
    print(f"ll = {ll_ab:.2f}+{ll_ba:.2f} ={ll_ab + ll_ba:.2f}, ll_rho={ll_rho:.2f}")

    # jac check
    grad = False
    if grad:
        Ris_ab = cal_R_n_r_off(d_ab, d_ba, beta)
        Ris_ba = cal_R_n_r_off(d_ba, d_ab, beta)
        events_array_ab = list(d_ab.values())
        events_array_ba = list(d_ba.values())
        diff_sum_ab = np.sum(1 - np.exp(-beta * (end_time_sim - np.concatenate(events_array_ab))))
        diff_sum_ba = np.sum(1 - np.exp(-beta * (end_time_sim - np.concatenate(events_array_ba))))
        eps = np.sqrt(np.finfo(float).eps)
        print("NLL gradients (d_mu, d_alpha_n, d_alpha_r) - blockpair(b,a)")
        print("Approximation of the gradient - (a, b)")
        print(approx_fprime(param_ba[0:3], NLL_n_r_one_off_2, eps, d_ba, d_ab, end_time_sim, len(a_nodes), M, beta, Ris_ba, diff_sum_ba,
                            diff_sum_ab))
        print("Actual gradient - (a, b)")
        print(NLL_n_r_one_off_jac_2(param_ba[0:3], d_ba, d_ab, end_time_sim, len(a_nodes), M, beta, Ris_ba, diff_sum_ba, diff_sum_ab))
        print("Approximation of the gradient - rho")
        print(approx_fprime(param_rho[0:-1], NLL_n_r_one_off_rho, eps, d_ab, d_ba, end_time_sim, len(a_nodes), len(b_nodes), beta,
                            Ris_ab, Ris_ba, diff_sum_ab, diff_sum_ba))
        print("Actual gradient - rho")
        print(NLL_n_r_one_off_jac_rho(param_rho[0:-1], d_ab, d_ba, end_time_sim, len(a_nodes), len(b_nodes), beta,
                        Ris_ab, Ris_ba, diff_sum_ab, diff_sum_ba))

    # # fitting off-diagonal pair (n, r)
    start_fit_time = time.time()
    param_est_ab = fit_n_r_one_off_2(d_ab, d_ba, end_time_sim, len(b_nodes), M, beta)
    end_fit_time = time.time()
    print("Estimated parameters of block pair (a, b):")
    print(f"\tmu={param_est_ab[0]}, alpha_n={param_est_ab[1]}, alpha_r={param_est_ab[2]}")
    print(f"\tfit time = {(end_fit_time - start_fit_time):.4f} s")
    ll_est_ab = LL_n_r_one_off_2(param_est_ab, d_ab, d_ba, end_time_sim, len(b_nodes), M)
    print("\tlog-likelihood= ", ll_est_ab)
    print("Estimated parameters of block pair (b, a):")
    start_fit_time = time.time()
    param_est_ba = fit_n_r_one_off_2(d_ba, d_ab, end_time_sim, len(a_nodes), M, beta)
    end_fit_time = time.time()
    print(f"\tmu={param_est_ba[0]}, alpha_n={param_est_ba[1]}, alpha_r={param_est_ba[2]}")
    print(f"\tfit time = {(end_fit_time - start_fit_time):.4f} s")
    ll_est_ba = LL_n_r_one_off_2(param_est_ba, d_ba, d_ab, end_time_sim, len(a_nodes), M)
    print("\tlog-likelihood= ", ll_est_ba)
    print("fitted ll of both = ", ll_est_ab + ll_est_ba)
    print("\nEstimated constrained paramters for both (a, b) & (b, a)")
    start_fit_time = time.time()
    param_est = fit_n_r_one_off_rho(d_ab, d_ba, end_time_sim, len(a_nodes), len(b_nodes), beta)
    end_fit_time = time.time()
    print(f"\t(a, b): mu={param_est[0]}, alpha_n={param_est[2]}")
    print(f"\t(b, a): mu={param_est[1]}, alpha_n={param_est[3]}")
    print(f"\trho={param_est[4]}")
    print(f"\tfit time = {(end_fit_time - start_fit_time):.4f} s")
    ll_est = LL_n_r_one_off_rho(param_est, d_ab, d_ba, end_time_sim, len(a_nodes), len(b_nodes))
    print("\tlog-likelihood= ", ll_est)

def test_simulate_fit_one_beta_dia_n_r_br_gr_al_alr():
    # # Hawkes parameters
    beta = 5
    sim_param = (0.001, 0.3, 0.4, 0.0005, 0.0001, 0.0003, 0.0002, beta)
    # netwerk parameters
    N = 50   # number of nodes
    N_list = list(range(N))
    M = N*(N-1)
    sim_end_time = 2500

    # matrix parameters for simulation and detailed log-likelihood test
    print("simulating from (n, r, br, gr, al, alr) Block model with single known beta")
    sim_param_matrix = get_array_param_n_r_br_gr_al_alr_dia(sim_param, N)
    sim_list, sim_dict = simulate_one_beta_dia_2(sim_param, N_list, sim_end_time, return_list=True)
    print(f"#nodes={N}, #events={cal_num_events_2(sim_dict)}")
    # log-likelihood function
    ll_model = LL_n_r_br_gr_al_alr_one_dia(sim_param, sim_dict, sim_end_time, N, M)
    # ll_detailed = log_likelihood_detailed_2(sim_param_matrix, sim_list, sim_end_time, M)
    ll_detailed = 0
    print(f"model LL={ll_model:.2f}, detailed LL={ll_detailed:.2f}")

    # log-likelihood jac function check
    diff_sum = cal_diff_sum(sim_dict, sim_end_time, beta)
    Ris = cal_R_n_r_br_gr_al_alr_dia(sim_dict, beta)
    eps = np.sqrt(np.finfo(float).eps)
    print("LL derivates")
    print("Finite-difference approximation of the gradient")
    print(approx_fprime(sim_param[:-1], NLL_n_r_br_gr_al_alr_one_dia, eps, sim_dict, sim_end_time, N, M, beta, Ris, diff_sum))
    print("Actual gradient")
    print(NLL_n_r_br_gr_al_alr_one_dia_jac(sim_param[:-1], sim_dict, sim_end_time, N, M, beta, Ris, diff_sum))

    # fitting (n, r) global (br, gr) model with single known beta
    print("\nfitting (n, r) global (br, gr, al, alr) model with single known beta")
    start_fit_time = time.time()
    param_est = fit_n_r_br_gr_al_alr_one_dia(sim_dict, sim_end_time, N, M, beta)
    end_fit_time = time.time()
    for pr in range(len(param_est)):
        print(param_est[pr])
    # elapsed time
    print(f"time to fit = {(end_fit_time - start_fit_time):.4f} s")
    ll_est = LL_n_r_br_gr_al_alr_one_dia(param_est, sim_dict, sim_end_time, N, M)
    print("log-likelihood= ", ll_est)

def test_simulate_fit_one_beta_off_n_r_br_gr_al_alr():
    sim_p_ab = (0.001, 0.4, 0.2, 0.03, 0.0001, 0.002, 0.0001, 5)
    sim_p_ba = (0.002, 0.1, 0.5, 0.0003, 0.001, 0.0002, 0.001, 5)
    beta = sim_p_ab[-1]
    # simulation
    n_a, n_b = 25, 14
    n_a_list, n_b_list = list(range(n_a)), list(range(n_a, n_a + n_b))
    T = 2000
    M_ab = n_a * n_b
    print("simulating from (n, r, br, gr, al, alr) Block model with single known beta")
    print("Actual parameters:")
    print_param(sim_p_ab)
    print_param(sim_p_ba)
    sim_list, sim_dict_ab, sim_dict_ba = simulate_one_beta_off_2(sim_p_ab, sim_p_ba, n_a_list, n_b_list, T, return_list=True)
    print(f"n_a={n_a}, n_b={n_b}, #events_ab={cal_num_events_2(sim_dict_ab)} ,#events_ba={cal_num_events_2(sim_dict_ba)}")
    # log-likelihood function
    ll_ab = LL_n_r_br_gr_al_alr_one_off(sim_p_ab, sim_dict_ab, sim_dict_ba, T, n_b, M_ab)
    ll_ba = LL_n_r_br_gr_al_alr_one_off(sim_p_ba, sim_dict_ba, sim_dict_ab, T, n_a, M_ab)
    sim_param_matrix = get_array_param_n_r_br_gr_al_alr_off(sim_p_ab, sim_p_ba, n_a, n_b)
    # ll_detailed = log_likelihood_detailed_2(sim_param_matrix, sim_list, T, 2 * M_ab)
    ll_detailed = 0
    print(f"LL_model={ll_ab:.2f} + {ll_ba:.2f} = {ll_ab + ll_ba}, detailed LL={ll_detailed:.2f}")

    # log-likelihood jac function check
    diff_sum = cal_diff_sum(sim_dict_ab, T, sim_p_ab[-1])
    diff_sum_r = cal_diff_sum(sim_dict_ba, T, sim_p_ba[-1])
    Ris = cal_R_n_r_br_gr_al_alr_off(sim_dict_ab, sim_dict_ba, beta)
    eps = np.sqrt(np.finfo(float).eps)
    print("LL derivates")
    print("Finite-difference approximation of the gradient")
    print(approx_fprime(sim_p_ab[:-1], NLL_n_r_br_gr_al_alr_one_off, eps, sim_dict_ab, sim_dict_ba, T, n_b, M_ab, beta, Ris, diff_sum,
                        diff_sum_r))
    print("Actual gradient")
    print(NLL_n_r_br_gr_al_alr_one_off_jac(sim_p_ab[:-1], sim_dict_ab, sim_dict_ba, T, n_b, M_ab, beta, Ris, diff_sum, diff_sum_r))

    # fit to simulated data
    print("\nfitting one beta off-diagonal block pair (a, b)")
    start_fit_time = time.time()
    param_est_ab = fit_n_r_br_gr_al_alr_one_off(sim_dict_ab, sim_dict_ba, T, n_b, M_ab, beta)
    end_fit_time = time.time()
    print_param(param_est_ab)
    # elapsed time and estimated parameters log-likelihood
    ll_est_ab = LL_n_r_br_gr_al_alr_one_off(param_est_ab, sim_dict_ab, sim_dict_ba, T, n_b, M_ab)
    print(f"time to fit = {(end_fit_time - start_fit_time):.4f} s \t log-likelihood= {ll_est_ab:.4f}")

    print("\nfitting off-diagonal block pair ba (n, r, br, gr)")
    start_fit_time = time.time()
    param_est_ba = fit_n_r_br_gr_al_alr_one_off(sim_dict_ba, sim_dict_ab, T, n_a, M_ab, beta)
    end_fit_time = time.time()
    print_param(param_est_ba)
    # elapsed time and estimated parameters log-likelihood
    ll_est_ba = LL_n_r_br_gr_al_alr_one_off(param_est_ba, sim_dict_ba, sim_dict_ab, T, n_a, M_ab)
    print(f"time to fit = {(end_fit_time - start_fit_time):.4f} s \t log-likelihood= {ll_est_ba:.4f}")
    print("fitted ll of both = ", ll_est_ab + ll_est_ba)

def test_simulate_fit_one_beta_dia_n_r_br_gr():
    params = (0.002, .2, 0.3, 0.02, 0.004, 5)
    p_jac = (0.002, .2, 0.3, 0.02, 0.004)
    beta = params[5]
    a_nodes = list(range(15))
    n_nodes = len(a_nodes)
    end_time_sim = 3000
    print("Simulate and fit diagonal pair with alpha (n, r, br, gr) parameters")
    print("simulation #nodes = ", n_nodes, " Simulation duration = ", end_time_sim)
    print("actual parameters:")
    print_param(params)
    M = n_nodes * (n_nodes - 1)
    # n r br gr diagonal block simulation
    events_list_dia, events_dict_dia = simulate_one_beta_dia_2(params, a_nodes, end_time_sim, return_list=True)
    print("simulated events: ", cal_num_events_2(events_dict_dia))
    # Actual parameters log-likelihood - Two ways
    ll = LL_n_r_br_gr_one_dia_2(params, events_dict_dia, end_time_sim, n_nodes, M)
    param_array_actual = get_array_param_n_r_br_gr_dia(params, n_nodes)
    ll_detailed = log_likelihood_detailed_2(param_array_actual, events_list_dia, end_time_sim, M)
    print(f"ll = {ll}, detailed = {ll_detailed}")

    # check jacobian
    Ris = cal_R_n_r_br_gr_dia(events_dict_dia, beta)
    events_array = list(events_dict_dia.values())
    diff_sum = np.sum(1 - np.exp(-beta * (end_time_sim - np.concatenate(events_array))))
    eps = np.sqrt(np.finfo(float).eps)
    print("Derivates of the log-likelihood function for one diagonal block pair")
    print("Finite-difference approximation of the gradient")
    print(approx_fprime(p_jac, NLL_n_r_br_gr_one_dia_2, eps, events_dict_dia, end_time_sim, n_nodes, M,
                      beta, Ris, diff_sum))
    print("Actual gradient")
    print(NLL_n_r_br_gr_one_dia_jac_2(p_jac, events_dict_dia, end_time_sim, n_nodes, M, beta, Ris, diff_sum))

    # fitting diagonal pair (n, r, br, gr) model with one known beta
    print("\nfitting one known beta")
    start_fit_time = time.time()
    param_est = fit_n_r_br_gr_one_dia_2(events_dict_dia, end_time_sim, n_nodes, M, beta)
    end_fit_time = time.time()
    print_param(param_est)
    # elapsed time
    print(f"time to fit = {(end_fit_time - start_fit_time):.4f} s")
    ll_est = LL_n_r_br_gr_one_dia_2(param_est, events_dict_dia, end_time_sim, n_nodes, M)
    print("log-likelihood= ", ll_est)

def test_simulate_fit_one_beta_off_n_r_br_gr():
    # two off-diagonal block pais simulation
    param_ab = (0.0002, .1, 0.3, 0.01, 0.006, 5)  # assuming that beta_ab = beta_ba
    param_ba = (0.0001, .2, 0.1, 0.009, 0.005, 5)  # assuming that beta_ab = beta_ba
    beta = param_ab[5]
    a_nodes = list(np.arange(0, 20))
    b_nodes = list(np.arange(20, 50))
    end_time_sim = 2000
    M = len(a_nodes) * len(b_nodes)
    print("Block pair ab parameters:")
    print_param(param_ab)
    print("Block pair ba parameters:")
    print_param(param_ba)
    print("number of nodes_a = ", len(a_nodes), ", nodes_b = ", len(b_nodes), ", Duration = ", end_time_sim)
    l, d_ab, d_ba = simulate_one_beta_off_2(param_ab, param_ba, a_nodes, b_nodes, end_time_sim, return_list=True)
    # true paramters log-likelihoods
    ll_ab = LL_n_r_br_gr_one_off_2(param_ab, d_ab, d_ba, end_time_sim, len(b_nodes), M)
    ll_ba = LL_n_r_br_gr_one_off_2(param_ba, d_ba, d_ab, end_time_sim, len(a_nodes), M)
    param_array_actual = get_array_param_n_r_br_gr_off(param_ab, param_ba, len(a_nodes), len(b_nodes))
    ll_detailed = log_likelihood_detailed_2(param_array_actual, l, end_time_sim, 2 * M)
    print(f"ll = {ll_ab}+{ll_ba} = {ll_ab + ll_ba}, detailed = {ll_detailed}")
    # jacobian check
    Ris_ab = cal_R_n_r_br_gr_off(d_ab, d_ba, beta)
    Ris_ba = cal_R_n_r_br_gr_off(d_ba, d_ab, beta)
    events_array_ab = list(d_ab.values())
    events_array_ba = list(d_ba.values())
    diff_sum_ab = np.sum(1 - np.exp(-beta * (end_time_sim - np.concatenate(events_array_ab))))
    diff_sum_ba = np.sum(1 - np.exp(-beta * (end_time_sim - np.concatenate(events_array_ba))))
    eps = np.sqrt(np.finfo(float).eps)
    print("NLL gradients (d_mu, d_alpha_n, d_alpha_r, d_alpha_br, d_alpha_gr) - blockpair(b,a)")
    print("Finite-difference approximation of the gradient")
    print(approx_fprime(param_ba[0:5], NLL_n_r_br_gr_one_off_2, eps, d_ba, d_ab, end_time_sim, len(a_nodes), M, beta,
                        Ris_ba, diff_sum_ba, diff_sum_ab))
    print("Actual gradient")
    print(NLL_n_r_br_gr_one_off_jac_2(param_ba[0:5], d_ba, d_ab, end_time_sim, len(a_nodes), M, beta, Ris_ba, diff_sum_ba, diff_sum_ab))
    # fitting off-diagonal pair (n, r, br, gr)
    print("\nfitting one beta off-diagonal block pair ab (n, r, br, gr)")
    start_fit_time = time.time()
    param_est_ab = fit_n_r_br_gr_one_off_2(d_ab, d_ba, end_time_sim, len(b_nodes), M, beta)
    end_fit_time = time.time()
    print_param(param_est_ab)
    # elapsed time and estimated parameters log-likelihood
    ll_est_ab = LL_n_r_br_gr_one_off_2(param_est_ab, d_ab, d_ba, end_time_sim, len(b_nodes), M)
    print(f"time to fit = {(end_fit_time - start_fit_time):.4f} s \t log-likelihood= {ll_est_ab:.4f}")

    print("\nfitting off-diagonal block pair ba (n, r, br, gr)")
    start_fit_time = time.time()
    param_est_ba = fit_n_r_br_gr_one_off_2(d_ba, d_ab, end_time_sim, len(a_nodes), M, beta)
    end_fit_time = time.time()
    print_param(param_est_ba)
    # elapsed time and estimated parameters log-likelihood
    ll_est_ba = LL_n_r_br_gr_one_off_2(param_est_ba, d_ba, d_ab, end_time_sim, len(a_nodes), M)
    print(f"time to fit = {(end_fit_time - start_fit_time):.4f} s \t log-likelihood= {ll_est_ba:.4f}")
    print("fitted ll of both = ", ll_est_ab + ll_est_ba)

def test_simulate_fit_one_beta_dia_n_r():
    params = (0.002, .2, 0.3, 5)
    beta = params[3]
    a_nodes = list(range(15))
    n_nodes = len(a_nodes)
    end_time_sim = 3000
    print("actual parameters:")
    print(f"mu={params[0]}, alpha_n={params[1]}, alpha_r={params[2]}, beta={params[3]}")
    print("number of nodes = ", n_nodes, " sim time = ", end_time_sim)
    M = n_nodes * (n_nodes - 1)
    # n r br gr diagonal block simulation
    events_list_dia, events_dict_dia = simulate_one_beta_dia_2(params, a_nodes, end_time_sim, return_list=True)

    # Actual parameters log-likelihood
    ll = LL_n_r_one_dia_2(params, events_dict_dia, end_time_sim, n_nodes, M)
    param_array_actual = get_array_param_n_r_dia(params, n_nodes)
    ll_detailed = log_likelihood_detailed_2(param_array_actual, events_list_dia, end_time_sim, M)
    print(f"ll = {ll}, detailed = {ll_detailed}")

    # check jac function
    # check jacobian
    Ris = cal_R_n_r_dia(events_dict_dia, beta)
    events_array = list(events_dict_dia.values())
    diff_sum = np.sum(1 - np.exp(-beta * (end_time_sim - np.concatenate(events_array))))
    eps = np.sqrt(np.finfo(float).eps)
    print("Derivates of the log-likelihood function for one diagonal block pair")
    print("Finite-difference approximation of the gradient")
    print(approx_fprime(params[0:3], NLL_n_r_one_dia_2, eps, events_dict_dia, end_time_sim, n_nodes, M, beta, Ris, diff_sum))
    print("Actual gradient")
    print(NLL_n_r_one_dia_jac_2(params[0:3], events_dict_dia, end_time_sim, n_nodes, M, beta, Ris, diff_sum))

    # # fitting diagonal pair (n, r)
    print("\nfitting diagonal pair (n, r)")
    start_fit_time = time.time()
    param_est = fit_n_r_one_dia_2(events_dict_dia, end_time_sim, n_nodes, M, beta)
    end_fit_time = time.time()
    print("Estimated parameters:")
    print("\tmu = ", param_est[0], "\t\tbeta = ", param_est[3])
    print("\talpha_n = ", param_est[1], "\t\talpha_r = ", param_est[2])
    # elapsed time
    print(f"time to fit = {(end_fit_time - start_fit_time):.4f} s")
    ll_est = LL_n_r_one_dia_2(param_est, events_dict_dia, end_time_sim, n_nodes, M)
    print("fitted parameters log-likelihood= ", ll_est)

def test_simulate_fit_one_beta_off_n_r():
    # two off-diagonal block pais simulation
    param_ab = (0.0002, .1, 0.3, 5)  # assuming that beta_ab = beta_ba
    param_ba = (0.0001, .2, 0.1, 5)  # assuming that beta_ab = beta_ba
    beta = param_ab[3]
    a_nodes = list(np.arange(0, 20))
    b_nodes = list(np.arange(20, 50))
    end_time_sim = 2000
    M = len(a_nodes) * len(b_nodes)
    print("Block pair ab parameters:")
    print(f"mu={param_ab[0]}, alpha_n={param_ab[1]}, alpha_r={param_ab[2]}, beta={param_ab[3]}")
    print("Block pair ba parameters:")
    print(f"mu={param_ba[0]}, alpha_n={param_ba[1]}, alpha_r={param_ba[2]}, beta={param_ba[3]}")
    print("#nodes_a = ", len(a_nodes), ",\t#nodes_b = ", len(b_nodes), "\tsim time = ", end_time_sim)
    l, d_ab, d_ba = simulate_one_beta_off_2(param_ab, param_ba, a_nodes, b_nodes, end_time_sim, return_list=True)
    # true parameters log-likelihood
    ll_ab = LL_n_r_one_off_2(param_ab, d_ab, d_ba, end_time_sim, len(b_nodes), M)
    ll_ba = LL_n_r_one_off_2(param_ba, d_ba, d_ab, end_time_sim, len(a_nodes), M)
    param_array_actual = get_array_param_n_r_off(param_ab, param_ba, len(a_nodes), len(b_nodes))
    ll_detailed = log_likelihood_detailed_2(param_array_actual, l, end_time_sim, 2 * M)
    print(f"ll = {ll_ab}+{ll_ba} ={ll_ab + ll_ba}, detailed = {ll_detailed}")
    # jac check
    Ris_ab = cal_R_n_r_off(d_ab, d_ba, beta)
    Ris_ba = cal_R_n_r_off(d_ba, d_ab, beta)
    events_array_ab = list(d_ab.values())
    events_array_ba = list(d_ba.values())
    diff_sum_ab = np.sum(1 - np.exp(-beta * (end_time_sim - np.concatenate(events_array_ab))))
    diff_sum_ba = np.sum(1 - np.exp(-beta * (end_time_sim - np.concatenate(events_array_ba))))
    eps = np.sqrt(np.finfo(float).eps)
    print("NLL gradients (d_mu, d_alpha_n, d_alpha_r) - blockpair(b,a)")
    print("Finite-difference approximation of the gradient")
    print(approx_fprime(param_ba[0:3], NLL_n_r_one_off_2, eps, d_ba, d_ab, end_time_sim, len(a_nodes), M, beta, Ris_ba, diff_sum_ba,
                        diff_sum_ab))
    print("Actual gradient")
    print(NLL_n_r_one_off_jac_2(param_ba[0:3], d_ba, d_ab, end_time_sim, len(a_nodes), M, beta, Ris_ba, diff_sum_ba, diff_sum_ab))
    # # fitting off-diagonal pair (n, r)
    print("\nfitting off-diagonal block pair ab (n, r)")
    start_fit_time = time.time()
    param_est_ab = fit_n_r_one_off_2(d_ab, d_ba, end_time_sim, len(b_nodes), M, beta)
    end_fit_time = time.time()
    print("Estimated parameters of block pair ab:")
    print(f"mu={param_est_ab[0]}, alpha_n={param_est_ab[1]}, alpha_r={param_est_ab[2]}, beta={param_est_ab[3]}")
    print(f"time to fit = {(end_fit_time - start_fit_time):.4f} s")
    ll_est_ab = LL_n_r_one_off_2(param_est_ab, d_ab, d_ba, end_time_sim, len(b_nodes), M)
    print("fitted parameters log-likelihood= ", ll_est_ab)
    print("\nfitting off-diagonal block pair ba (n, r)")
    start_fit_time = time.time()
    param_est_ba = fit_n_r_one_off_2(d_ba, d_ab, end_time_sim, len(a_nodes), M, beta)
    end_fit_time = time.time()
    print("Estimated parameters of block pair ba:")
    print(f"mu={param_est_ba[0]}, alpha_n={param_est_ba[1]}, alpha_r={param_est_ba[2]}, beta={param_est_ba[3]}")
    print(f"time to fit = {(end_fit_time - start_fit_time):.4f} s")
    ll_est_ba = LL_n_r_one_off_2(param_est_ba, d_ba, d_ab, end_time_sim, len(a_nodes), M)
    print("fitted parameters log-likelihood= ", ll_est_ba)
    print("fitted ll of both = ", ll_est_ab + ll_est_ba)

def test_simulate_fit_sum_kernels_al_dia():
    betas = np.array([.01, 0.1, 15])
    C = np.array([0.3, 0.3, 0.4])
    param_sim_k = (0.001, .2, 0.3, 0.02, 0.004, 0.01, 0.0001, C, betas)
    param_sim_d = (0.001, .2, 0.3, 0.02, 0.004, 0.01, 0.0001, 999)
    p = (0.001, .2, 0.3, 0.02, 0.004, 0.01, 0.0001, 0.3, 0.3, 0.4)
    a_nodes = list(range(15))
    n_nodes = len(a_nodes)
    end_time_sim = 3000

    print(f"Simulate and fit diagonal sum of kernels (6 alphas) at betas = {betas}")
    print("simulation #nodes = ", n_nodes, " Simulation duration = ", end_time_sim)
    M = n_nodes * (n_nodes - 1)
    # sum of kernels diagonal block simulation
    events_list_dia, events_dict_dia = simulate_kernel_sum_dia_2(param_sim_k, a_nodes, end_time_sim, return_list=True)
    n_events = cal_num_events_2(events_dict_dia)
    print("number of events = ", n_events)
    print("actual parameters:")
    print_param_kernels(param_sim_k)

    # Actual parameters log-likelihood
    ll_sum = LL_6_alpha_kernel_sum_dia(param_sim_k, events_dict_dia, end_time_sim, n_nodes, M)
    # Actual parameters detailed log-likelihood (two values should match up)
    param_array = get_array_param_n_r_br_gr_al_alr_dia(param_sim_d, n_nodes)
    # ll_detailed = detailed_LL_kernel_sum_2(param_array, C, betas, events_list_dia, end_time_sim, M)
    ll_detailed = 0
    print(f"ll_sum = {ll_sum}, detailed = {ll_detailed}")

    # test NLL jacobian function
    Ris = cal_R_6_alpha_kernel_sum_dia(events_dict_dia, betas)
    T_diff_sums = cal_diff_sums_Q(events_dict_dia, end_time_sim, betas)
    eps = np.sqrt(np.finfo(float).eps)
    print("Derivates of NLL function Sum of kernels (6-alphas)")
    print("Finite-difference approximation of the gradient")
    print(approx_fprime(p, NLL_6_alpha_kernel_sum_dia, eps, betas, events_dict_dia, end_time_sim, n_nodes, M, T_diff_sums, Ris))
    print("Analytical gradient")
    print(NLL_6_alpha_kernel_sum_dia_jac(p, betas, events_dict_dia, end_time_sim, n_nodes, M, T_diff_sums, Ris))

    # fit Sum of kernels diagonal block pair
    start_fit_time = time.time()
    est_params = fit_6_alpha_kernel_sum_dia(events_dict_dia, end_time_sim, n_nodes, M, betas)
    end_fit_time = time.time()
    print_param_kernels(est_params)
    estimated_ll = LL_6_alpha_kernel_sum_dia(est_params, events_dict_dia, end_time_sim, n_nodes, M)
    print(f"estimated ll = {estimated_ll}, time to fit = {(end_fit_time - start_fit_time):.4f} s")

def test_simulate_fit_sum_kernels_al_off():
    # test simulation from sum of kernels (6 alphas)
    print("Simulate and fit (sum of kernels) two off-diagonal bp - (6 alphas)")
    betas = np.array([0.01, 0.2, 15])
    C = np.array([0.2, 0.3, 0.5])
    C_r = np.array([0.4, 0.3, 0.3])
    p_ab = [0.001, 0.4, 0.3, 0.02, 0.002, 0.01, 0.001, C, betas]
    p_ba = [0.002, 0.2, 0.4, 0.01, 0.01, 0.03, 0.003, C_r, betas]
    p_ab_d = p_ab[:7] + [0.2, 0.3, 0.5]
    n_a, n_b = 12, 5
    a_nodes = list(np.arange(0, n_a))
    b_nodes = list(np.arange(n_a, n_b + n_a))
    end_time_sim = 3000
    M_ab = n_a * n_b
    print("Block pair ab parameters:")
    print_param_kernels(p_ab)
    print("Block pair ba parameters:")
    print_param_kernels(p_ba)
    print("number of nodes_a = ", n_a, ", nodes_b = ", n_b, ", Duration = ", end_time_sim)
    l, d_ab, d_ba = simulate_kernel_sum_off_2(p_ab, p_ba, a_nodes, b_nodes, end_time_sim, return_list=True)
    print("number of simulated events = ", cal_num_events_2(d_ab) + cal_num_events_2(d_ba))
    ll_ab_1 = LL_6_alpha_kernel_sum_off(p_ab, d_ab, d_ba, end_time_sim, n_b, M_ab)
    ll_ba_1 = LL_6_alpha_kernel_sum_off(p_ba, d_ba, d_ab, end_time_sim, n_a, M_ab)
    print(f"ll_sum = {ll_ab_1} + {ll_ba_1} = {ll_ab_1 + ll_ba_1}")
    param_array = get_array_param_n_r_br_gr_al_alr_off(p_ab[:7] + [999], p_ba[:7] + [999], n_a, n_b)
    ll_detailed = detailed_LL_kernel_sum_2(param_array, C, betas, l, end_time_sim, 2 * M_ab, C_r=C_r)
    print(f"detailed ll = ", ll_detailed)

    # test NLL jacobian function
    Ris = cal_R_6_alpha_kernel_sum_off(d_ab, d_ba, betas)
    T_diff_sums = cal_diff_sums_Q(d_ab, end_time_sim, betas)
    T_diff_sums_r = cal_diff_sums_Q(d_ba, end_time_sim, betas)
    eps = np.sqrt(np.finfo(float).eps)
    print("Finite-difference approximation of the gradient")
    print(approx_fprime(p_ab_d, NLL_6_alpha_kernel_sum_off, eps, betas, d_ab, d_ba, end_time_sim, n_b, M_ab, T_diff_sums, T_diff_sums_r, Ris))
    print("Analytical gradient")
    print(NLL_6_alpha_kernel_sum_off_jac(p_ab_d, betas, d_ab, d_ba, end_time_sim, n_b, M_ab, T_diff_sums, T_diff_sums_r, Ris))

    # fit sum of kernels on two reciprocal block pairs on actual and different betas
    for i in range(2):
        if i == 1: betas = [0.01, 20]
        print(f"\nfitting 2 sim recip bp using sum of kernels at {betas}")
        # fit block pair (a, b)
        start_fit_time = time.time()
        est_p_ab = fit_6_alpha_kernel_sum_off(d_ab, d_ba, end_time_sim, n_b, M_ab, betas)
        end_fit_time = time.time()
        print("Estimated parameters of block pair (a, b):")
        print_param_kernels(est_p_ab)
        ll_ab_est = LL_6_alpha_kernel_sum_off(est_p_ab, d_ab, d_ba, end_time_sim, n_b, M_ab)
        print(f"fit time = {(end_fit_time - start_fit_time):.4f} s, ll_ab = {ll_ab_est}")
        # fit block pair (b, a)
        start_fit_time = time.time()
        est_p_ba = fit_6_alpha_kernel_sum_off(d_ba, d_ab, end_time_sim, n_a, M_ab, betas)
        end_fit_time = time.time()
        print("Estimated parameters of block pair (b, a):")
        print_param_kernels(est_p_ba)
        ll_ba_est = LL_6_alpha_kernel_sum_off(est_p_ba, d_ba, d_ab, end_time_sim, n_a, M_ab)
        print(f"fit time = {(end_fit_time - start_fit_time):.4f} s, ll_ab = {ll_ba_est}")
        print("total ll = ", (ll_ab_est + ll_ba_est))

def test_simulate_fit_sum_kernels_dia():
    # test simulation from sum of kernels
    betas = np.array([.1, 1, 10])
    C = np.array(np.array([0.2, 0.3, 0.5]))
    param_sim_k = (0.003, .3, 0.04, 0.02, 0.004, C, betas)
    scale = np.sum(C / betas)
    param_sim_detailed = (0.003, .3, 0.04, 0.02, 0.004, 999)
    a_nodes = list(range(20))
    n_nodes = len(a_nodes)
    end_time_sim = 3000

    print(f"Simulate and fit diagonal sum of kernels (new kernel) at betas = {betas}")
    print("simulation #nodes = ", n_nodes, " Simulation duration = ", end_time_sim)
    M = n_nodes * (n_nodes - 1)
    # sum of kernels diagonal block simulation
    events_list_dia, events_dict_dia = simulate_kernel_sum_dia_2(param_sim_k, a_nodes, end_time_sim, return_list=True)
    n_events = cal_num_events_2(events_dict_dia)
    print("number of events = ", n_events)
    print("actual parameters:")
    print_param_kernels(param_sim_k)

    # Actual parameters log-likelihood
    ll_sum = LL_4_alpha_kernel_sum_dia(param_sim_k, events_dict_dia, end_time_sim, n_nodes, M)
    # Actual parameters detailed log-likelihood (two values should match up)
    param_array = get_array_param_n_r_br_gr_dia(param_sim_detailed, n_nodes)
    ll_detailed = detailed_LL_kernel_sum_2(param_array, C, betas, events_list_dia, end_time_sim, M)
    print(f"ll_sum = {ll_sum}, detailed = {ll_detailed}")

    # fit Sum of kernels diagonal block pair
    start_fit_time = time.time()
    est_params = fit_4_alpha_kernel_sum_dia(events_dict_dia, end_time_sim, n_nodes, M, betas)
    end_fit_time = time.time()
    print_param_kernels(est_params)
    estimated_ll = LL_4_alpha_kernel_sum_dia(est_params, events_dict_dia, end_time_sim, n_nodes, M)
    print(f"estimated ll = {estimated_ll}, time to fit = {(end_fit_time - start_fit_time):.4f} s")

def test_simulate_fit_sum_kernels_off():
    print("Simulate and fit (sum of kernels) two off-diagonal bp - new kernel")
    betas_actual = np.array([0.01, 0.2, 15])
    C = np.array([0.2, 0.3, 0.5])
    C_r = np.array([0.4, 0.3, 0.3])
    param_sum_ab = (0.001, .4, 0.3, 0.02, 0.007, C, betas_actual)
    param_sum_ba = (0.002, 0.3, 0.5, 0.02, 0.01, C_r, betas_actual)
    a_nodes = list(np.arange(0, 13))
    b_nodes = list(np.arange(13, 18))
    end_time_sim = 2000
    M = len(a_nodes) * len(b_nodes)
    print("Block pair ab parameters:")
    print_param_kernels(param_sum_ab)
    print("Block pair ba parameters:")
    print_param_kernels(param_sum_ba)
    print("number of nodes_a = ", len(a_nodes), ", nodes_b = ", len(b_nodes), ", Duration = ", end_time_sim)
    l, d_ab, d_ba = simulate_kernel_sum_off_2(param_sum_ab, param_sum_ba, a_nodes, b_nodes, end_time_sim, return_list=True)
    print("number of simulated events = ", cal_num_events_2(d_ab) + cal_num_events_2(d_ba))
    ll_ab_1 = LL_4_alpha_kernel_sum_off(param_sum_ab, d_ab, d_ba, end_time_sim, len(b_nodes), M)
    ll_ba_1 = LL_4_alpha_kernel_sum_off(param_sum_ba, d_ba, d_ab, end_time_sim, len(a_nodes), M)
    print(f"ll_sum = {ll_ab_1} + {ll_ba_1} = {ll_ab_1 + ll_ba_1}")
    param_array = get_array_param_n_r_br_gr_off((0.001, .4, 0.3, 0.02, 0.007, 99), (0.002, 0.3, 0.5, 0.02, 0.01, 99), len(a_nodes),
                                                len(b_nodes))
    ll_detailed = detailed_LL_kernel_sum_2(param_array, C, betas_actual, l, end_time_sim, 2 * M, C_r=C_r)
    print(f"detailed ll = ", ll_detailed)

    # fit sum of kernels on two reciprocal block pairs on actual betas
    print("\nfitting 2 simulated reciprocal block pairs using sum of kernels")
    # fit block pair (a, b)
    start_fit_time = time.time()
    est_p_ab = fit_4_alpha_kernel_sum_off(d_ab, d_ba, end_time_sim, len(b_nodes), M, betas_actual)
    end_fit_time = time.time()
    print("Estimated parameters of block pair (a, b):")
    print_param_kernels(est_p_ab)
    ll_ab_est = LL_4_alpha_kernel_sum_off(est_p_ab, d_ab, d_ba, end_time_sim, len(b_nodes), M)
    print(f"fit time = {(end_fit_time - start_fit_time):.4f} s, ll_ab = {ll_ab_est}")
    # fit block pair (b, a)
    start_fit_time = time.time()
    est_p_ba = fit_4_alpha_kernel_sum_off(d_ba, d_ab, end_time_sim, len(a_nodes), M, betas_actual)
    end_fit_time = time.time()
    print("Estimated parameters of block pair (b, a):")
    print_param_kernels(est_p_ba)
    ll_ba_est = LL_4_alpha_kernel_sum_off(est_p_ba, d_ba, d_ab, end_time_sim, len(a_nodes), M)
    print(f"fit time = {(end_fit_time - start_fit_time):.4f} s, ll_ab = {ll_ba_est}")
    print("total ll = ", (ll_ab_est + ll_ba_est))

    # fit sum of kernels on two reciprocal block pairs on different range
    betas = [0.01, 5, 30]
    print("\nfitting 2 simulated reciprocal block pairs using sum of kernels")
    # fit block pair (a, b)
    start_fit_time = time.time()
    est_p_ab = fit_4_alpha_kernel_sum_off(d_ab, d_ba, end_time_sim, len(b_nodes), M, betas)
    end_fit_time = time.time()
    print("Estimated parameters of block pair (a, b):")
    print_param_kernels(est_p_ab)
    ll_ab_est = LL_4_alpha_kernel_sum_off(est_p_ab, d_ab, d_ba, end_time_sim, len(b_nodes), M)
    print(f"fit time = {(end_fit_time - start_fit_time):.4f} s, ll_ab = {ll_ab_est}")
    # fit block pair (b, a)
    start_fit_time = time.time()
    est_p_ba = fit_4_alpha_kernel_sum_off(d_ba, d_ab, end_time_sim, len(a_nodes), M, betas)
    end_fit_time = time.time()
    print("Estimated parameters of block pair (b, a):")
    print_param_kernels(est_p_ba)
    ll_ba_est = LL_4_alpha_kernel_sum_off(est_p_ba, d_ba, d_ab, end_time_sim, len(a_nodes), M)
    print(f"fit time = {(end_fit_time - start_fit_time):.4f} s, ll_ab = {ll_ba_est}")
    print("total ll = ", (ll_ab_est + ll_ba_est))

def test_simulate_fit_sum_kernels_nr_off():
    print("Simulate and fit two off-diagonal pair with self & reciprocity (sum of kernels) parameters")
    betas_actual = np.array([0.05, 0.6, 20])
    # C = np.array([0.01, 0.19, 0.8])
    # # C_r = C # just to test detailed function
    # C_r = np.array([0.02, 0.28, 0.7])
    # param_sum_ab = (0.002, .5, 0.7, C, betas_actual)
    # param_sum_ba = (0.001, 0.3, 0.5, C_r, betas_actual)
    C = np.array([0.1, 0.3, 0.6])
    C_r = np.array([0.5, 0.4, 0.1])
    param_sum_ab = (0.002, .5, 0.6, C, betas_actual)
    param_sum_ba = (0.001, 0.2, 0.4, C_r, betas_actual)

    a_nodes = list(np.arange(0, 13))
    b_nodes = list(np.arange(13, 20))
    end_time_sim = 3500
    M = len(a_nodes) * len(b_nodes)
    print("Block pair ab parameters:")
    print_param_kernels(param_sum_ab)
    print("Block pair ba parameters:")
    print_param_kernels(param_sum_ba)
    print("number of nodes_a = ", len(a_nodes), ", nodes_b = ", len(b_nodes), ", Duration = ", end_time_sim)
    l, d_ab, d_ba = simulate_kernel_sum_off_2(param_sum_ab, param_sum_ba, a_nodes, b_nodes, end_time_sim, return_list=True)
    print("number of simulated events = ", cal_num_events_2(d_ab) + cal_num_events_2(d_ba))
    ll_ab_1 = LL_2_alpha_kernel_sum_off(param_sum_ab, d_ab, d_ba, end_time_sim, len(b_nodes), M)
    ll_ba_1 = LL_2_alpha_kernel_sum_off(param_sum_ba, d_ba, d_ab, end_time_sim, len(a_nodes), M)
    print(f"ll_sum = {ll_ab_1} + {ll_ba_1} = {ll_ab_1 + ll_ba_1}")
    param_array = get_array_param_n_r_off((0.002, .5, 0.6,99), (0.001, 0.2, 0.4,99), len(a_nodes), len(b_nodes))
    ll_detailed = detailed_LL_kernel_sum_2(param_array, C, betas_actual, l, end_time_sim, 2*M, C_r=C_r)
    print(f"detailed ll = ", ll_detailed)

    # jacobian test
    Q = len(betas_actual)
    # test jacobian of sum of kernels method for off-diagonal block pairs
    p_ab = (0.002, .5, 0.6, 0.1, 0.3, 0.6)
    # calculate fixed terms in log-likelihood
    Ris = cal_R_2_alpha_kernel_sum_off(d_ab, d_ba, betas_actual)
    events_array = list(d_ab.values())
    events_array_r = list(d_ba.values())
    T_diff_sums = np.zeros(Q, )
    T_diff_sums_r = np.zeros(Q, )
    if len(events_array) != 0:
        T_diff = end_time_sim - np.concatenate(events_array)
        for q in range(Q):
            T_diff_sums[q] = np.sum(1 - np.exp(-betas_actual[q] * T_diff))
    if len(events_array_r) != 0:
        T_diff = end_time_sim - np.concatenate(events_array_r)
        for q in range(Q):
            T_diff_sums_r[q] = np.sum(1 - np.exp(-betas_actual[q] * T_diff))
    eps = np.sqrt(np.finfo(float).eps)
    print("Finite-difference approximation of the gradient")
    print(
        approx_fprime(p_ab, NLL_2_alpha_kernel_sum_off, eps, betas_actual, d_ab, d_ba, end_time_sim, len(b_nodes), M, T_diff_sums, T_diff_sums_r,
                      Ris))
    print("Actual gradient")
    print(NLL_2_alpha_kernel_sum_off_jac(p_ab, betas_actual, d_ab, d_ba, end_time_sim, len(b_nodes), M, T_diff_sums, T_diff_sums_r, Ris))


    # fit sum of kernels on two reciprocal block pairs on actual betas
    print("\nfitting 2 simulated reciprocal block pairs using (self&recip) sum of kernels")
    # fit block pair (a, b)
    start_fit_time = time.time()
    est_p_ab = fit_2_alpha_kernel_sum_off(d_ab, d_ba, end_time_sim, len(b_nodes), M, betas_actual)
    end_fit_time = time.time()
    print("Estimated parameters of block pair (a, b):")
    print_param_kernels(est_p_ab)
    ll_ab_est = LL_2_alpha_kernel_sum_off(est_p_ab, d_ab, d_ba, end_time_sim, len(b_nodes), M)
    print(f"fit time = {(end_fit_time - start_fit_time):.4f} s, ll_ab = {ll_ab_est}")
    # fit block pair (b, a)
    start_fit_time = time.time()
    est_p_ba = fit_2_alpha_kernel_sum_off(d_ba, d_ab, end_time_sim, len(a_nodes), M, betas_actual)
    end_fit_time = time.time()
    print("Estimated parameters of block pair (b, a):")
    print_param_kernels(est_p_ba)
    ll_ba_est = LL_2_alpha_kernel_sum_off(est_p_ba, d_ba, d_ab, end_time_sim, len(a_nodes), M)
    print(f"fit time = {(end_fit_time - start_fit_time):.4f} s, ll_ab = {ll_ba_est}")
    print("total ll = ", (ll_ab_est + ll_ba_est))

    # fit sum of kernels on two reciprocal block pairs on different range
    betas = [0.01, 1, 50]
    print("\nfitting 2 simulated reciprocal block pairs using sum of kernels")
    # fit block pair (a, b)
    start_fit_time = time.time()
    est_p_ab = fit_2_alpha_kernel_sum_off(d_ab, d_ba, end_time_sim, len(b_nodes), M, betas)
    end_fit_time = time.time()
    print("Estimated parameters of block pair (a, b):")
    print_param_kernels(est_p_ab)
    ll_ab_est = LL_2_alpha_kernel_sum_off(est_p_ab, d_ab, d_ba, end_time_sim, len(b_nodes), M)
    print(f"fit time = {(end_fit_time - start_fit_time):.4f} s, ll_ab = {ll_ab_est}")
    # fit block pair (b, a)
    start_fit_time = time.time()
    est_p_ba = fit_2_alpha_kernel_sum_off(d_ba, d_ab, end_time_sim, len(a_nodes), M, betas)
    end_fit_time = time.time()
    print("Estimated parameters of block pair (b, a):")
    print_param_kernels(est_p_ba)
    ll_ba_est = LL_2_alpha_kernel_sum_off(est_p_ba, d_ba, d_ab, end_time_sim, len(a_nodes), M)
    print(f"fit time = {(end_fit_time - start_fit_time):.4f} s, ll_ab = {ll_ba_est}")
    print("total ll = ", (ll_ab_est + ll_ba_est))

def test_simulate_fit_sum_kernels_nr_dia():
    # betas = np.array([.05, 5, 20])
    # C = np.array(np.array([0.01, 0.14, 0.85]))
    # param_sim_k = (0.005, .5, 0.7, C, betas)
    # param_sim = (0.005, .5, 0.7, 5)
    # a_nodes = list(range(12))
    # n_nodes = len(a_nodes)
    # end_time_sim = 3000

    # test simulation from sum of kernels
    betas = np.array([.1, 1, 10])
    C = np.array(np.array([0.2, 0.3, 0.5]))
    param_sim_k = (0.003, .3, 0.04, C, betas)
    scale = np.sum(C/betas)
    param_sim_detailed = (0.003, .3, 0.04, 999)
    a_nodes = list(range(20))
    n_nodes = len(a_nodes)
    end_time_sim = 3000

    print(f"Simulate and fit diagonal (n&r) sum of kernels (new kernel) at betas = {betas}")
    print("simulation #nodes = ", n_nodes, " Simulation duration = ", end_time_sim)
    M = n_nodes * (n_nodes - 1)
    # sum of kernels diagonal block simulation
    events_list_dia, events_dict_dia = simulate_kernel_sum_dia_2(param_sim_k, a_nodes, end_time_sim, return_list=True)
    n_events = cal_num_events_2(events_dict_dia)
    print("number of events = ", n_events)
    print("actual parameters:")
    print_param_kernels(param_sim_k)

    # Actual parameters log-likelihood
    ll_sum = LL_2_alpha_kernel_sum_dia(param_sim_k, events_dict_dia, end_time_sim, n_nodes, M)
    # Actual parameters detailed log-likelihood (two values should match up)
    param_array = get_array_param_n_r_dia(param_sim_detailed, n_nodes)
    ll_detailed = detailed_LL_kernel_sum_2(param_array, C, betas, events_list_dia, end_time_sim, M)
    print(f"ll_sum = {ll_sum}, detailed = {ll_detailed}")

    # test NLL jacobian function
    Q = len(betas)
    Ris = cal_R_2_alpha_kernel_sum_dia(events_dict_dia, betas)
    events_array = list(events_dict_dia.values())
    T_diff = end_time_sim - np.concatenate(events_array)
    T_diff_sums = np.zeros(Q, )
    for q in range(Q):
        T_diff_sums[q] = np.sum(1 - np.exp(-betas[q] * T_diff))
    p = (0.005, .5, 0.7, 0.01, 0.14, 0.85)
    eps = np.sqrt(np.finfo(float).eps)
    print("Derivates of NLL function Sum of kernels (new kernel) diganal")
    print("d_mu, d_alpha_n, d_alpha_r, d_C")
    print("Finite-difference approximation of the gradient")
    print(approx_fprime(p, NLL_2_alpha_kernel_sum_dia, eps, betas, events_dict_dia, end_time_sim, n_nodes, M, T_diff_sums, Ris))
    print("Actual gradient")
    print(NLL_2_alpha_kernel_sum_dia_jac(p, betas, events_dict_dia, end_time_sim, n_nodes, M, T_diff_sums, Ris))

    # fit Sum of kernels diagonal block pair
    start_fit_time = time.time()
    est_params = fit_2_alpha_kernel_sum_dia(events_dict_dia, end_time_sim, n_nodes, M, betas)
    end_fit_time = time.time()
    print_param_kernels(est_params)
    estimated_ll = LL_2_alpha_kernel_sum_dia(est_params, events_dict_dia, end_time_sim, n_nodes, M)
    print(f"estimated ll = {estimated_ll}, time to fit = {(end_fit_time - start_fit_time):.4f} s")

    # fit Sum of kernels diagonal block pair on different range of betas
    betas_1 = np.array([0.02, 6, 20])
    print("Testing fitting at betas = ", betas_1)
    start_fit_time = time.time()
    est_params = fit_2_alpha_kernel_sum_dia(events_dict_dia, end_time_sim, n_nodes, M, betas_1)
    end_fit_time = time.time()
    print_param_kernels(est_params)
    estimated_ll = LL_2_alpha_kernel_sum_dia(est_params, events_dict_dia, end_time_sim, n_nodes, M)
    print(f"estimated ll = {estimated_ll}, time to fit = {(end_fit_time - start_fit_time):.4f} s")

# %% code to test fitting function

# temporary functions for lambda plot
def np_list(nodes_a, nodes_b):
    np_list = []
    for i in nodes_a:
        for j in nodes_b:
            if i != j:
                np_list.append((i,j))
    return np_list


if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    # test_simulate_fit_one_beta_dia_n_r_br_gr_al_alr()
    # test_simulate_fit_one_beta_dia_n_r_br_gr()
    # test_simulate_fit_one_beta_off_n_r_br_gr()
    # test_simulate_fit_one_beta_dia_n_r()
    # test_simulate_fit_one_beta_off_n_r()
    # test_sim_one_beta_fit_sum_of_kernels_diagonal()
    # test_simulate_fit_sum_kernels_dia()
    # test_simulate_fit_sum_kernels_off()
    # test_simulate_fit_sum_kernels_nr_dia()
    # test_simulate_fit_sum_kernels_nr_off()
    # test_simulate_fit_one_beta_off_n_r_br_gr_al_alr()
    # test_simulate_fit_sum_kernels_al_off()
    # test_simulate_fit_sum_kernels_al_dia()
    # two off-diagonal block pais simulation

    p_ab = (0.001, 0.4, 0.2, 0.03, 0.0001, 0.002, 0.0001, 5)
    p_ba = (0.002, 0.1, 0.5, 0.0003, 0.001, 0.0002, 0.001, 5)
    beta = p_ab[-1]
    n_a, n_b = 5, 4

    _, alpha_off, _ = get_array_param_n_r_br_gr_al_alr_off(p_ab, p_ba, n_a, n_b)

    w, v = np.linalg.eig(alpha_off)
    me = np.amax(np.abs(w))
    print('Max eigenvalue: %1.9f' % me)
    print(np.sum(alpha_off, axis=1))
    print(0.4 + 0.2 + 3*(0.03+ 0.0001) + 4*(0.002+ 0.0001))

    # alpha_dia = get_alpha_n_r_br_gr_al_alr_dia((0.4, 0.2, 0.03, 0.0001, 0.002, 0.0001), n_nodes=5)
    # w_dia, _ = np.linalg.eig(alpha_dia)
    # print('Max eigenvalue: %1.5f' % np.amax(np.abs(w_dia)))
    # print(np.sum(alpha_dia, axis=1))














