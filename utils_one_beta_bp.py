import numpy as np
from scipy.optimize import minimize, approx_fprime
from bisect import bisect_left
import time
import sys

import utils_generate_bp as generate_bp
from os import path, getcwd
sys.path.append(path.join(getcwd(), "hawkes"))


#%% helper functions
def print_param(param):
    if len(param)==4:
        print(f"mu={param[0]:.7f}, alpha_n={param[1]:.4f}, alpha_r={param[2]:.4f}, beta={param[3]:.3f}")
    elif len(param)==6:
        print(f"mu={param[0]:.7f}, alpha_n={param[1]:.4f}, alpha_r={param[2]:.4f}, alpha_br={param[3]:.4f},"
          f" alpha_gr={param[4]:.4f}, beta={param[5]:.3f}")
    elif len(param)==8:
        print(f"mu={param[0]:.7f}, alpha_n={param[1]:.2f}, alpha_r={param[2]:.2f}, alpha_br={param[3]:.5f},"
              f" alpha_gr={param[4]:.5f}, alpha_al={param[5]:.4f}, alpha_alr={param[6]:.4f}, beta={param[7]:.3f}")

def cal_diff_sum(events_dict, end_time, beta):
    events_array = list(events_dict.values())
    if len(events_array) !=0 :
        return np.sum(1 - np.exp(-beta * (end_time - np.concatenate(events_array))))
    else:
        return 0

def cal_num_events(events_dict):
    num_events = 0
    for events_array in events_dict.values():
        num_events += len(events_array)
    return num_events
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

#%% 6-alpha diagonal block pair log-likelihood and fit functions (single kernel)

def cal_R_6_alpha_one_beta_dia(events_dict, beta):
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
def LL_6_alpha_one_beta_dia(params, events_dict, end_time, n_nodes, M, Ris=None, diff_sum=None):
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
        Ris = cal_R_6_alpha_one_beta_dia(events_dict, beta)  # list of M_np elements, each is (n_events_np, 6) array
    third = 0
    for i in range(len(Ris)):
        third += np.sum(np.log(mu + beta*(alpha_n * Ris[i][:, 0] + alpha_r * Ris[i][:, 1] +
                                          alpha_br * Ris[i][:, 2] + alpha_gr * Ris[i][:, 3] +
                                          alpha_al * Ris[i][:, 4] + alpha_alr * Ris[i][:, 5])))
    log_likelihood_value = first + second + third
    return log_likelihood_value
def NLL_6_alpha_one_beta_dia(params, events_dict, end_time, n_nodes, M, beta, Ris, diff_sum):
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr = params
    params_fixed_b = mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, beta
    return -LL_6_alpha_one_beta_dia(params_fixed_b, events_dict, end_time, n_nodes, M, Ris, diff_sum)
def NLL_6_alpha_one_beta_dia_jac(params, events_dict, end_time, n_nodes, M, beta, Ris, diff_sum):
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
def fit_6_alpha_one_beta_dia(events_dict, end_time, n_nodes, M, beta):
    # events_dict : (u,v):array_of_events
    if len(events_dict) == 0:  # handling empty block pair with no events
        return (1e-10, 0, 0, 0, 0, 0, 0, beta)
    mu_i = np.random.uniform(1e-6, 1e-2)
    alpha_n_i, alpha_r_i = np.random.uniform(0.1, 0.5, 2)
    alpha_br_i, alpha_gr_i, alpha_al_i, alpha_alr_i = np.random.uniform(1e-5, 0.1, 4)
    init_param = tuple([mu_i, alpha_n_i, alpha_r_i, alpha_br_i, alpha_gr_i, alpha_al_i, alpha_alr_i]) # <-- random initialization
    # init_param = (1e-2, 2e-2, 2e-2, 1e-2, 1e-2, 1e-2, 1e-2)
    Ris = cal_R_6_alpha_one_beta_dia(events_dict, beta)
    diff_sum = cal_diff_sum(events_dict, end_time, beta)
    res = minimize(NLL_6_alpha_one_beta_dia, init_param, method='L-BFGS-B',
                   bounds=((1e-7, None), (1e-7, None), (1e-7, None), (1e-7, None), (1e-7, None), (1e-7, None), (1e-7, None)),
                   jac=NLL_6_alpha_one_beta_dia_jac,
                   args=(events_dict, end_time, n_nodes, M, beta, Ris, diff_sum), tol=1e-12)
    results = res.x
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr = results[0:]
    return (mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, beta)

#%% 6-alpha off-diagonal block pair log-likelihood and fit functions (single kernel)
def cal_R_6_alpha_one_beta_off(events_dict, events_dict_r, beta):
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
def LL_6_alpha_one_beta_off(params, events_dict, events_dict_r, end_time, N_b, M_ab, Ris=None, diff_sum=None, diff_sum_r=None):
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
        Ris = cal_R_6_alpha_one_beta_off(events_dict, events_dict_r, beta)  # list of M_np elements, each is (n_events_np,6) array
    third = 0
    for i in range(len(Ris)):
        third += np.sum(np.log(mu + beta * (alpha_n * Ris[i][:, 0] + alpha_r * Ris[i][:, 1]
                                            + alpha_br * Ris[i][:, 2] + alpha_gr * Ris[i][:, 3]
                                            + alpha_al * Ris[i][:, 4] + alpha_alr * Ris[i][:, 5])))
    log_likelihood_value = first + second + third
    # print("LL ", first, second, third)
    return log_likelihood_value
def NLL_6_alpha_one_beta_off(p, d_ab, d_ba, end_time, n_nodes_to, M, beta, Ris, diff_sum, diff_sum_r):
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr = p
    param = mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, beta
    return -LL_6_alpha_one_beta_off(param, d_ab, d_ba, end_time, n_nodes_to, M, Ris, diff_sum, diff_sum_r)
def NLL_6_alpha_one_beta_off_jac(params, events_dict, events_dict_r, end_time, N_b, M_ab, beta, Ris, diff_sum, diff_sum_r):
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
def fit_6_alpha_one_beta_off(events_dict, events_dict_r, end_time, n_nodes_to, M, beta):
    # events_dict : (u,v):array_of_events
    if len(events_dict) == 0:  # handling empty block pair with no events
        return (1e-10, 0, 0, 0, 0, 0, 0, beta)
    mu_i = np.random.uniform(1e-6, 1e-2)
    alpha_n_i, alpha_r_i = np.random.uniform(0.1, 0.5, 2)
    alpha_br_i, alpha_gr_i, alpha_al_i, alpha_alr_i = np.random.uniform(1e-5, 0.1, 4)
    init_param = tuple([mu_i, alpha_n_i, alpha_r_i, alpha_br_i, alpha_gr_i, alpha_al_i, alpha_alr_i])  # <-- random initialization
    # init_param = (1e-2, 2e-2, 2e-2, 1e-2, 1e-2, 1e-2, 1e-2)
    Ris = cal_R_6_alpha_one_beta_off(events_dict, events_dict_r, beta)
    diff_sum = cal_diff_sum(events_dict, end_time, beta)
    diff_sum_r = cal_diff_sum(events_dict_r, end_time, beta)
    res = minimize(NLL_6_alpha_one_beta_off, init_param, method='L-BFGS-B',
                   bounds=((1e-7, None), (1e-7, None), (1e-7, None), (1e-7, None), (1e-7, None), (1e-7, None), (1e-7, None))
                   , jac=NLL_6_alpha_one_beta_off_jac,
                   args=(events_dict, events_dict_r, end_time, n_nodes_to, M, beta, Ris, diff_sum, diff_sum_r), tol=1e-12)
    results = res.x
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr = results
    return (mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, beta)
#%% 4-alpha diagonal block pair log-likelihood and fit functions (single kernel)

def cal_R_4_alpha_one_beta_dia(events_dict, beta):
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

def LL_4_alpha_one_beta_dia(params, events_dict, end_time, n_nodes, M, Ris=None, diff_sum=None):
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
        Ris = cal_R_4_alpha_one_beta_dia(events_dict, beta)  # list of M_np elements, each is (n_events_np,4) array
    # print("time after R()  = ", time.time())
    third = 0
    for i in range(len(Ris)):
        third += np.sum(np.log(mu + beta*(alpha_n * Ris[i][:, 0] + alpha_r * Ris[i][:, 1] +
                                          alpha_br * Ris[i][:, 2] + alpha_gr * Ris[i][:, 3])))
    # print("third r = ", third)
    log_likelihood_value = first + second + third
    return log_likelihood_value

def NLL_4_alpha_one_beta_dia(params, events_dict, end_time, n_nodes, M, beta, Ris, diff_sum):
    mu, alpha_n, alpha_r, alpha_br, alpha_gr = params
    params_fixed_b = mu, alpha_n, alpha_r, alpha_br, alpha_gr, beta
    return -LL_4_alpha_one_beta_dia(params_fixed_b, events_dict, end_time, n_nodes, M, Ris, diff_sum)

def NLL_4_alpha_one_beta_dia_jac(params, events_dict, end_time, n_nodes, M, beta, Ris, diff_sum):
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

def fit_4_alpha_one_beta_dia(events_dict, end_time, n_nodes, M, beta):
    # events_dict : (u,v):array_of_events
    if len(events_dict) == 0:  # handling empty block pair with no events
        return (1e-10, 0, 0, 0, 0, beta)
    mu_i = np.random.uniform(1e-6, 1e-2)
    alpha_n_i, alpha_r_i = np.random.uniform(0.1, 0.5, 2)
    alpha_br_i, alpha_gr_i = np.random.uniform(1e-5, 0.1, 2)
    init_param = tuple([mu_i, alpha_n_i, alpha_r_i, alpha_br_i, alpha_gr_i]) # <-- random initialization
    # init_param = (1e-2, 2e-2, 2e-2, 1e-2, 1e-2)
    Ris = cal_R_4_alpha_one_beta_dia(events_dict, beta)
    events_array = list(events_dict.values())
    diff_sum = np.sum(1 - np.exp(-beta * (end_time - np.concatenate(events_array))))
    res = minimize(NLL_4_alpha_one_beta_dia, init_param, method='L-BFGS-B',
                   bounds=((1e-7, None), (1e-7, None), (1e-7, None), (1e-7, None), (1e-7, None)),
                   jac=NLL_4_alpha_one_beta_dia_jac,
                   args=(events_dict, end_time, n_nodes, M, beta, Ris, diff_sum), tol=1e-12)
    results = res.x
    mu = results[0]
    alpha_n = results[1]
    alpha_r = results[2]
    alpha_br = results[3]
    alpha_gr = results[4]
    return (mu, alpha_n, alpha_r, alpha_br, alpha_gr, beta)

#%% 4-alpha off-diagonal block pair log-likelihood and fit functions (single kernel)
def cal_R_4_alpha_one_beta_off(events_dict, events_dict_r, beta):
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

def LL_4_alpha_one_beta_off(params, events_dict, events_dict_r, end_time, n_nodes_to, M, Ris=None, diff_sum=None, diff_sum_r=None):
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
        Ris = cal_R_4_alpha_one_beta_off(events_dict, events_dict_r, beta)  # list of M_np elements, each is (n_events_np,4) array
    third = 0
    for i in range(len(Ris)):
        third += np.sum(np.log(mu + beta * (alpha_n * Ris[i][:, 0] + alpha_r * Ris[i][:, 1]
                                            + alpha_br * Ris[i][:, 2] + alpha_gr * Ris[i][:, 3])))
    # print("third r = ", third)
    log_likelihood_value = first + second + third
    return log_likelihood_value

def NLL_4_alpha_one_beta_off(params, d_ab, d_ba, end_time, n_nodes_to, M, beta, Ris, diff_sum, diff_sum_r):
    mu, alpha_n, alpha_r, alpha_br, alpha_gr = params
    params_fixed_b = mu, alpha_n, alpha_r, alpha_br, alpha_gr, beta
    return -LL_4_alpha_one_beta_off(params_fixed_b, d_ab, d_ba, end_time, n_nodes_to, M, Ris, diff_sum, diff_sum_r)

def NLL_4_alpha_one_beta_off_jac(params, events_dict, events_dict_r, end_time, n_nodes_to, M, beta, Ris, diff_sum, diff_sum_r):
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

def fit_4_alpha_one_beta_off(events_dict, events_dict_r, end_time, n_nodes_to, M, beta):
    # events_dict : (u,v):array_of_events
    if len(events_dict) == 0:  # handling empty block pair with no events
        return (1e-10, 0, 0, 0, 0, beta)
    mu_i = np.random.uniform(1e-6, 1e-2)
    alpha_n_i, alpha_r_i = np.random.uniform(0.1, 0.5, 2)
    alpha_br_i, alpha_gr_i = np.random.uniform(1e-5, 0.1, 2)
    init_param = tuple([mu_i, alpha_n_i, alpha_r_i, alpha_br_i, alpha_gr_i]) # <-- random initialization
    # init_param = (1e-2, 2e-2, 2e-2, 1e-2, 1e-2)
    Ris = cal_R_4_alpha_one_beta_off(events_dict, events_dict_r, beta)
    events_array = list(events_dict.values())
    events_array_r = list(events_dict_r.values())
    diff_sum = np.sum(1 - np.exp(-beta * (end_time - np.concatenate(events_array))))
    if len(events_array_r) == 0:
        diff_sum_r = 0
    else:
        diff_sum_r = np.sum(1 - np.exp(-beta * (end_time - np.concatenate(events_array_r))))
    res = minimize(NLL_4_alpha_one_beta_off, init_param, method='L-BFGS-B',
                   bounds=((1e-7, None), (1e-7, None), (1e-7, None), (1e-7, None), (1e-7, None)), jac=NLL_4_alpha_one_beta_off_jac,
                   args=(events_dict, events_dict_r, end_time, n_nodes_to, M, beta, Ris, diff_sum, diff_sum_r), tol=1e-12)
    results = res.x
    mu = results[0]
    alpha_n = results[1]
    alpha_r = results[2]
    alpha_br = results[3]
    alpha_gr = results[4]
    return (mu, alpha_n, alpha_r, alpha_br, alpha_gr, beta)


#%% 2-alpha diagonal block pair log-likelihood and fit functions (single kernel)
def cal_R_2_alpha_one_beta_dia(events_dict, beta):
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
def LL_2_alpha_one_beta_dia(params, events_dict, end_time, n_nodes, M, Ris=None, diff_sum=None):
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
        Ris = cal_R_2_alpha_one_beta_dia(events_dict, beta)  # list of M_np elements, each is (n_events_np,2) array
    third = 0
    for i in range(len(Ris)):
        third += np.sum(np.log(mu + beta*(alpha_n * Ris[i][:, 0] + alpha_r * Ris[i][:, 1])))
    log_likelihood_value = first + second + third
    return log_likelihood_value
def NLL_2_alpha_one_beta_dia(params, events_dict, end_time, n_nodes, M, beta, Ris, diff_sum):
    mu, alpha_n, alpha_r = params
    params_fixed_b = mu, alpha_n, alpha_r, beta
    return - LL_2_alpha_one_beta_dia(params_fixed_b, events_dict, end_time, n_nodes, M, Ris, diff_sum)
def NLL_2_alpha_one_beta_dia_jac(params, events_dict, end_time, n_nodes, M, beta, Ris, diff_sum):
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
def fit_2_alpha_one_beta_dia(events_dict, end_time, n_nodes, M, beta):
    # events_dict : (u,v):array_of_events
    if len(events_dict) == 0:  # handling empty block pair with no events
        return (1e-10, 0, 0, beta)
    # parameters initialization
    mu_i = np.random.uniform(1e-6, 1e-2)
    alpha_n_i, alpha_r_i = np.random.uniform(0.1, 0.5, 2)
    init_param = tuple([mu_i, alpha_n_i, alpha_r_i]) # <-- random initialization
    # init_param = (1e-2, 2e-2, 2e-2)
    Ris = cal_R_2_alpha_one_beta_dia(events_dict, beta)
    events_array = list(events_dict.values())
    diff_sum = np.sum(1 - np.exp(-beta * (end_time - np.concatenate(events_array))))
    res = minimize(NLL_2_alpha_one_beta_dia, init_param, method='L-BFGS-B', bounds=((1e-7, None), (1e-7, None), (1e-7, None)),
                   jac=NLL_2_alpha_one_beta_dia_jac, args=(events_dict, end_time, n_nodes, M, beta, Ris, diff_sum), tol=1e-12)
    results = res.x
    mu = results[0]
    alpha_n = results[1]
    alpha_r = results[2]
    return (mu, alpha_n, alpha_r, beta)

#%% 2-alpha off-diagonal block pair log-likelihood and fit functions (single kernel)
def cal_R_2_alpha_one_beta_off(events_dict, events_dict_r, beta):
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
def LL_2_alpha_one_beta_off(params, events_dict, events_dict_r, end_time, n_nodes_to, M, Ris=None, diff_sum=None, diff_sum_r=None):
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
        Ris = cal_R_2_alpha_one_beta_off(events_dict, events_dict_r, beta)  # list of M_np elements, each is (n_events_np,4) array
    third = 0
    for i in range(len(Ris)):
        third += np.sum(np.log(mu + beta * (alpha_n * Ris[i][:, 0] + alpha_r * Ris[i][:, 1])))
    log_likelihood_value = first + second + third
    # print(f"ll = {first} + {second} + {third} = {first + second + third}")
    return log_likelihood_value
def NLL_2_alpha_one_beta_off(params, events_dict, events_dict_r, end_time, n_nodes_to, M, beta, Ris, diff_sum, diff_sum_r):
    mu, alpha_n, alpha_r = params
    params_fixed_b = mu, alpha_n, alpha_r, beta
    return -LL_2_alpha_one_beta_off(params_fixed_b, events_dict, events_dict_r, end_time, n_nodes_to, M, Ris, diff_sum, diff_sum_r)
def NLL_2_alpha_one_beta_off_jac(params, events_dict, events_dict_r, end_time, n_nodes_to, M, beta, Ris, diff_sum, diff_sum_r):
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
def fit_2_alpha_one_beta_off(events_dict, events_dict_r, end_time, n_nodes_to, M, beta):
    # events_dict : (u,v):array_of_events
    if len(events_dict) == 0:  # handling empty block pair with no events
        return (1e-10, 0, 0, beta)
    mu_i = np.random.uniform(1e-6, 1e-2)
    alpha_n_i, alpha_r_i = np.random.uniform(0.1, 0.5, 2)
    init_param = tuple([mu_i, alpha_n_i, alpha_r_i]) # <-- random initialization
    # init_param = (1e-2, 2e-2, 2e-2) # <-- fixed initialization
    Ris = cal_R_2_alpha_one_beta_off(events_dict, events_dict_r, beta)
    events_array = list(events_dict.values())
    events_array_r = list(events_dict_r.values())
    diff_sum = np.sum(1 - np.exp(-beta * (end_time - np.concatenate(events_array))))
    if len(events_array_r) == 0:
        diff_sum_r = 0
    else:
        diff_sum_r = np.sum(1 - np.exp(-beta * (end_time - np.concatenate(events_array_r))))
    res = minimize(NLL_2_alpha_one_beta_off, init_param, method='L-BFGS-B', bounds=((1e-7, None), (1e-7, None), (1e-7, None)),
                   jac=NLL_2_alpha_one_beta_off_jac, args=(events_dict, events_dict_r, end_time, n_nodes_to, M, beta, Ris, diff_sum, diff_sum_r),
                   tol=1e-12)
    results = res.x
    mu = results[0]
    alpha_n = results[1]
    alpha_r = results[2]
    return (mu, alpha_n, alpha_r, beta)

#%% 2-alpha off-diagonal block pair (restricted rho) log-likelihood and fit functions (single kernel)
def LL_2_alpha_one_beta_off_rho(params, events_ab, events_ba, end_time, N_a, N_b,
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
        Ris_ab = cal_R_2_alpha_one_beta_off(events_ab, events_ba, beta)  # list of np with events in (a, b), each is (n_events_np,2) array
        Ris_ba = cal_R_2_alpha_one_beta_off(events_ba, events_ab, beta)  # list of np with events in (b, a), each is (n_events_np,2) array
    third = 0
    for i in range(len(Ris_ab)):
        third += np.sum(np.log(mu_ab + beta * (alpha_ab * Ris_ab[i][:, 0] + rho * alpha_ab * Ris_ab[i][:, 1])))
    for i in range(len(Ris_ba)):
        third += np.sum(np.log(mu_ba + beta * (alpha_ba * Ris_ba[i][:, 0] + rho * alpha_ba * Ris_ba[i][:, 1])))
    log_likelihood_value = first + second + third
    # print(f"ll_rho = {first} + {second} + {third} = {first + second + third}")
    return log_likelihood_value
def NLL_2_alpha_one_beta_off_rho(params, events_ab, events_ba, end_time, N_a, N_b, beta,
                                 Ris_ab, Ris_ba, diff_sum_ab, diff_sum_ba):
    mu_ab, mu_ba, alpha_ab, alpha_ba, rho = params
    params_beta = mu_ab, mu_ba, alpha_ab, alpha_ba, rho, beta
    return -LL_2_alpha_one_beta_off_rho(params_beta, events_ab, events_ba, end_time, N_a, N_b,
                                        Ris_ab, Ris_ba, diff_sum_ab, diff_sum_ba)
def NLL_2_alpha_one_beta_off_rho_jac(params, events_ab, events_ba, end_time, N_a, N_b, beta,
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
def fit_2_alpha_one_beta_off_rho(events_ab, events_ba, end_time, N_a, N_b, beta):
    # if both bp's have no events
    if len(events_ab) == 0 and len(events_ba)==0:
        return (1e-10, 1e-10, 0, 0, 0, beta)    #(mu_ab, mu_ba, alpha_ab, alpha_ba, rho, beta)
    # if only bp(b,a) has events
    elif len(events_ab) == 0:
        mu_ba, alpha_ba, rho_alpha_ba, beta = fit_2_alpha_one_beta_off(events_ba, events_ab, end_time, N_a, N_a * N_b, beta)
        rho = rho_alpha_ba/ alpha_ba
        return (1e-10, mu_ba, 0, alpha_ba, rho, beta)
    # if only bp(a, b) has events
    elif len(events_ba) == 0:
        mu_ab, alpha_ab, rho_alpha_ab, beta = fit_2_alpha_one_beta_off(events_ab, events_ba, end_time, N_b, N_a * N_b, beta)
        rho = rho_alpha_ab / alpha_ab
        return (mu_ab, 1e-10, alpha_ab, 0, rho, beta)

    # initialize parameter randomly
    mu_ab_i, mu_ba_i = np.random.uniform(1e-6, 1e-2, 2)
    alpha_ab_i, alpha_ba_i = np.random.uniform(0.1, 0.5, 2)
    rho_i = np.random.uniform(0.1, 0.5)
    param_i = tuple([mu_ab_i, mu_ba_i, alpha_ab_i, alpha_ba_i, rho_i]) # <-- random initialization

    Ris_ab = cal_R_2_alpha_one_beta_off(events_ab, events_ba, beta)  # list of np with events in (a, b), each is (n_events_np,2) array
    Ris_ba = cal_R_2_alpha_one_beta_off(events_ba, events_ab, beta)  # list of np with events in (b, a), each is (n_events_np,2) array

    diff_sum_ab, diff_sum_ba = 0, 0
    events_array_ab = list(events_ab.values())
    if len(events_array_ab) != 0:  # block pair ab is not empty
        diff_sum_ab = np.sum(1 - np.exp(-beta * (end_time - np.concatenate(events_array_ab))))
    events_array_ba = list(events_ba.values())
    if len(events_array_ba) != 0:  # block pair ba is not empty
        diff_sum_ba = np.sum(1 - np.exp(-beta * (end_time - np.concatenate(events_array_ba))))

    res = minimize(NLL_2_alpha_one_beta_off_rho, param_i, method='L-BFGS-B', bounds=tuple([(1e-7, None)] * 4 + [(0, None)]),
                   jac=NLL_2_alpha_one_beta_off_rho_jac,
                   args=(events_ab, events_ba, end_time, N_a, N_b, beta, Ris_ab, Ris_ba, diff_sum_ab, diff_sum_ba), tol=1e-12)
    results = res.x
    mu_ab, mu_ba, alpha_ab, alpha_ba, rho = results
    return (mu_ab, mu_ba, alpha_ab, alpha_ba, rho, beta)

#%% detailed log-likelihood functions
""" only for checking code """
# single beta - beta*alpha*exp(-beta*t) kernel
def log_likelihood_detailed(mu_alpha_array, beta, events_list, end_time, M):
    """ mu_array : (M,) array : baseline intensity of each process
        alpoha_array: (M,M) narray: adjacency*beta (actual jumbs) """
    mu_array, alpha_array = mu_alpha_array
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
#%% Test functions
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
    l, d_ab, d_ba = generate_bp.simulate_one_beta_off_2(param_ab, param_ba, a_nodes, b_nodes, end_time_sim, return_list=True)
    print(f"number of events = {cal_num_events(d_ab)} + {cal_num_events(d_ba)} = {cal_num_events(d_ba)+cal_num_events(d_ab)}")
    # true parameters log-likelihood
    ll_ab = LL_2_alpha_one_beta_off(param_ab, d_ab, d_ba, end_time_sim, len(b_nodes), M)
    ll_ba = LL_2_alpha_one_beta_off(param_ba, d_ba, d_ab, end_time_sim, len(a_nodes), M)
    # param_array_actual = get_array_param_n_r_off(param_ab, param_ba, len(a_nodes), len(b_nodes))
    # ll_detailed = log_likelihood_detailed_2(param_array_actual, l, end_time_sim, 2 * M)
    ll_rho = LL_2_alpha_one_beta_off_rho(param_rho, d_ab, d_ba, end_time_sim, len(a_nodes), len(b_nodes))
    print(f"ll = {ll_ab:.2f}+{ll_ba:.2f} ={ll_ab + ll_ba:.2f}, ll_rho={ll_rho:.2f}")

    # jac check
    grad = False
    if grad:
        Ris_ab = cal_R_2_alpha_one_beta_off(d_ab, d_ba, beta)
        Ris_ba = cal_R_2_alpha_one_beta_off(d_ba, d_ab, beta)
        events_array_ab = list(d_ab.values())
        events_array_ba = list(d_ba.values())
        diff_sum_ab = np.sum(1 - np.exp(-beta * (end_time_sim - np.concatenate(events_array_ab))))
        diff_sum_ba = np.sum(1 - np.exp(-beta * (end_time_sim - np.concatenate(events_array_ba))))
        eps = np.sqrt(np.finfo(float).eps)
        print("NLL gradients (d_mu, d_alpha_n, d_alpha_r) - blockpair(b,a)")
        print("Approximation of the gradient - (a, b)")
        print(approx_fprime(param_ba[0:3], NLL_2_alpha_one_beta_off, eps, d_ba, d_ab, end_time_sim, len(a_nodes), M, beta, Ris_ba, diff_sum_ba,
                            diff_sum_ab))
        print("Actual gradient - (a, b)")
        print(NLL_2_alpha_one_beta_off_jac(param_ba[0:3], d_ba, d_ab, end_time_sim, len(a_nodes), M, beta, Ris_ba, diff_sum_ba, diff_sum_ab))
        print("Approximation of the gradient - rho")
        print(approx_fprime(param_rho[0:-1], NLL_2_alpha_one_beta_off_rho, eps, d_ab, d_ba, end_time_sim, len(a_nodes), len(b_nodes), beta,
                            Ris_ab, Ris_ba, diff_sum_ab, diff_sum_ba))
        print("Actual gradient - rho")
        print(NLL_2_alpha_one_beta_off_rho_jac(param_rho[0:-1], d_ab, d_ba, end_time_sim, len(a_nodes), len(b_nodes), beta,
                                               Ris_ab, Ris_ba, diff_sum_ab, diff_sum_ba))

    # # fitting off-diagonal pair (n, r)
    start_fit_time = time.time()
    param_est_ab = fit_2_alpha_one_beta_off(d_ab, d_ba, end_time_sim, len(b_nodes), M, beta)
    end_fit_time = time.time()
    print("Estimated parameters of block pair (a, b):")
    print(f"\tmu={param_est_ab[0]}, alpha_n={param_est_ab[1]}, alpha_r={param_est_ab[2]}")
    print(f"\tfit time = {(end_fit_time - start_fit_time):.4f} s")
    ll_est_ab = LL_2_alpha_one_beta_off(param_est_ab, d_ab, d_ba, end_time_sim, len(b_nodes), M)
    print("\tlog-likelihood= ", ll_est_ab)
    print("Estimated parameters of block pair (b, a):")
    start_fit_time = time.time()
    param_est_ba = fit_2_alpha_one_beta_off(d_ba, d_ab, end_time_sim, len(a_nodes), M, beta)
    end_fit_time = time.time()
    print(f"\tmu={param_est_ba[0]}, alpha_n={param_est_ba[1]}, alpha_r={param_est_ba[2]}")
    print(f"\tfit time = {(end_fit_time - start_fit_time):.4f} s")
    ll_est_ba = LL_2_alpha_one_beta_off(param_est_ba, d_ba, d_ab, end_time_sim, len(a_nodes), M)
    print("\tlog-likelihood= ", ll_est_ba)
    print("fitted ll of both = ", ll_est_ab + ll_est_ba)
    print("\nEstimated constrained paramters for both (a, b) & (b, a)")
    start_fit_time = time.time()
    param_est = fit_2_alpha_one_beta_off_rho(d_ab, d_ba, end_time_sim, len(a_nodes), len(b_nodes), beta)
    end_fit_time = time.time()
    print(f"\t(a, b): mu={param_est[0]}, alpha_n={param_est[2]}")
    print(f"\t(b, a): mu={param_est[1]}, alpha_n={param_est[3]}")
    print(f"\trho={param_est[4]}")
    print(f"\tfit time = {(end_fit_time - start_fit_time):.4f} s")
    ll_est = LL_2_alpha_one_beta_off_rho(param_est, d_ab, d_ba, end_time_sim, len(a_nodes), len(b_nodes))
    print("\tlog-likelihood= ", ll_est)

def test_simulate_6_alpha_fit_one_beta_dia_bp():
    # # Hawkes parameters
    beta = 5
    par = (0.001, 0.3, 0.4, 0.0005, 0.0001, 0.0003, 0.0002, beta)
    n_nodes = 30   # number of nodes
    N_list = list(range(n_nodes))
    M = n_nodes*(n_nodes-1)
    sim_end_time = 1000

    # matrix parameters for simulation and detailed log-likelihood test
    print("Simulate and fit (6-alpha) diagonal block pair single kernel at beta=", beta)
    sim_list, sim_dict = generate_bp.simulate_one_beta_dia_2(par, N_list, sim_end_time, return_list=True)
    print(f"#nodes={n_nodes}, #events={cal_num_events(sim_dict)}")
    # log-likelihood function
    ll_model = LL_6_alpha_one_beta_dia(par, sim_dict, sim_end_time, n_nodes, M)
    mu_alpha_arrays = generate_bp.get_mu_array_alpha_matrix_dia_bp(par[0], par[1:7], n_nodes)
    ll_detailed = log_likelihood_detailed(mu_alpha_arrays, beta, sim_list, sim_end_time, M)
    print(f"model LL={ll_model:.2f}, detailed LL={ll_detailed:.2f}")

    # log-likelihood jac function check
    diff_sum = cal_diff_sum(sim_dict, sim_end_time, beta)
    Ris = cal_R_6_alpha_one_beta_dia(sim_dict, beta)
    eps = np.sqrt(np.finfo(float).eps)
    print("LL derivates")
    print("Finite-difference approximation of the gradient")
    print(approx_fprime(par[:-1], NLL_6_alpha_one_beta_dia, eps, sim_dict, sim_end_time, n_nodes, M, beta, Ris, diff_sum))
    print("Actual gradient")
    print(NLL_6_alpha_one_beta_dia_jac(par[:-1], sim_dict, sim_end_time, n_nodes, M, beta, Ris, diff_sum))

    # fitting (n, r) global (br, gr) model with single known beta
    print("\nfitting (n, r) global (br, gr, al, alr) model with single known beta")
    start_fit_time = time.time()
    param_est = fit_6_alpha_one_beta_dia(sim_dict, sim_end_time, n_nodes, M, beta)
    end_fit_time = time.time()
    print_param(param_est)
    # elapsed time
    print(f"time to fit = {(end_fit_time - start_fit_time):.4f} s")
    ll_est = LL_6_alpha_one_beta_dia(param_est, sim_dict, sim_end_time, n_nodes, M)
    print("log-likelihood= ", ll_est)

def test_simulate_6_alpha_fit_one_beta_off_bp():
    sim_p_ab = (0.001, 0.4, 0.2, 0.03, 0.0001, 0.002, 0.0001, 5)
    sim_p_ba = (0.002, 0.1, 0.5, 0.0003, 0.001, 0.0002, 0.001, 5)
    beta = sim_p_ab[-1]
    # simulation
    n_a, n_b = 25, 14
    n_a_list, n_b_list = list(range(n_a)), list(range(n_a, n_a + n_b))
    T = 2000
    M_ab = n_a * n_b
    print("simulating (6-alpha) two off-diagonal block pairs at single beta=", beta)
    print("Actual parameters:")
    print_param(sim_p_ab)
    print_param(sim_p_ba)
    sim_list, sim_dict_ab, sim_dict_ba = generate_bp.simulate_one_beta_off_2(sim_p_ab, sim_p_ba, n_a_list, n_b_list, T, return_list=True)
    print(f"n_a={n_a}, n_b={n_b}, #events_ab={cal_num_events(sim_dict_ab)} ,#events_ba={cal_num_events(sim_dict_ba)}")
    # log-likelihood function
    ll_ab = LL_6_alpha_one_beta_off(sim_p_ab, sim_dict_ab, sim_dict_ba, T, n_b, M_ab)
    ll_ba = LL_6_alpha_one_beta_off(sim_p_ba, sim_dict_ba, sim_dict_ab, T, n_a, M_ab)
    print(f"LL_model={ll_ab:.2f} + {ll_ba:.2f} = {ll_ab + ll_ba}")

    # log-likelihood jac function check
    diff_sum = cal_diff_sum(sim_dict_ab, T, sim_p_ab[-1])
    diff_sum_r = cal_diff_sum(sim_dict_ba, T, sim_p_ba[-1])
    Ris = cal_R_6_alpha_one_beta_off(sim_dict_ab, sim_dict_ba, beta)
    eps = np.sqrt(np.finfo(float).eps)
    print("LL derivates")
    print("Finite-difference approximation of the gradient")
    print(approx_fprime(sim_p_ab[:-1], NLL_6_alpha_one_beta_off, eps, sim_dict_ab, sim_dict_ba, T, n_b, M_ab, beta, Ris, diff_sum,
                        diff_sum_r))
    print("Actual gradient")
    print(NLL_6_alpha_one_beta_off_jac(sim_p_ab[:-1], sim_dict_ab, sim_dict_ba, T, n_b, M_ab, beta, Ris, diff_sum, diff_sum_r))

    # fit to simulated data
    print("\nfitting one beta off-diagonal block pair (a, b)")
    start_fit_time = time.time()
    param_est_ab = fit_6_alpha_one_beta_off(sim_dict_ab, sim_dict_ba, T, n_b, M_ab, beta)
    end_fit_time = time.time()
    print_param(param_est_ab)
    # elapsed time and estimated parameters log-likelihood
    ll_est_ab = LL_6_alpha_one_beta_off(param_est_ab, sim_dict_ab, sim_dict_ba, T, n_b, M_ab)
    print(f"time to fit = {(end_fit_time - start_fit_time):.4f} s \t log-likelihood= {ll_est_ab:.4f}")

    print("\nfitting off-diagonal block pair ba (n, r, br, gr)")
    start_fit_time = time.time()
    param_est_ba = fit_6_alpha_one_beta_off(sim_dict_ba, sim_dict_ab, T, n_a, M_ab, beta)
    end_fit_time = time.time()
    print_param(param_est_ba)
    # elapsed time and estimated parameters log-likelihood
    ll_est_ba = LL_6_alpha_one_beta_off(param_est_ba, sim_dict_ba, sim_dict_ab, T, n_a, M_ab)
    print(f"time to fit = {(end_fit_time - start_fit_time):.4f} s \t log-likelihood= {ll_est_ba:.4f}")
    print("fitted ll of both = ", ll_est_ab + ll_est_ba)


if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    # test_simulate_6_alpha_fit_one_beta_dia_bp()
    test_simulate_6_alpha_fit_one_beta_off_bp()
