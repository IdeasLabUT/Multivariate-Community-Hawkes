"""MULCH fit and log-likelihood functions on block pair level

@author: Hadeel Soliman
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, approx_fprime
from bisect import bisect_left
import time
import utils_generate_model as generate_bp



#%% 6-alpha diagonal block pair log-likelihood and fit functions
def cal_R_6_alpha_dia_bp(events_dict, betas):
    """
    calculate recursive term in log-likelihood of a diagonal block pair (a, a) - 6 excitation types

    :param dict events_dict: dictionary of events within block pair (a, b),
        where {(u, v) node pair in (a, b) : [t1, t2, ..] array of events between (u, v)}
    :param betas: (Q,) array of decays
    :return: list of recursive function values for each (u, v) in events_dict
    """
    Q = len(betas)
    Ris = []
    for (u, v) in events_dict:
        # array of events of node pair (u,v)
        uv_events = events_dict[(u, v)]
        num_events_uv = len(uv_events)
        if num_events_uv == 0:
            Ris.append(np.array([0]))  # R=0 if node_pair (u,v) has no events
        else:
            # 6*Q columns (alpha_s, alpha_r, alpha_tc, alpha_gr, alpha_al, alpha_alr)*Q
            Ri = np.zeros((num_events_uv, 6 * Q))
            uv_intertimes = (uv_events[1:] - uv_events[:-1])
            # (#_uv_events-1, Q) array
            e_intertimes_Q = np.zeros((len(uv_intertimes), Q))
            for q in range(Q):
                e_intertimes_Q[:, q] = np.exp(-betas[q] * uv_intertimes)
            for (x, y) in events_dict:
                # same node_pair events (alpha_s)
                if (u, v) == (x, y):
                    for k in range(1, num_events_uv):
                        for q in range(Q):
                            Ri[k, 0 + q * 6] = e_intertimes_Q[k - 1, q] * (1 + Ri[k - 1, 0 + q * 6])
                # reciprocal node_pair events (alpha_r)
                elif (v, u) == (x, y):
                    Ri_temp = get_Ri_temp_Q(uv_events, e_intertimes_Q, events_dict[(x, y)], betas)
                    for q in range(Q):
                        Ri[:, 1 + q * 6] = Ri_temp[:, q]
                # br node_pairs events (alpha_tc)
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

def LL_6_alpha_dia_bp(params, events_dict, end_time, n_a, m, T_diff_sums=None, Ris=None):
    """
    calculate log-likelihood of one diagonal block pair - 6 excitation types

    :param tuple params: MULCH block pair parameters (mu, alpha_1, .. alpha_n, C, betas)
    :param dict dict events_dict: dictionary of events within a block pair (a, b),
        where {(u, v) node pair in (a, b) : [t1, t2, ..] array of events between (u, v)}
    :param float end_time: duration of the network
    :param int n_a: number of nodes in block a
    :param int m: number of node pairs in the diagonal block pair
    :param T_diff_sums: Optional (Q,) array(float)
    :param Ris: Optional list(size of events_dict)
    :return: block pair log-likelihood
    :rtype: float
    """

    # C: scaling parameters - same length as betas
    mu, alpha_s, alpha_r, alpha_tc, alpha_gr, alpha_al, alpha_alr, C, betas = params
    Q = len(betas)
    # first term
    first = -m * mu * end_time
    # block pair has no events (empty)
    if len(events_dict) == 0:
        return first
    # second term
    if T_diff_sums is None:
        T_diff_sums = cal_diff_sums_Q(events_dict, end_time, betas)
    second = -(alpha_s + alpha_r + (alpha_tc + alpha_gr + alpha_al + alpha_alr) * (n_a - 2)) * C @ T_diff_sums
    # third term
    if Ris is None:
        Ris = cal_R_6_alpha_dia_bp(events_dict, betas)
    third = 0
    for i in range(len(Ris)):
        col_sum = np.zeros(Ris[i].shape[0])
        for q in range(Q):
            col_sum[:] += C[q]*betas[q] * (alpha_s * Ris[i][:, 0 + q * 6]  + alpha_r * Ris[i][:, 1 + q * 6] +
                                           alpha_tc * Ris[i][:, 2 + q * 6] + alpha_gr * Ris[i][:,3 + q * 6] +
                                           alpha_al * Ris[i][:, 4 + q * 6] + alpha_alr * Ris[i][:,5 + q * 6])
        col_sum += mu
        third += np.sum(np.log(col_sum))
    log_likelihood_value = first + second + third
    # print("ll: ", first, second, third)
    return log_likelihood_value

def NLL_6_alpha_dia_bp(p, betas, events_dict, end_time, n_a, m, T_diff_sums, Ris):
    """
    negative log-likelihood of one diagonal block pair- 6 excitation type

    called by scipy.minimize() function for parameters estimation

    :param tuple p: MULCH block pair raveled parameters (mu, alpha_1, .., alpha_6, C_1, .., C_Q )
    :param betas: (Q,) array of decays
    :param dict events_dict: dictionary of events within a block pair (a, b),
        where {(u, v) node pair in (a, b) : [t1, t2, ..] array of events between (u, v)}
    :param float end_time: duration of the network
    :param int n_a: number of nodes in block a
    :param int m: number of node pairs in the diagonal block pair
    :param T_diff_sums: Optional (Q,) array(float)
    :param Ris: Optional list(size of events_dict)
    :return: block pair negative log-likelihood
    :rtype: float
    """

    mu, alpha_s, alpha_r, alpha_tc, alpha_gr, alpha_al, alpha_alr = p[:7]
    C = np.array(p[7:])
    # scaling step - constaint C sums to 1
    C_sum = np.sum(C)
    if (C_sum != 0):
        C = C / C_sum
        alpha_s = alpha_s * C_sum
        alpha_r = alpha_r * C_sum
        alpha_tc = alpha_tc * C_sum
        alpha_gr = alpha_gr * C_sum
        alpha_al = alpha_al * C_sum
        alpha_alr = alpha_alr * C_sum
    params = mu, alpha_s, alpha_r, alpha_tc, alpha_gr, alpha_al, alpha_alr, C, betas
    return -LL_6_alpha_dia_bp(params, events_dict, end_time, n_a, m, T_diff_sums, Ris)
def NLL_6_alpha_dia_bp_jac(p, betas, events_dict, end_time, n_a, m, T_diff_sums, Ris):
    """
    jacobian of negative log-likelihood of one diagonal block pair- 6 excitations

    called by scipy.minimize() function for parameters estimation

    :param tuple p: MULCH block pair parameters (mu, alpha_1, .., alpha_6, C_1, .., C_Q )
    :param betas: (Q,) array of decays
    :param dict events_dict: dictionary of events within a block pair (a, b),
        where {(u, v) node pair in (a, b) : [t1, t2, ..] array of events between (u, v)}
    :param float end_time: duration of the network
    :param int n_a: number of nodes in block a
    :param int m: number of node pairs in the diagonal block pair
    :param T_diff_sums: Optional (Q,) array(float)
    :param Ris: Optional list(size of events_dict)
    :return: jacobian array of negative log-likelihood function with respect to MULCH parameters
    """

    mu, alpha_s, alpha_r, alpha_tc, alpha_gr, alpha_al, alpha_alr = p[:7]
    C = np.array(p[7:])
    Q = len(C)
    # derivatives of second term
    d_mu = m * end_time
    d_alphas = np.zeros(6)
    d_alphas[:2] = C @ T_diff_sums
    d_alphas[2:] = (n_a - 2) * C @ T_diff_sums
    d_C = (alpha_s + alpha_r + (n_a - 2) * (alpha_tc + alpha_gr + alpha_al + alpha_alr)) * T_diff_sums
    # derivatives of third term
    for i in range(len(Ris)):
        denominator = np.zeros(Ris[i].shape[0])
        # one column for each alpha_j
        numerator_alphas = np.zeros((Ris[i].shape[0], 6))
        numerator_C = []
        for q in range(Q):
            numerator_C.append(betas[q]* (alpha_s * Ris[i][:, 0 + q * 6] + alpha_r * Ris[i][:, 1 + q * 6] +
                                          alpha_tc * Ris[i][:, 2 + q * 6] + alpha_gr * Ris[i][:,3 + q * 6] +
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

def fit_6_alpha_dia_bp(events_dict, end_time, n_a, m, betas):
    """
    fit mulch one diagonal block pair (a, b) - 6 excitations

    :param dict events_dict: dictionary of events within a block pair (a, b),
        where {(u, v) node pair in (a, b) : [t1, t2, ..] array of events between (u, v)}
    :param float end_time: duration of the network
    :param int n_a: number of nodes in block a
    :param int m: number of node pairs in the diagonal block pair
    :param betas: (Q,) array of decays
    :return: estimated parameters (mu, alpha_1,.. , alpha_n, C, betas)
    :rtype: tuple
    """

    Q = len(betas)
    # events_dict : (u,v):array_of_events
    if len(events_dict) == 0:  # handling empty block pair with no events
        # (mu, alpha_s, alpha_r, alpha_tc, alpha_gr, alpha_al, alpha_alr, C, betas)
        C = np.zeros(Q)
        return (1e-10, 0, 0, 0, 0, 0, 0, C, betas)

    # calculate fixed terms in log-likelihood
    Ris = cal_R_6_alpha_dia_bp(events_dict, betas)
    T_diff_sums = cal_diff_sums_Q(events_dict, end_time, betas)

    # initialize parameters (mu, alpha_s, alpha_r, alpha_tc, alpha_gr, c1, ..., cQ)
    mu_i = np.random.uniform(1e-6, 1e-2)
    alpha_s_i, alpha_r_i = np.random.uniform(0.1, 0.5, 2)
    alpha_tc_i, alpha_gr_i, alpha_al_i, alpha_alr_i = np.random.uniform(1e-5, 0.1, 4)
    mu_alpha_init = [mu_i, alpha_s_i, alpha_r_i, alpha_tc_i, alpha_gr_i, alpha_al_i, alpha_alr_i] # <-- random initialization
    # mu_alpha_init = [1e-2, 2e-2, 2e-2, 1e-2, 1e-2, 1e-2, 1e-2]  # <-- fixed initialization
    C = [1 / Q] * Q  # <-- fixed initialization
    init_param = tuple(mu_alpha_init + C)
    # define bounds
    mu_alpha_bo = [(1e-7, None)] * 7
    C_bo = [(0, 1)] * Q
    bounds = tuple(mu_alpha_bo + C_bo)

    # minimize function
    # to print optimization details , options={'disp': True}
    res = minimize(NLL_6_alpha_dia_bp, init_param, method='L-BFGS-B', bounds=bounds, jac=NLL_6_alpha_dia_bp_jac,
                   args=(betas, events_dict, end_time, n_a, m, T_diff_sums, Ris), tol=1e-12)
    results = res.x
    # print("success ", res.success, ", status ", res.status, ", fun value ", res.fun)
    # print("message ", res.message)
    mu, alpha_s, alpha_r, alpha_tc, alpha_gr, alpha_al, alpha_alr = results[:7]
    C = np.array(results[7:])

    # scaling step
    C_sum = np.sum(C)
    if C_sum != 0:
        C = C / C_sum
        alpha_s = alpha_s * C_sum
        alpha_r = alpha_r * C_sum
        alpha_tc = alpha_tc * C_sum
        alpha_gr = alpha_gr * C_sum
        alpha_al = alpha_al * C_sum
        alpha_alr = alpha_alr * C_sum

    return (mu, alpha_s, alpha_r, alpha_tc, alpha_gr, alpha_al, alpha_alr, C, betas)

#%% 6-alpha off-diagonal block pair log-likelihood and fit functions
def cal_R_6_alpha_off_bp(events_dict, events_dict_r, betas):
    """
    calculate recursive term in log-likelihood of one off-diagonal block pair (a, b) - 6 excitations

    :param events_dict: dictionary of events within block pair (a, b),
        where {(u, v) node pair in (a, b) : [t1, t2, ..] array of events between (u, v)}
    :param events_dict_r: dictionary of events within reciprocal block pair (b, a)
    :param betas: (Q,) array of decays
    :return: list of recursive function values for each (u, v) in events_dict
    """
    Q = len(betas)
    Ris = []
    for (u, v) in events_dict:
        # array of events of node pair (u,v)
        uv_events = events_dict[(u, v)]
        num_events_uv = len(uv_events)
        if num_events_uv == 0:
            Ris.append(np.array([0]))  # R=0 if node_pair (u,v) has no events
        else:
            # 6*Q columns (alpha_s, alpha_r, alpha_tc, alpha_gr, alpha_al, alpha_alr)*Q
            Ri = np.zeros((num_events_uv, 6 * Q))
            uv_intertimes = (uv_events[1:] - uv_events[:-1])
            # (#_uv_events-1, Q) array
            e_intertimes_Q = np.zeros((len(uv_intertimes), Q))
            for q in range(Q):
                e_intertimes_Q[:, q] = np.exp(-betas[q] * uv_intertimes)

            # loop through node pairs in block pair ab
            for (x, y) in events_dict:
                # same node_pair events (alpha_s)
                if (u, v) == (x, y):
                    for k in range(1, num_events_uv):
                        for q in range(Q):
                            Ri[k, 0 + q * 6] = e_intertimes_Q[k - 1, q] * (1 + Ri[k - 1, 0 + q * 6])
                # br node_pairs events (alpha_tc)
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

def LL_6_alpha_off_bp(params, ed, ed_r, end_time, n_b, m_ab, T_diff_sums=None, T_diff_sums_r=None, Ris=None):
    """
    calculate log-likelihood of one off-diagonal block pair - 6 excitation types

    :param tuple params: MULCH block pair parameters (mu, alpha_1, .. alpha_n, C, betas)
    :param dict ed: dictionary of events within a block pair (a, b),
        where {(u, v) node pair in (a, b) : [t1, t2, ..] array of events between (u, v)}
    :param dict ed_r: dictionary of events within reciprocal block pair (b, a)
    :param float end_time: duration of the network
    :param int n_b: number of nodes in block b
    :param int m_ab: number of node pairs in off-diagonal block pair (a, b)
    :param T_diff_sums: Optional (Q,) array(float)
    :param Ris: Optional list(size of events_dict)
    :return: block pair log-likelihood
    :rtype: float
    """
    # C: scaling parameters - same length as betas
    mu, alpha_s, alpha_r, alpha_tc, alpha_gr, alpha_al, alpha_alr, C, betas = params
    Q = len(betas)
    N_a = m_ab // n_b
    ##first term
    first = -m_ab * mu * end_time
    ##second term
    if T_diff_sums is None:
        T_diff_sums = cal_diff_sums_Q(ed, end_time, betas)
        T_diff_sums_r = cal_diff_sums_Q(ed_r, end_time, betas)
    second = -(alpha_s + alpha_tc * (n_b - 1) + alpha_al * (N_a - 1)) * C @ T_diff_sums
    second -= (alpha_r + alpha_gr * (n_b - 1) + alpha_alr * (N_a - 1)) * C @ T_diff_sums_r
    ##third term
    if Ris is None:
        Ris = cal_R_6_alpha_off_bp(ed, ed_r, betas)
    third = 0
    for i in range(len(Ris)):
        col_sum = np.zeros(Ris[i].shape[0])
        for q in range(Q):
            col_sum[:] += C[q] * betas[q]  * (alpha_s * Ris[i][:, 0 + q * 6] + alpha_r * Ris[i][:, 1 + q * 6] +
                                              alpha_tc * Ris[i][:, 2 + q * 6] + alpha_gr * Ris[i][:, 3 + q * 6] +
                                              alpha_al * Ris[i][:, 4 + q * 6] + alpha_alr * Ris[i][:,5 + q * 6])
        col_sum += mu
        third += np.sum(np.log(col_sum))
    log_likelihood_value = first + second + third
    return log_likelihood_value
def NLL_6_alpha_off_bp(p, betas, ed, ed_r, end_time, n_b, m_ab, T_diff_sums, T_diff_sums_r, Ris):
    """
    negative log-likelihood of an off-diagonal block pair (a,b) - 6 excitation type

    called by scipy.minimize() function for parameters estimation

    :param tuple p: MULCH block pair raveled parameters (mu, alpha_1, .., alpha_6, C_1, .., C_Q )
    :param betas: (Q,) array of decays
    :param dict ed: dictionary of events within a block pair (a, b),
        where {(u, v) node pair in (a, b) : [t1, t2, ..] array of events between (u, v)}
    :param dict ed_r: dictionary of events within reciprocal block pair (b, a)
    :param float end_time: duration of the network
    :param int n_b: number of nodes in block b
    :param int m_ab: number of node pairs in off-diagonal block pair (a, b)
    :param T_diff_sums: Optional (Q,) array(float)
    :param Ris: Optional list(size of events_dict)
    :return: block pair negative log-likelihood
    :rtype: float
    """
    mu, alpha_s, alpha_r, alpha_tc, alpha_gr, alpha_al, alpha_alr = p[:7]
    C = np.array(p[7:])
    # scaling step - constaint C sums to 1
    C_sum = np.sum(C)
    if (C_sum != 0):
        C = C / C_sum
        alpha_s = alpha_s * C_sum
        alpha_r = alpha_r * C_sum
        alpha_tc = alpha_tc * C_sum
        alpha_gr = alpha_gr * C_sum
        alpha_al = alpha_al * C_sum
        alpha_alr = alpha_alr * C_sum
    params = mu, alpha_s, alpha_r, alpha_tc, alpha_gr, alpha_al, alpha_alr, C, betas
    return -LL_6_alpha_off_bp(params, ed, ed_r, end_time, n_b, m_ab, T_diff_sums, T_diff_sums_r, Ris)
def NLL_6_alpha_off_bp_jac(p, betas, ed, ed_r, end_time, n_b, m_ab, T_diff_sums, T_diff_sums_r, Ris):
    """
    jacobian of negative log-likelihood of an off-diagonal block pair- 6 excitations

    called by scipy.minimize() function for parameters estimation

    :param tuple p: MULCH block pair raveled parameters (mu, alpha_1, .., alpha_6, C_1, .., C_Q )
    :param betas: (Q,) array of decays
    :param dict ed: dictionary of events within a block pair (a, b),
        where {(u, v) node pair in (a, b) : [t1, t2, ..] array of events between (u, v)}
    :param dict ed_r: dictionary of events within reciprocal block pair (b, a)
    :param float end_time: duration of the network
    :param int n_b: number of nodes in block b
    :param int m_ab: number of node pairs in off-diagonal block pair (a, b)
    :param T_diff_sums: Optional (Q,) array(float)
    :param Ris: Optional list(size of events_dict)
    :return: block pair negative log-likelihood
    :rtype: float
    """
    mu, alpha_s, alpha_r, alpha_tc, alpha_gr, alpha_al, alpha_alr = p[:7]
    C = np.array(p[7:])
    Q = len(C)
    N_a = m_ab // n_b
    # derivatives of second term
    d_mu = m_ab * end_time
    d_alphas = np.zeros(6)
    d_alphas[0] = C @ T_diff_sums
    d_alphas[1] = C @ T_diff_sums_r
    d_alphas[2] = (n_b - 1) * C @ T_diff_sums
    d_alphas[3] = (n_b - 1) * C @ T_diff_sums_r
    d_alphas[4] = (N_a - 1) * C @ T_diff_sums
    d_alphas[5] = (N_a - 1) * C @ T_diff_sums_r
    d_C = (alpha_s + alpha_tc * (n_b - 1) + alpha_al * (N_a - 1)) * T_diff_sums \
          + (alpha_r + alpha_gr * (n_b - 1) + alpha_alr * (N_a - 1)) * T_diff_sums_r
    # derivatives of third term
    for i in range(len(Ris)):
        denominator = np.zeros(Ris[i].shape[0])
        # one column for each alpha_j
        numerator_alphas = np.zeros((Ris[i].shape[0], 6))
        numerator_C = []
        for q in range(Q):
            numerator_C.append(betas[q]*( alpha_s * Ris[i][:, 0 + q * 6] + alpha_r * Ris[i][:, 1 + q * 6] +
                                          alpha_tc * Ris[i][:, 2 + q * 6] + alpha_gr * Ris[i][:,3 + q * 6] +
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
def fit_6_alpha_off_bp(ed, ed_r, end_time, n_b, m_ab, betas):
    """
    fit mulch one off-diagonal block pair (a, b) - 6 excitations

    :param dict ed: dictionary of events within a block pair (a, b),
        where {(u, v) node pair in (a, b) : [t1, t2, ..] array of events between (u, v)}
    :param dict ed_r: dictionary of events within reciprocal block pair (b, a)
    :param float end_time: duration of the network
    :param int n_b: number of nodes in block b
    :param int m_ab: number of node pairs in off-diagonal block pair (a, b)
    :param betas: (Q,) array of decays
    :return: estimated parameters (mu, alpha_1,.. , alpha_n, C, betas)
    :rtype: tuple
    """
    Q = len(betas)
    # events_dict : (u,v):array_of_events
    if len(ed) == 0:  # handling empty block pair with no events
        # (mu, alpha_s, alpha_r, alpha_tc, alpha_gr, alpha_al, alpha_alr, C, betas)
        C = np.zeros(Q)
        return (1e-10, 0, 0, 0, 0, 0, 0, C, betas)

    # calculate fixed terms in log-likelihood
    Ris = cal_R_6_alpha_off_bp(ed, ed_r, betas)
    T_diff_sums = cal_diff_sums_Q(ed, end_time, betas)
    T_diff_sums_r = cal_diff_sums_Q(ed_r, end_time, betas)

    # initialize parameters
    # mu_alpha_init = [1e-2, 2e-2, 2e-2, 1e-2, 1e-2]  # <-- fixed initialization
    mu_i = np.random.uniform(1e-6, 1e-2)
    alpha_s_i, alpha_r_i = np.random.uniform(0.1, 0.5, 2)
    alpha_tc_i, alpha_gr_i, alpha_al_i, alpha_alr_i = np.random.uniform(1e-5, 0.1, 4)
    mu_alpha_init = [mu_i, alpha_s_i, alpha_r_i, alpha_tc_i, alpha_gr_i, alpha_al_i, alpha_alr_i] # <-- random initialization
    C = [1 / Q] * Q  # <-- fixed initialization
    init_param = tuple(mu_alpha_init + C)
    # define bounds
    mu_alpha_bo = [(1e-7, None)] * 7
    C_bo = [(0, 1)] * Q
    bounds = tuple(mu_alpha_bo + C_bo)

    # minimize function
    # options = {'disp': True}
    res = minimize(NLL_6_alpha_off_bp, init_param, method='L-BFGS-B', bounds=bounds, jac=NLL_6_alpha_off_bp_jac,
                   args=(betas, ed, ed_r, end_time, n_b, m_ab, T_diff_sums, T_diff_sums_r, Ris), tol=1e-12)
    results = res.x
    mu, alpha_s, alpha_r, alpha_tc, alpha_gr, alpha_al, alpha_alr = results[:7]
    C = np.array(results[7:])

    # scaling step -
    C_sum = np.sum(C)
    if C_sum != 0:
        C = C / C_sum
        alpha_s = alpha_s * C_sum
        alpha_r = alpha_r * C_sum
        alpha_tc = alpha_tc * C_sum
        alpha_gr = alpha_gr * C_sum
        alpha_al = alpha_al * C_sum
        alpha_alr = alpha_alr * C_sum
    return (mu, alpha_s, alpha_r, alpha_tc, alpha_gr, alpha_al, alpha_alr, C, betas)


#%% 4-alpha diagonal block pair log-likelihood and fit functions
def cal_R_4_alpha_dia_bp(events_dict, betas):
    """
    calculate recursive term in log-likelihood of a diagonal block pair (a, a) - 4 excitation types

    :param dict events_dict: dictionary of events in a block pair (a, b),
        where {(u, v) node pair in (a, b) : [t1, t2, ..] array of events between (u, v)}
    :param betas: (Q,) array of decays
    :return: list of recursive function values for each (u, v) in events_dict
    """
    Q = len(betas)
    Ris = []
    for (u, v) in events_dict:
        # array of events of node pair (u,v)
        uv_events = events_dict[(u, v)]
        num_events_uv = len(uv_events)
        if num_events_uv == 0:
            Ris.append(np.array([0]))  # R=0 if node_pair (u,v) has no events
        else:
            # 4*Q columns (alpha_s, alpha_r, alpha_tc, alpha_gr)*Q
            Ri = np.zeros((num_events_uv, 4 * Q))
            uv_intertimes = (uv_events[1:] - uv_events[:-1])
            # (#_uv_events-1, Q) array
            e_intertimes_Q = np.zeros((len(uv_intertimes), Q))
            for q in range(Q):
                e_intertimes_Q[:, q] = np.exp(-betas[q] * uv_intertimes)
            for (x, y) in events_dict:
                if x == u or y == u:
                    prev_index = 0
                    # same node_pair events (alpha_s)
                    if (u, v) == (x, y):
                        for k in range(1, num_events_uv):
                            for q in range(Q):
                                Ri[k, 0 + q * 4] = e_intertimes_Q[k - 1, q] * (1 + Ri[k - 1, 0 + q * 4])
                    # reciprocal node_pair events (alpha_r)
                    elif (v, u) == (x, y):
                        Ri_temp = get_Ri_temp_Q(uv_events, e_intertimes_Q, events_dict[(x, y)], betas)
                        for q in range(Q):
                            Ri[:, 1 + q * 4] = Ri_temp[:, q]
                    # br node_pairs events (alpha_tc)
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
def LL_4_alpha_dia_bp(params, events_dict, end_time, n_a, m, T_diff_sums=None, Ris=None):
    """
    calculate log-likelihood of one diagonal block pair - 4 excitation types

    :param tuple params: MULCH block pair parameters (mu, alpha_1, .. alpha_n, C, betas)
    :param dict dict events_dict: dictionary of events within a block pair (a, b),
        where {(u, v) node pair in (a, b) : [t1, t2, ..] array of events between (u, v)}
    :param float end_time: duration of the network
    :param int n_a: number of nodes in block a
    :param int m: number of node pairs in the diagonal block pair
    :param T_diff_sums: Optional (Q,) array(float)
    :param Ris: Optional list(size of events_dict)
    :return: block pair log-likelihood
    :rtype: float
    """
    mu, alpha_s, alpha_r, alpha_tc, alpha_gr, C, betas = params
    Q = len(betas)
    ##first term
    first = -m * mu * end_time
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
    second = -(alpha_s + alpha_r + (alpha_tc + alpha_gr) * (n_a - 2)) * C @ T_diff_sums
    ##third term
    if Ris is None:
        Ris = cal_R_4_alpha_dia_bp(events_dict, betas)
    third = 0
    for i in range(len(Ris)):
        col_sum = np.zeros(Ris[i].shape[0])
        for q in range(Q):
            col_sum[:] += C[q]*betas[q] * (alpha_s * Ris[i][:, 0 + q * 4] + alpha_r * Ris[i][:, 1 + q * 4] +
                                  alpha_tc * Ris[i][:, 2 + q * 4] + alpha_gr * Ris[i][:,3 + q * 4])
        col_sum += mu
        third += np.sum(np.log(col_sum))
    log_likelihood_value = first + second + third
    return log_likelihood_value
def NLL_4_alpha_dia_bp(p, betas, events_dict, end_time, n_a, m, T_diff_sums, Ris):
    """
    negative log-likelihood of one diagonal block pair- 4 excitation type

    called by scipy.minimize() function for parameters estimation

    :param tuple p: MULCH block pair raveled parameters (mu, alpha_1, .., alpha_6, C_1, .., C_Q )
    :param betas: (Q,) array of decays
    :param dict events_dict: dictionary of events within a block pair (a, b),
        where {(u, v) node pair in (a, b) : [t1, t2, ..] array of events between (u, v)}
    :param float end_time: duration of the network
    :param int n_a: number of nodes in block a
    :param int m: number of node pairs in the diagonal block pair
    :param T_diff_sums: Optional (Q,) array(float)
    :param Ris: Optional list(size of events_dict)
    :return: block pair negative log-likelihood
    :rtype: float
    """
    mu, alpha_s, alpha_r, alpha_tc, alpha_gr = p[:5]
    C = np.array(p[5:])

    # scaling step - constaint C sums to 1
    C_sum = np.sum(C)
    if (C_sum != 0):
        C = C / C_sum
        alpha_s = alpha_s * C_sum
        alpha_r = alpha_r * C_sum
        alpha_tc = alpha_tc * C_sum
        alpha_gr = alpha_gr * C_sum

    params = mu, alpha_s, alpha_r, alpha_tc, alpha_gr, C, betas
    return -LL_4_alpha_dia_bp(params, events_dict, end_time, n_a, m, T_diff_sums, Ris)
def NLL_4_alpha_dia_bp_jac(p, betas, events_dict, end_time, n_a, m, T_diff_sums, Ris):
    """
    jacobian of negative log-likelihood of one diagonal block pair- 4 excitations

    called by scipy.minimize() function for parameters estimation

    :param tuple p: MULCH block pair parameters (mu, alpha_1, .., alpha_6, C_1, .., C_Q )
    :param betas: (Q,) array of decays
    :param dict events_dict: dictionary of events within a block pair (a, b),
        where {(u, v) node pair in (a, b) : [t1, t2, ..] array of events between (u, v)}
    :param float end_time: duration of the network
    :param int n_a: number of nodes in block a
    :param int m: number of node pairs in the diagonal block pair
    :param T_diff_sums: Optional (Q,) array(float)
    :param Ris: Optional list(size of events_dict)
    :return: jacobian array of negative log-likelihood function with respect to MULCH parameters
    """
    mu, alpha_s, alpha_r, alpha_tc, alpha_gr = p[:5]
    C = np.array(p[5:])
    Q = len(C)
    # derivatives of second term
    d_mu = m * end_time
    d_alpha_s = C @ T_diff_sums
    d_alpha_r = C @ T_diff_sums
    d_alpha_tc = (n_a - 2) * C @ T_diff_sums
    d_alpha_gr = (n_a - 2) * C @ T_diff_sums
    d_C = (alpha_s + alpha_r + (n_a - 2) * (alpha_tc + alpha_gr)) * T_diff_sums
    # derivatives of third term
    for i in range(len(Ris)):
        denominator = np.zeros(Ris[i].shape[0])
        # one column for each alpha_j
        numerator_alphas = np.zeros((Ris[i].shape[0], 4))
        numerator_C = []
        for q in range(Q):
            numerator_C.append(betas[q]* (alpha_s * Ris[i][:, 0 + q * 4] + alpha_r * Ris[i][:, 1 + q * 4] +
                               alpha_tc * Ris[i][:, 2 + q * 4] + alpha_gr * Ris[i][:,3 + q * 4]))
            denominator += C[q] * numerator_C[q]
            for j in range(4):
                numerator_alphas[:, j] += C[q] * betas[q] * Ris[i][:, j + q * 4]
        denominator += mu
        d_mu -= np.sum(1 / denominator)
        d_alpha_s -= np.sum(numerator_alphas[:, 0] / denominator)
        d_alpha_r -= np.sum(numerator_alphas[:, 1] / denominator)
        d_alpha_tc -= np.sum(numerator_alphas[:, 2] / denominator)
        d_alpha_gr -= np.sum(numerator_alphas[:, 3] / denominator)
        for q in range(Q):
            d_C[q] -= np.sum(numerator_C[q] / denominator)
    return np.hstack((d_mu, d_alpha_s, d_alpha_r, d_alpha_tc, d_alpha_gr, d_C))
def fit_4_alpha_dia_bp(events_dict, end_time, n_a, m, betas):
    """
    fit mulch one diagonal block pair (a, b) - 4 excitations

    :param dict events_dict: dictionary of events within a block pair (a, b),
        where {(u, v) node pair in (a, b) : [t1, t2, ..] array of events between (u, v)}
    :param float end_time: duration of the network
    :param int n_a: number of nodes in block a
    :param int m: number of node pairs in the diagonal block pair
    :param betas: (Q,) array of decays
    :return: estimated parameters (mu, alpha_1,.. , alpha_n, C, betas)
    :rtype: tuple
    """
    Q = len(betas)
    # events_dict : (u,v):array_of_events
    if len(events_dict) == 0:  # handling empty block pair with no events
        # (mu, alpha_s, alpha_r, alpha_tc, alpha_gr, C, betas)
        C = np.zeros(Q)
        return (1e-10, 0, 0, 0, 0, C, betas)

    # calculate fixed terms in log-likelihood
    Ris = cal_R_4_alpha_dia_bp(events_dict, betas)
    events_array = list(events_dict.values())
    T_diff = end_time - np.concatenate(events_array)
    T_diff_sums = np.zeros(Q, )
    for q in range(Q):
        T_diff_sums[q] = np.sum(1 - np.exp(-betas[q] * T_diff))

    # initialize parameters (mu, alpha_s, alpha_r, alpha_tc, alpha_gr, c1, ..., cQ)
    mu_i = np.random.uniform(1e-6, 1e-2)
    alpha_s_i, alpha_r_i = np.random.uniform(0.1, 0.5, 2)
    alpha_tc_i, alpha_gr_i = np.random.uniform(1e-5, 0.1, 2)
    mu_alpha_init = [mu_i, alpha_s_i, alpha_r_i, alpha_tc_i, alpha_gr_i] # <-- random initialization
    # mu_alpha_init = [1e-2, 2e-2, 2e-2, 1e-2, 1e-2]  # <-- fixed initialization
    C = [1 / Q] * Q  # <-- fixed initialization
    init_param = tuple(mu_alpha_init + C)
    # define bounds
    mu_alpha_bo = [(1e-7, None)] * 5
    C_bo = [(0, 1)] * Q
    bounds = tuple(mu_alpha_bo + C_bo)

    # minimize function
    res = minimize(NLL_4_alpha_dia_bp, init_param, method='L-BFGS-B', bounds=bounds, jac=NLL_4_alpha_dia_bp_jac,
                   args=(betas, events_dict, end_time, n_a, m, T_diff_sums, Ris), tol=1e-12)
    results = res.x
    mu, alpha_s, alpha_r, alpha_tc, alpha_gr = results[:5]
    C = np.array(results[5:])

    # scaling step
    C_sum = np.sum(C)
    if C_sum != 0:
        C = C / C_sum
        alpha_s = alpha_s * C_sum
        alpha_r = alpha_r * C_sum
        alpha_tc = alpha_tc * C_sum
        alpha_gr = alpha_gr * C_sum

    return (mu, alpha_s, alpha_r, alpha_tc, alpha_gr, C, betas)


#%% 4-alpha off-diagonal block pair log-likelihood and fit functions
def cal_R_4_alpha_off_bp(events_dict, events_dict_r, betas):
    """
    calculate recursive term in log-likelihood of one off-diagonal block pair (a, b) - 4 excitations

    :param events_dict: dictionary of events within block pair (a, b),
        where {(u, v) node pair in (a, b) : [t1, t2, ..] array of events between (u, v)}
    :param events_dict_r: dictionary of events within reciprocal block pair (b, a)
    :param betas: (Q,) array of decays
    :return: list of recursive function values for each (u, v) in events_dict
    """
    Q = len(betas)
    Ris = []
    for (u, v) in events_dict:
        # array of events of node pair (u,v)
        uv_events = events_dict[(u, v)]
        num_events_uv = len(uv_events)
        if num_events_uv == 0:
            Ris.append(np.array([0]))  # R=0 if node_pair (u,v) has no events
        else:
            # 4*Q columns (alpha_s, alpha_r, alpha_tc, alpha_gr)*Q
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
                    # same node_pair events (alpha_s)
                    if (u, v) == (x, y):
                        for k in range(1, num_events_uv):
                            for q in range(Q):
                                Ri[k, 0 + q * 4] = e_intertimes_Q[k - 1, q] * (1 + Ri[k - 1, 0 + q * 4])
                    # br node_pairs events (alpha_tc)
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
def LL_4_alpha_off_bp(params, ed, ed_r, end_time, n_b, m_ab, T_diff_sums=None, T_diff_sums_r=None, Ris=None):
    """
    calculate log-likelihood of one off-diagonal block pair - 4 excitation types

    :param tuple params: MULCH block pair parameters (mu, alpha_1, .. alpha_n, C, betas)
    :param dict ed: dictionary of events within a block pair (a, b),
        where {(u, v) node pair in (a, b) : [t1, t2, ..] array of events between (u, v)}
    :param dict ed_r: dictionary of events within reciprocal block pair (b, a)
    :param float end_time: duration of the network
    :param int n_b: number of nodes in block b
    :param int m_ab: number of node pairs in off-diagonal block pair (a, b)
    :param T_diff_sums: Optional (Q,) array(float)
    :param Ris: Optional list(size of events_dict)
    :return: block pair log-likelihood
    :rtype: float
    """
    # C: scaling parameters - same length as betas
    mu, alpha_s, alpha_r, alpha_tc, alpha_gr, C, betas = params
    Q = len(betas)
    ##first term
    first = -m_ab * mu * end_time
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
    second = -(alpha_s + alpha_tc * (n_b - 1)) * C @ T_diff_sums
    second -= (alpha_r + alpha_gr * (n_b - 1)) * C @ T_diff_sums_r
    ##third term
    if Ris is None:
        Ris = cal_R_4_alpha_off_bp(ed, ed_r, betas)
    third = 0
    for i in range(len(Ris)):
        col_sum = np.zeros(Ris[i].shape[0])
        for q in range(Q):
            col_sum[:] += C[q] * betas[q]  * (alpha_s * Ris[i][:, 0 + q * 4] + alpha_r * Ris[i][:, 1 + q * 4] +
                                              alpha_tc * Ris[i][:, 2 + q * 4] + alpha_gr * Ris[i][:, 3 + q * 4])
        col_sum += mu
        third += np.sum(np.log(col_sum))
    log_likelihood_value = first + second + third
    return log_likelihood_value
def NLL_4_alpha_off_bp(p, betas, ed, ed_r, end_time, n_b, m_ab, T_diff_sums, T_diff_sums_r, Ris):
    """
    negative log-likelihood of an off-diagonal block pair (a,b) - 4 excitation type

    called by scipy.minimize() function for parameters estimation

    :param tuple p: MULCH block pair raveled parameters (mu, alpha_1, .., alpha_6, C_1, .., C_Q )
    :param betas: (Q,) array of decays
    :param dict ed: dictionary of events within a block pair (a, b),
        where {(u, v) node pair in (a, b) : [t1, t2, ..] array of events between (u, v)}
    :param dict ed_r: dictionary of events within reciprocal block pair (b, a)
    :param float end_time: duration of the network
    :param int n_b: number of nodes in block b
    :param int m_ab: number of node pairs in off-diagonal block pair (a, b)
    :param T_diff_sums: Optional (Q,) array(float)
    :param Ris: Optional list(size of events_dict)
    :return: block pair negative log-likelihood
    :rtype: float
    """
    mu, alpha_s, alpha_r, alpha_tc, alpha_gr = p[:5]
    C = np.array(p[5:])

    # scaling step - constaint C sums to 1
    C_sum = np.sum(C)
    if (C_sum != 0):
        C = C / C_sum
        alpha_s = alpha_s * C_sum
        alpha_r = alpha_r * C_sum
        alpha_tc = alpha_tc * C_sum
        alpha_gr = alpha_gr * C_sum

    params = mu, alpha_s, alpha_r, alpha_tc, alpha_gr, C, betas
    return -LL_4_alpha_off_bp(params, ed, ed_r, end_time, n_b, m_ab, T_diff_sums, T_diff_sums_r, Ris)
def NLL_4_alpha_off_bp_jac(p, betas, ed, ed_r, end_time, n_b, m_ab, T_diff_sums, T_diff_sums_r, Ris):
    """
    jacobian of negative log-likelihood of an off-diagonal block pair- 4 excitations

    called by scipy.minimize() function for parameters estimation

    :param tuple p: MULCH block pair raveled parameters (mu, alpha_1, .., alpha_6, C_1, .., C_Q )
    :param betas: (Q,) array of decays
    :param dict ed: dictionary of events within a block pair (a, b),
        where {(u, v) node pair in (a, b) : [t1, t2, ..] array of events between (u, v)}
    :param dict ed_r: dictionary of events within reciprocal block pair (b, a)
    :param float end_time: duration of the network
    :param int n_b: number of nodes in block b
    :param int m_ab: number of node pairs in off-diagonal block pair (a, b)
    :param T_diff_sums: Optional (Q,) array(float)
    :param Ris: Optional list(size of events_dict)
    :return: block pair negative log-likelihood
    :rtype: float
    """
    mu, alpha_s, alpha_r, alpha_tc, alpha_gr = p[:5]
    C = np.array(p[5:])
    Q = len(C)
    # derivatives of second term
    d_mu = m_ab * end_time
    d_alpha_s = C @ T_diff_sums
    d_alpha_r = C @ T_diff_sums_r
    d_alpha_tc = (n_b - 1) * C @ T_diff_sums
    d_alpha_gr = (n_b - 1) * C @ T_diff_sums_r
    d_C = (alpha_s + (n_b - 1) * alpha_tc) * T_diff_sums + (alpha_r + (n_b - 1) * alpha_gr) * T_diff_sums_r
    # derivatives of third term
    for i in range(len(Ris)):
        denominator = np.zeros(Ris[i].shape[0])
        # one column for each alpha_j
        numerator_alphas = np.zeros((Ris[i].shape[0], 4))
        numerator_C = []
        for q in range(Q):
            numerator_C.append(betas[q]*( alpha_s * Ris[i][:, 0 + q * 4] + alpha_r * Ris[i][:, 1 + q * 4] +
                                alpha_tc * Ris[i][:, 2 + q * 4] + alpha_gr * Ris[i][:,3 + q * 4]))
            denominator += C[q] * numerator_C[q]
            for j in range(4):
                numerator_alphas[:, j] += C[q] * betas[q] * Ris[i][:, j + q * 4]
        denominator += mu
        d_mu -= np.sum(1 / denominator)
        d_alpha_s -= np.sum(numerator_alphas[:, 0] / denominator)
        d_alpha_r -= np.sum(numerator_alphas[:, 1] / denominator)
        d_alpha_tc -= np.sum(numerator_alphas[:, 2] / denominator)
        d_alpha_gr -= np.sum(numerator_alphas[:, 3] / denominator)
        for q in range(Q):
            d_C[q] -= np.sum(numerator_C[q] / denominator)
    return np.hstack((d_mu, d_alpha_s, d_alpha_r, d_alpha_tc, d_alpha_gr, d_C))
def fit_4_alpha_off_bp(ed, ed_r, end_time, n_b, m_ab, betas):
    """
    fit mulch one off-diagonal block pair (a, b) - 4 excitations

    :param dict ed: dictionary of events within a block pair (a, b),
        where {(u, v) node pair in (a, b) : [t1, t2, ..] array of events between (u, v)}
    :param dict ed_r: dictionary of events within reciprocal block pair (b, a)
    :param float end_time: duration of the network
    :param int n_b: number of nodes in block b
    :param int m_ab: number of node pairs in off-diagonal block pair (a, b)
    :param betas: (Q,) array of decays
    :return: estimated parameters (mu, alpha_1,.. , alpha_n, C, betas)
    :rtype: tuple
    """
    Q = len(betas)
    # events_dict : (u,v):array_of_events
    if len(ed) == 0:  # handling empty block pair with no events
        # (mu, alpha_s, alpha_r, alpha_tc, alpha_gr, C, betas)
        C = np.zeros(Q)
        return (1e-10, 0, 0, 0, 0, C, betas)

    # calculate fixed terms in log-likelihood
    Ris = cal_R_4_alpha_off_bp(ed, ed_r, betas)
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
    alpha_s_i, alpha_r_i = np.random.uniform(0.1, 0.5, 2)
    alpha_tc_i, alpha_gr_i = np.random.uniform(1e-5, 0.1, 2)
    mu_alpha_init = [mu_i, alpha_s_i, alpha_r_i, alpha_tc_i, alpha_gr_i] # <-- random initialization
    C = [1 / Q] * Q  # <-- fixed initialization
    init_param = tuple(mu_alpha_init + C)
    # define bounds
    mu_alpha_bo = [(1e-7, None)] * 5
    C_bo = [(0, 1)] * Q
    bounds = tuple(mu_alpha_bo + C_bo)

    # minimize function
    res = minimize(NLL_4_alpha_off_bp, init_param, method='L-BFGS-B', bounds=bounds, jac=NLL_4_alpha_off_bp_jac,
                   args=(betas, ed, ed_r, end_time, n_b, m_ab, T_diff_sums, T_diff_sums_r, Ris), tol=1e-12)
    results = res.x
    mu, alpha_s, alpha_r, alpha_tc, alpha_gr = results[:5]
    C = np.array(results[5:])

    # scaling step -
    C_sum = np.sum(C)
    if C_sum != 0:
        C = C / C_sum
        alpha_s = alpha_s * C_sum
        alpha_r = alpha_r * C_sum
        alpha_tc = alpha_tc * C_sum
        alpha_gr = alpha_gr * C_sum

    return (mu, alpha_s, alpha_r, alpha_tc, alpha_gr, C, betas)


#%% 2-alpha diagonal block pair log-likelihood and fit functions

def cal_R_2_alpha_dia_bp(events_dict, betas):
    """
    calculate recursive term in log-likelihood of a diagonal block pair (a, a) - 2 excitation types

    :param dict events_dict: dictionary of events within a block pair (a, b),
        where {(u, v) node pair in (a, b) : [t1, t2, ..] array of events between (u, v)}
    :param betas: (Q,) array of decays
    :return: list of recursive function values for each (u, v) in events_dict
    """
    Q = len(betas)
    Ris = []
    for (u, v) in events_dict:
        # array of events of node pair (u,v)
        uv_events = events_dict[(u, v)]
        num_events_uv = len(uv_events)
        if num_events_uv == 0:
            Ris.append(np.array([0]))  # R=0 if node_pair (u,v) has no events
        else:
            # 2*Q columns (alpha_s, alpha_r)*Q
            Ri = np.zeros((num_events_uv, 2 * Q))
            uv_intertimes = (uv_events[1:] - uv_events[:-1])
            # (#_uv_events-1, Q) array
            e_intertimes_Q = np.zeros((len(uv_intertimes), Q))
            for q in range(Q):
                e_intertimes_Q[:, q] = np.exp(-betas[q] * uv_intertimes)
            for (x, y) in events_dict:
                if x == u or y == u:
                    prev_index = 0
                    # same node_pair events (alpha_s)
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
def LL_2_alpha_dia_bp(params, events_dict, end_time, n_a, m, T_diff_sums=None, Ris=None):
    """
    calculate log-likelihood of one diagonal block pair - 2 excitation types

    :param tuple params: MULCH block pair parameters (mu, alpha_1, alpha_2, C, betas)
    :param dict dict events_dict: dictionary of events within a block pair (a, b),
        where {(u, v) node pair in (a, b) : [t1, t2, ..] array of events between (u, v)}
    :param float end_time: duration of the network
    :param int n_a: number of nodes in block a
    :param int m: number of node pairs in the diagonal block pair
    :param T_diff_sums: Optional (Q,) array(float)
    :param Ris: Optional list(size of events_dict)
    :return: block pair log-likelihood
    :rtype: float
    """
    mu, alpha_s, alpha_r, C, betas = params
    Q = len(betas)
    ##first term
    first = -m * mu * end_time
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
    second = -(alpha_s + alpha_r) * C @ T_diff_sums
    ##third term
    if Ris is None:
        Ris = cal_R_2_alpha_dia_bp(events_dict, betas)
    third = 0
    for i in range(len(Ris)):
        col_sum = np.zeros(Ris[i].shape[0])
        for q in range(Q):
            col_sum[:] += C[q] * betas[q] * (alpha_s * Ris[i][:, 0 + q * 2] + alpha_r * Ris[i][:, 1 + q * 2])
        col_sum += mu
        third += np.sum(np.log(col_sum))
    log_likelihood_value = first + second + third
    return log_likelihood_value
def NLL_2_alpha_dia_bp(p, betas, events_dict, end_time, n_a, m, T_diff_sums, Ris):
    """
    negative log-likelihood of one diagonal block pair- 2 excitation type

    called by scipy.minimize() function for parameters estimation

    :param tuple p: MULCH block pair raveled parameters (mu, alpha_1, alpha_2, C_1, .., C_Q )
    :param betas: (Q,) array of decays
    :param dict events_dict: dictionary of events within a block pair (a, b),
        where {(u, v) node pair in (a, b) : [t1, t2, ..] array of events between (u, v)}
    :param float end_time: duration of the network
    :param int n_a: number of nodes in block a
    :param int m: number of node pairs in the diagonal block pair
    :param T_diff_sums: Optional (Q,) array(float)
    :param Ris: Optional list(size of events_dict)
    :return: block pair negative log-likelihood
    :rtype: float
    """
    mu, alpha_s, alpha_r = p[:3]
    C = np.array(p[3:])

    # scaling step - constaint C sums to 1
    C_sum = np.sum(C)
    if (C_sum != 0):
        C = C / C_sum
        alpha_s = alpha_s * C_sum
        alpha_r = alpha_r * C_sum

    params = mu, alpha_s, alpha_r, C, betas
    return -LL_2_alpha_dia_bp(params, events_dict, end_time, n_a, m, T_diff_sums, Ris)
def NLL_2_alpha_dia_bp_jac(p, betas, events_dict, end_time, n_a, m, T_diff_sums, Ris):
    """
    jacobian of negative log-likelihood of one diagonal block pair- 6 excitations

    called by scipy.minimize() function for parameters estimation

    :param tuple p: MULCH block pair parameters (mu, alpha_1, .., alpha_6, C_1, .., C_Q )
    :param betas: (Q,) array of decays
    :param dict events_dict: dictionary of events within a block pair (a, b),
        where {(u, v) node pair in (a, b) : [t1, t2, ..] array of events between (u, v)}
    :param float end_time: duration of the network
    :param int n_a: number of nodes in block a
    :param int m: number of node pairs in the diagonal block pair
    :param T_diff_sums: Optional (Q,) array(float)
    :param Ris: Optional list(size of events_dict)
    :return: jacobian array of negative log-likelihood function with respect to MULCH parameters
    """
    mu, alpha_s, alpha_r = p[:3]
    C = np.array(p[3:])
    Q = len(C)
    # derivatives of second term
    d_mu = m * end_time
    d_alpha_s = C @ T_diff_sums
    d_alpha_r = C @ T_diff_sums
    d_C = (alpha_s + alpha_r) * T_diff_sums
    # derivatives of third term
    for i in range(len(Ris)):
        denominator = np.zeros(Ris[i].shape[0])
        # one column for each alpha_j
        numerator_alphas = np.zeros((Ris[i].shape[0], 2))
        numerator_C = []
        for q in range(Q):
            numerator_C.append(betas[q] * (alpha_s * Ris[i][:, 0 + q * 2] + alpha_r * Ris[i][:, 1 + q * 2]))
            denominator += C[q] * numerator_C[q]
            for j in range(2):
                numerator_alphas[:, j] += C[q] * betas[q] * Ris[i][:, j + q * 2]
        denominator += mu
        d_mu -= np.sum(1 / denominator)
        d_alpha_s -= np.sum(numerator_alphas[:, 0] / denominator)
        d_alpha_r -= np.sum(numerator_alphas[:, 1] / denominator)
        for q in range(Q):
            d_C[q] -= np.sum(numerator_C[q] / denominator)
    return np.hstack((d_mu, d_alpha_s, d_alpha_r, d_C))
def fit_2_alpha_dia_bp(events_dict, end_time, n_a, m, betas):
    """
    fit mulch one diagonal block pair (a, b) - 2 excitations

    :param dict events_dict: dictionary of events within a block pair (a, b),
        where {(u, v) node pair in (a, b) : [t1, t2, ..] array of events between (u, v)}
    :param float end_time: duration of the network
    :param int n_a: number of nodes in block a
    :param int m: number of node pairs in the diagonal block pair
    :param betas: (Q,) array of decays
    :return: estimated parameters (mu, alpha_1, alpha_2, C, betas)
    :rtype: tuple
    """
    Q = len(betas)
    # events_dict : (u,v):array_of_events
    if len(events_dict) == 0:  # handling empty block pair with no events
        # (mu, alpha_s, alpha_r, C, betas)
        C = np.zeros(Q)
        return (1e-10, 0, 0, C, betas)

    # calculate fixed terms in log-likelihood
    Ris = cal_R_2_alpha_dia_bp(events_dict, betas)
    events_array = list(events_dict.values())
    T_diff = end_time - np.concatenate(events_array)
    T_diff_sums = np.zeros(Q, )
    for q in range(Q):
        T_diff_sums[q] = np.sum(1 - np.exp(-betas[q] * T_diff))

    # initialize parameters (mu, alpha_s, alpha_r, c1, ..., cQ)
    mu_i = np.random.uniform(1e-6, 1e-2)
    alpha_s_i, alpha_r_i = np.random.uniform(0.1, 0.5, 2)
    mu_alpha_init = [mu_i, alpha_s_i, alpha_r_i] # <-- random initialization
    # mu_alpha_init = [1e-2, 2e-2, 2e-2]  # <-- fixed initialization
    C = [1 / Q] * Q  # <-- fixed initialization
    init_param = tuple(mu_alpha_init + C)
    # define bounds
    mu_alpha_bo = [(1e-7, None)] * 3
    C_bo = [(0, 1)] * Q
    bounds = tuple(mu_alpha_bo + C_bo)

    # minimize function
    res = minimize(NLL_2_alpha_dia_bp, init_param, method='L-BFGS-B', bounds=bounds, jac=NLL_2_alpha_dia_bp_jac,
                   args=(betas, events_dict, end_time, n_a, m, T_diff_sums, Ris), tol=1e-12)
    results = res.x
    mu, alpha_s, alpha_r = results[:3]
    C = np.array(results[3:])

    # scaling step
    C_sum = np.sum(C)
    if C_sum != 0:
        C = C / C_sum
        alpha_s = alpha_s * C_sum
        alpha_r = alpha_r * C_sum

    return (mu, alpha_s, alpha_r, C, betas)

#%% 2-alpha off-diagonal block pair log-likelihood and fit functions
def cal_R_2_alpha_off_bp(events_dict, events_dict_r, betas):
    """
    calculate recursive term in log-likelihood of one off-diagonal block pair (a, b) - 2 excitations

    :param events_dict: dictionary of events within block pair (a, b),
        where {(u, v) node pair in (a, b) : [t1, t2, ..] array of events between (u, v)}
    :param events_dict_r: dictionary of events within reciprocal block pair (b, a)
    :param betas: (Q,) array of decays
    :return: list of recursive function values for each (u, v) in events_dict
    """
    Q = len(betas)
    Ris = []
    for (u, v) in events_dict:
        # array of events of node pair (u,v)
        uv_events = events_dict[(u, v)]
        num_events_uv = len(uv_events)
        if num_events_uv == 0:
            Ris.append(np.array([0]))  # R=0 if node_pair (u,v) has no events
        else:
            # 2*Q columns (alpha_s, alpha_r)*Q
            Ri = np.zeros((num_events_uv, 2 * Q))
            uv_intertimes = (uv_events[1:] - uv_events[:-1])
            # (#_uv_events-1, Q) array
            e_intertimes_Q = np.zeros((len(uv_intertimes), Q))
            for q in range(Q):
                e_intertimes_Q[:, q] = np.exp(-betas[q] * uv_intertimes)

            # loop through node pairs in block pair ab
            for (x, y) in events_dict:
                # same node_pair events (alpha_s)
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
def LL_2_alpha_off_bp(params, ed, ed_r, end_time, n_b, m_ab, T_diff_sums=None, T_diff_sums_r=None, Ris=None):
    """
    calculate log-likelihood of one off-diagonal block pair - 4 excitation types

    :param tuple params: MULCH block pair parameters (mu, alpha_1, alpha_2, C, betas)
    :param dict ed: dictionary of events within a block pair (a, b),
        where {(u, v) node pair in (a, b) : [t1, t2, ..] array of events between (u, v)}
    :param dict ed_r: dictionary of events within reciprocal block pair (b, a)
    :param float end_time: duration of the network
    :param int n_b: number of nodes in block b
    :param int m_ab: number of node pairs in off-diagonal block pair (a, b)
    :param T_diff_sums: Optional (Q,) array(float)
    :param Ris: Optional list(size of events_dict)
    :return: block pair log-likelihood
    :rtype: float
    """
    # C: scaling parameters - same length as betas
    mu, alpha_s, alpha_r, C, betas = params
    Q = len(betas)
    ##first term
    first = -m_ab * mu * end_time
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
    second = -alpha_s * C @ T_diff_sums
    second -= alpha_r * C @ T_diff_sums_r
    ##third term
    if Ris is None:
        Ris = cal_R_2_alpha_off_bp(ed, ed_r, betas)
    third = 0
    for i in range(len(Ris)):
        col_sum = np.zeros(Ris[i].shape[0])
        for q in range(Q):
            col_sum[:] += C[q] * betas[q] * (alpha_s * Ris[i][:, 0 + q * 2] + alpha_r * Ris[i][:, 1 + q * 2])
        col_sum += mu
        third += np.sum(np.log(col_sum))
    log_likelihood_value = first + second + third
    return log_likelihood_value
def NLL_2_alpha_off_bp(p, betas, ed, ed_r, end_time, n_b, m_ab, T_diff_sums, T_diff_sums_r, Ris):
    """
    negative log-likelihood of an off-diagonal block pair (a,b) - 2 excitation type

    called by scipy.minimize() function for parameters estimation

    :param tuple p: MULCH block pair raveled parameters (mu, alpha_1, .., alpha_6, C_1, .., C_Q )
    :param betas: (Q,) array of decays
    :param dict ed: dictionary of events within a block pair (a, b),
        where {(u, v) node pair in (a, b) : [t1, t2, ..] array of events between (u, v)}
    :param dict ed_r: dictionary of events within reciprocal block pair (b, a)
    :param float end_time: duration of the network
    :param int n_b: number of nodes in block b
    :param int m_ab: number of node pairs in off-diagonal block pair (a, b)
    :param T_diff_sums: Optional (Q,) array(float)
    :param Ris: Optional list(size of events_dict)
    :return: block pair negative log-likelihood
    :rtype: float
    """
    mu, alpha_s, alpha_r = p[:3]
    C = np.array(p[3:])

    # scaling step - constaint C sums to 1
    C_sum = np.sum(C)
    if (C_sum != 0):
        C = C / C_sum
        alpha_s = alpha_s * C_sum
        alpha_r = alpha_r * C_sum

    params = mu, alpha_s, alpha_r, C, betas
    return -LL_2_alpha_off_bp(params, ed, ed_r, end_time, n_b, m_ab, T_diff_sums, T_diff_sums_r, Ris)
def NLL_2_alpha_off_bp_jac(p, betas, ed, ed_r, end_time, n_b, m_ab, T_diff_sums, T_diff_sums_r, Ris):
    """
    jacobian of negative log-likelihood of an off-diagonal block pair- 2 excitations

    called by scipy.minimize() function for parameters estimation

    :param tuple p: MULCH block pair raveled parameters (mu, alpha_1, .., alpha_6, C_1, .., C_Q )
    :param betas: (Q,) array of decays
    :param dict ed: dictionary of events within a block pair (a, b),
        where {(u, v) node pair in (a, b) : [t1, t2, ..] array of events between (u, v)}
    :param dict ed_r: dictionary of events within reciprocal block pair (b, a)
    :param float end_time: duration of the network
    :param int n_b: number of nodes in block b
    :param int m_ab: number of node pairs in off-diagonal block pair (a, b)
    :param T_diff_sums: Optional (Q,) array(float)
    :param Ris: Optional list(size of events_dict)
    :return: block pair negative log-likelihood
    :rtype: float
    """
    mu, alpha_s, alpha_r = p[:3]
    C = np.array(p[3:])
    Q = len(C)
    # derivatives of second term
    d_mu = m_ab * end_time
    d_alpha_s = C @ T_diff_sums
    d_alpha_r = C @ T_diff_sums_r
    d_C = alpha_s * T_diff_sums + alpha_r * T_diff_sums_r
    # derivatives of third term
    for i in range(len(Ris)):
        denominator = np.zeros(Ris[i].shape[0])
        # one column for each alpha_j
        numerator_alphas = np.zeros((Ris[i].shape[0], 2))
        numerator_C = []
        for q in range(Q):
            numerator_C.append(betas[q] * (alpha_s * Ris[i][:, 0 + q * 2] + alpha_r * Ris[i][:, 1 + q * 2]))
            denominator += C[q] * numerator_C[q]
            for j in range(2):
                numerator_alphas[:, j] += C[q] * betas[q] * Ris[i][:, j + q * 2]
        denominator += mu
        d_mu -= np.sum(1 / denominator)
        d_alpha_s -= np.sum(numerator_alphas[:, 0] / denominator)
        d_alpha_r -= np.sum(numerator_alphas[:, 1] / denominator)
        for q in range(Q):
            d_C[q] -= np.sum(numerator_C[q] / denominator)
    return np.hstack((d_mu, d_alpha_s, d_alpha_r, d_C))
def fit_2_alpha_off_bp(ed, ed_r, end_time, n_b, m_ab, betas):
    """
    fit mulch one off-diagonal block pair (a, b) - 2 excitations

    :param dict ed: dictionary of events within a block pair (a, b),
        where {(u, v) node pair in (a, b) : [t1, t2, ..] array of events between (u, v)}
    :param dict ed_r: dictionary of events within reciprocal block pair (b, a)
    :param float end_time: duration of the network
    :param int n_b: number of nodes in block b
    :param int m_ab: number of node pairs in off-diagonal block pair (a, b)
    :param betas: (Q,) array of decays
    :return: estimated parameters (mu, alpha_1, alpha_2, C, betas)
    :rtype: tuple
    """
    Q = len(betas)
    # events_dict : (u,v):array_of_events
    if len(ed) == 0:  # handling empty block pair with no events
        # (mu, alpha_s, alpha_r, C, betas)
        C = np.zeros(Q)
        return (1e-10, 0, 0, C, betas)

    # calculate fixed terms in log-likelihood
    Ris = cal_R_2_alpha_off_bp(ed, ed_r, betas)
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

    # initialize parameters (mu, alpha_s, alpha_r, c1, ..., cQ)
    mu_i = np.random.uniform(1e-6, 1e-2)
    alpha_s_i, alpha_r_i = np.random.uniform(0.1, 0.5, 2)
    mu_alpha_init = [mu_i, alpha_s_i, alpha_r_i] # <-- random initialization
    # mu_alpha_init = [1e-2, 2e-2, 2e-2]  # <-- fixed initialization
    C = [1 / Q] * Q  # <-- fixed initialization
    init_param = tuple(mu_alpha_init + C)
    # define bounds
    mu_alpha_bo = [(1e-7, None)] * 3
    C_bo = [(0, 1)] * Q
    bounds = tuple(mu_alpha_bo + C_bo)

    # minimize function
    res = minimize(NLL_2_alpha_off_bp, init_param, method='L-BFGS-B', bounds=bounds, jac=NLL_2_alpha_off_bp_jac,
                   args=(betas, ed, ed_r, end_time, n_b, m_ab, T_diff_sums, T_diff_sums_r, Ris), tol=1e-12)
    results = res.x
    mu, alpha_s, alpha_r = results[:3]
    C = np.array(results[3:])

    # scaling step -
    C_sum = np.sum(C)
    if C_sum != 0:
        C = C / C_sum
        alpha_s = alpha_s * C_sum
        alpha_r = alpha_r * C_sum

    return (mu, alpha_s, alpha_r, C, betas)



# %% helper functions

def plot_kernel(alpha, betas, C, time_range):
    """plot decay kernel of one block pair"""
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


def print_param_kernels(param):
    """print parameters of one block pair"""
    if len(param) == 9:
        print(f"C = {param[7]}")
        print(f"mu={param[0]:.7f}, alpha_s={param[1]:.4f}, alpha_r={param[2]:.4f}, alpha_tc={param[3]:.4f},"
              f" alpha_gr={param[4]:.4f}, alpha_al={param[5]:.4f}, alpha_alr={param[6]:.4f}")
    elif len(param) == 7:
        print(f"C = {param[5]}")
        print(f"mu={param[0]:.7f}, alpha_s={param[1]:.4f}, alpha_r={param[2]:.4f}, alpha_tc={param[3]:.4f},"
              f" alpha_gr={param[4]:.4f}")
    elif len(param) == 5:
        print(f"C = {param[3]}")
        print(f"mu={param[0]:.7f}, alpha_s={param[1]:.4f}, alpha_r={param[2]:.4f}")


def cal_num_events(events_dict):
    """calculate number of events in an event_dict"""
    num_events = 0
    for events_array in events_dict.values():
        num_events += len(events_array)
    return num_events


def get_Ri_temp_Q(uv_events, e_intertimes_Q, xy_events, betas):
    """block pair log-likelihood calculation helper function"""
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


def cal_diff_sums_Q(events_dict, end_time, betas):
    """
    for each decay_q, calculate sum_ti {exp(- beta_q * (T - t_i))}

    :param dict events_dict: keys are node pairs in the block pair & values are array of events between node pairs
    :param float end_time: duration of the network
    :param betas: (Q,) array of decays
    :return: (Q,) array(float)
    """
    Q = len(betas) # number of decays
    T_diff_sums = np.zeros(Q, )
    if len(events_dict) == 0:
        return T_diff_sums
    events_array = list(events_dict.values())
    T_diff = end_time - np.concatenate(events_array)
    for q in range(Q):
        T_diff_sums[q] = np.sum(1 - np.exp(-betas[q] * T_diff))
    return T_diff_sums

#%% detailed log-likelihood functions
""" only for checking code """
def detailed_LL_sum_betas(params_array, C, betas, events_list, end_time, M, C_r=None):
    mu_array, alpha_matrix = params_array
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
    def np_list(nodes_a, nodes_b):
        np_list = []
        for i in nodes_a:
            for j in nodes_b:
                if i != j:
                    np_list.append((i, j))
        return np_list
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
    alpha_ab_ba = generate_bp.get_6_alphas_matrix_off_bp(alphas_ab, alphas_ab, Na, Nb)
    alpha_aa = generate_bp.get_6_alphas_matrix_dia_bp(alphas_aa, Na)
    alpha_bb = generate_bp.get_6_alphas_matrix_dia_bp(alphas_aa, Nb)
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

def test_simulate_fit_kernel_sum_dia_bp(n_alpha=6):
    betas = np.array([.01, 0.1, 15])
    C = np.array([0.3, 0.3, 0.4])
    if n_alpha == 2:
        param_sim = (0.001, .2, 0.3, C, betas)
    elif n_alpha == 4:
        param_sim = (0.001, .2, 0.3, 0.02, 0.004, C, betas)
    else:
        param_sim = (0.001, .2, 0.3, 0.02, 0.004, 0.01, 0.0001, C, betas)

    a_nodes = list(range(15))
    n_nodes = len(a_nodes)
    end_time_sim = 3000

    print(f"Simulate and fit diagonal sum of kernels ({n_alpha} alphas) at betas = {betas}")
    # sum of kernels diagonal block simulation
    events_list_dia, events_dict_dia = generate_bp.simulate_dia_bp(param_sim, a_nodes, end_time_sim, return_list=True)
    n_events = cal_num_events(events_dict_dia)
    print("\nsimulation #nodes = ", n_nodes, " Simulation duration = ", end_time_sim, " number of events = ", n_events)
    print("\nActual simulation parameters:")
    print_param_kernels(param_sim)

    # Actual parameters log-likelihood
    M = n_nodes * (n_nodes - 1)  # number of nodes pair per block pair
    if n_alpha == 2:
        ll_sum = LL_2_alpha_dia_bp(param_sim, events_dict_dia, end_time_sim, n_nodes, M)
    elif n_alpha == 4:
        ll_sum = LL_4_alpha_dia_bp(param_sim, events_dict_dia, end_time_sim, n_nodes, M)
    else:
        ll_sum = LL_6_alpha_dia_bp(param_sim, events_dict_dia, end_time_sim, n_nodes, M)
    print(f"Actual log-likelihood = {ll_sum}")
    # Actual parameters detailed log-likelihood (only used for code checking)
    mu_alpha_arrays = generate_bp.get_mu_array_alpha_matrix_dia_bp(param_sim[0], param_sim[1:n_alpha + 1], n_nodes)
    ll_detailed = detailed_LL_sum_betas(mu_alpha_arrays, C, betas, events_list_dia, end_time_sim, M)
    print(f"detailed log-likelihood (only for checking) = {ll_detailed}")


    # test NLL jacobian function
    T_diff_sums = cal_diff_sums_Q(events_dict_dia, end_time_sim, betas)
    eps = np.sqrt(np.finfo(float).eps)
    p = tuple(list(param_sim[:n_alpha+1]) + list(C))
    print(f"\nDerivates of NLL function Sum of kernels ({n_alpha}-alphas)")
    if n_alpha == 2:
        Ris = cal_R_2_alpha_dia_bp(events_dict_dia, betas)
        print("Finite-difference approximation of the gradient (only for checking)")
        print(approx_fprime(p, NLL_2_alpha_dia_bp, eps, betas, events_dict_dia, end_time_sim, n_nodes, M,
                            T_diff_sums, Ris))
        print("Analytical gradient")
        print(NLL_2_alpha_dia_bp_jac(p, betas, events_dict_dia, end_time_sim, n_nodes, M, T_diff_sums, Ris))
    elif n_alpha == 4:
        Ris = cal_R_4_alpha_dia_bp(events_dict_dia, betas)
        print("Finite-difference approximation of the gradient (only for checking)")
        print(approx_fprime(p, NLL_4_alpha_dia_bp, eps, betas, events_dict_dia, end_time_sim, n_nodes, M,
                            T_diff_sums, Ris))
        print("Analytical gradient")
        print(NLL_4_alpha_dia_bp_jac(p, betas, events_dict_dia, end_time_sim, n_nodes, M, T_diff_sums, Ris))
    else:
        Ris = cal_R_6_alpha_dia_bp(events_dict_dia, betas)
        print("Finite-difference approximation of the gradient (only for checking)")
        print(approx_fprime(p, NLL_6_alpha_dia_bp, eps, betas, events_dict_dia, end_time_sim, n_nodes, M, T_diff_sums, Ris))
        print("Analytical gradient")
        print(NLL_6_alpha_dia_bp_jac(p, betas, events_dict_dia, end_time_sim, n_nodes, M, T_diff_sums, Ris))

    # fit Sum of kernels diagonal block pair
    print("\nFit block pair")
    start_fit_time = time.time()
    if n_alpha == 2:
        est_params = fit_2_alpha_dia_bp(events_dict_dia, end_time_sim, n_nodes, M, betas)
        end_fit_time = time.time()
        estimated_ll = LL_2_alpha_dia_bp(est_params, events_dict_dia, end_time_sim, n_nodes, M)
    elif n_alpha == 4:
        est_params = fit_4_alpha_dia_bp(events_dict_dia, end_time_sim, n_nodes, M, betas)
        end_fit_time = time.time()
        estimated_ll = LL_4_alpha_dia_bp(est_params, events_dict_dia, end_time_sim, n_nodes, M)
    else:
        est_params = fit_6_alpha_dia_bp(events_dict_dia, end_time_sim, n_nodes, M, betas)
        end_fit_time = time.time()
        estimated_ll = LL_6_alpha_dia_bp(est_params, events_dict_dia, end_time_sim, n_nodes, M)
    print("\nFit parameters")
    print_param_kernels(est_params)
    print(f"fit log-likelihood = {estimated_ll}, time to fit = {(end_fit_time - start_fit_time):.4f} s")

def test_simulate_fit_kernel_sum_two_off_bp(n_alpha=6):
    # test simulation from sum of kernels (6 alphas)
    print(f"Simulate and fit (sum of kernels) two off-diagonal block pairs - ({n_alpha} alphas)")
    betas = np.array([0.01, 0.2, 15])
    C_ab = np.array([0.2, 0.3, 0.5])
    C_ba = np.array([0.4, 0.3, 0.3])
    if n_alpha == 2:
        p_ab = (0.001, 0.4, 0.3, C_ab, betas)
        p_ba = (0.002, 0.2, 0.4, C_ba, betas)
    elif n_alpha == 4:
        p_ab = (0.001, 0.4, 0.3, 0.02, 0.002, C_ab, betas)
        p_ba = (0.002, 0.2, 0.4, 0.01, 0.01, C_ba, betas)
    else:
        p_ab = (0.001, 0.4, 0.3, 0.02, 0.002, 0.01, 0.001, C_ab, betas)
        p_ba = (0.002, 0.2, 0.4, 0.01, 0.01, 0.03, 0.003, C_ba, betas)

    n_a, n_b = 12, 5
    a_nodes = list(np.arange(0, n_a))
    b_nodes = list(np.arange(n_a, n_b + n_a))
    end_time_sim = 3000
    M_ab = n_a * n_b

    print(f"Simulate and fit two off-diagonal sum of kernels block pairs ({n_alpha} alphas) at betas = {betas}")
    print("\nBlock pair (a, b) actual parameters:")
    print_param_kernels(p_ab)
    print("Block pair (b, a) actual parameters:")
    print_param_kernels(p_ba)
    l, d_ab, d_ba = generate_bp.simulate_off_bp(p_ab, p_ba, a_nodes, b_nodes, end_time_sim, return_list=True)
    print("#nodes_a = ", n_a, ", #nodes_b = ", n_b, ", Duration = ", end_time_sim,
          " #simulated events = ", cal_num_events(d_ab) + cal_num_events(d_ba))

    if n_alpha == 2:
        ll_ab_1 = LL_2_alpha_off_bp(p_ab, d_ab, d_ba, end_time_sim, n_b, M_ab)
        ll_ba_1 = LL_2_alpha_off_bp(p_ba, d_ba, d_ab, end_time_sim, n_a, M_ab)
    elif n_alpha == 4:
        ll_ab_1 = LL_4_alpha_off_bp(p_ab, d_ab, d_ba, end_time_sim, n_b, M_ab)
        ll_ba_1 = LL_4_alpha_off_bp(p_ba, d_ba, d_ab, end_time_sim, n_a, M_ab)
    else:
        ll_ab_1 = LL_6_alpha_off_bp(p_ab, d_ab, d_ba, end_time_sim, n_b, M_ab)
        ll_ba_1 = LL_6_alpha_off_bp(p_ba, d_ba, d_ab, end_time_sim, n_a, M_ab)

    print(f"actual log-likelihood = {ll_ab_1} + {ll_ba_1} = {ll_ab_1 + ll_ba_1}")
    mu_alpha_arrays = generate_bp.get_mu_array_alpha_matrix_off_bp(p_ab[0], p_ab[1:n_alpha+1], p_ba[0]
                                                                   , p_ba[1:n_alpha+1], n_a, n_b)
    ll_detailed = detailed_LL_sum_betas(mu_alpha_arrays, C_ab, betas, l, end_time_sim, 2 * M_ab, C_r=C_ba)
    print(f"detailed log-likelihood (only for checking) = ", ll_detailed)

    # test NLL jacobian function
    T_diff_sums = cal_diff_sums_Q(d_ab, end_time_sim, betas)
    T_diff_sums_r = cal_diff_sums_Q(d_ba, end_time_sim, betas)
    eps = np.sqrt(np.finfo(float).eps)
    p_ab_d = tuple(list(p_ab[:n_alpha + 1]) + list(C_ab))
    print(f"\nDerivates of negative log-likelihood function Sum of kernels ({n_alpha}-alphas)")
    if n_alpha == 2:
        Ris = cal_R_2_alpha_off_bp(d_ab, d_ba, betas)
        print("Finite-difference approximation of the gradient (only for checking)")
        print(approx_fprime(p_ab_d, NLL_2_alpha_off_bp, eps, betas, d_ab, d_ba, end_time_sim, n_b, M_ab,
                            T_diff_sums, T_diff_sums_r, Ris))
        print("Analytical gradient")
        print(NLL_2_alpha_off_bp_jac(p_ab_d, betas, d_ab, d_ba, end_time_sim, n_b, M_ab, T_diff_sums,
                                     T_diff_sums_r, Ris))
    elif n_alpha == 4:
        Ris = cal_R_4_alpha_off_bp(d_ab, d_ba, betas)
        print("Finite-difference approximation of the gradient (only for checking)")
        print(approx_fprime(p_ab_d, NLL_4_alpha_off_bp, eps, betas, d_ab, d_ba, end_time_sim, n_b, M_ab,
                            T_diff_sums, T_diff_sums_r, Ris))
        print("Analytical gradient")
        print(NLL_4_alpha_off_bp_jac(p_ab_d, betas, d_ab, d_ba, end_time_sim, n_b, M_ab, T_diff_sums,
                                     T_diff_sums_r, Ris))
    else:
        Ris = cal_R_6_alpha_off_bp(d_ab, d_ba, betas)
        print("Finite-difference approximation of the gradient (only for checking)")
        print(approx_fprime(p_ab_d, NLL_6_alpha_off_bp, eps, betas, d_ab, d_ba, end_time_sim, n_b, M_ab,
                            T_diff_sums, T_diff_sums_r, Ris))
        print("Analytical gradient")
        print(NLL_6_alpha_off_bp_jac(p_ab_d, betas, d_ab, d_ba, end_time_sim, n_b, M_ab, T_diff_sums,
                                     T_diff_sums_r, Ris))

    # fit sum of kernels on two reciprocal block pairs on actual and different betas
    for i in range(2):
        if i == 1: betas = [0.01, 20]
        print(f"\nFit two block pair at {betas}")
        # fit block pair (a, b)
        if n_alpha == 2:
            start_fit_time = time.time()
            est_p_ab = fit_2_alpha_off_bp(d_ab, d_ba, end_time_sim, n_b, M_ab, betas)
            end_fit_time = time.time()
            print("Estimated parameters of block pair (a, b):")
            print_param_kernels(est_p_ab)
            ll_ab_est = LL_2_alpha_off_bp(est_p_ab, d_ab, d_ba, end_time_sim, n_b, M_ab)
            print(f"fit time = {(end_fit_time - start_fit_time):.4f} s, ll_ab = {ll_ab_est}")
            # fit block pair (b, a)
            start_fit_time = time.time()
            est_p_ba = fit_2_alpha_off_bp(d_ba, d_ab, end_time_sim, n_a, M_ab, betas)
            end_fit_time = time.time()
            print("Estimated parameters of block pair (b, a):")
            print_param_kernels(est_p_ba)
            ll_ba_est = LL_2_alpha_off_bp(est_p_ba, d_ba, d_ab, end_time_sim, n_a, M_ab)
            print(f"fit time = {(end_fit_time - start_fit_time):.4f} s, ll_ab = {ll_ba_est}")
            print("total ll = ", (ll_ab_est + ll_ba_est))
        elif n_alpha == 4:
            start_fit_time = time.time()
            est_p_ab = fit_4_alpha_off_bp(d_ab, d_ba, end_time_sim, n_b, M_ab, betas)
            end_fit_time = time.time()
            print("Estimated parameters of block pair (a, b):")
            print_param_kernels(est_p_ab)
            ll_ab_est = LL_4_alpha_off_bp(est_p_ab, d_ab, d_ba, end_time_sim, n_b, M_ab)
            print(f"fit time = {(end_fit_time - start_fit_time):.4f} s, ll_ab = {ll_ab_est}")
            # fit block pair (b, a)
            start_fit_time = time.time()
            est_p_ba = fit_4_alpha_off_bp(d_ba, d_ab, end_time_sim, n_a, M_ab, betas)
            end_fit_time = time.time()
            print("Estimated parameters of block pair (b, a):")
            print_param_kernels(est_p_ba)
            ll_ba_est = LL_4_alpha_off_bp(est_p_ba, d_ba, d_ab, end_time_sim, n_a, M_ab)
            print(f"fit time = {(end_fit_time - start_fit_time):.4f} s, ll_ab = {ll_ba_est}")
            print("total ll = ", (ll_ab_est + ll_ba_est))
        else:
            start_fit_time = time.time()
            est_p_ab = fit_6_alpha_off_bp(d_ab, d_ba, end_time_sim, n_b, M_ab, betas)
            end_fit_time = time.time()
            print("Estimated parameters of block pair (a, b):")
            print_param_kernels(est_p_ab)
            ll_ab_est = LL_6_alpha_off_bp(est_p_ab, d_ab, d_ba, end_time_sim, n_b, M_ab)
            print(f"fit time = {(end_fit_time - start_fit_time):.4f} s, ll_ab = {ll_ab_est}")
            # fit block pair (b, a)
            start_fit_time = time.time()
            est_p_ba = fit_6_alpha_off_bp(d_ba, d_ab, end_time_sim, n_a, M_ab, betas)
            end_fit_time = time.time()
            print("Estimated parameters of block pair (b, a):")
            print_param_kernels(est_p_ba)
            ll_ba_est = LL_6_alpha_off_bp(est_p_ba, d_ba, d_ab, end_time_sim, n_a, M_ab)
            print(f"fit time = {(end_fit_time - start_fit_time):.4f} s, ll_ab = {ll_ba_est}")
            print("total ll = ", (ll_ab_est + ll_ba_est))


# %% code to test fitting function


if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    # Choose to simulate and fit 1 diagonal block pair OR 2 off-diagonal block pairs
    # Also, choose # of excitation types

    # test_simulate_fit_kernel_sum_two_off_bp(n_alpha=2)
    test_simulate_fit_kernel_sum_dia_bp(n_alpha=6)















