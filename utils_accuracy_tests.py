"""MULCH performance evaluation helper functions

This scripts contains helper functions for:
 - motifs count experiment
 - link prediction experiment
"""

import networkx as nx
import numpy as np
from bisect import bisect_left
# from scipy import integrate
from sklearn import metrics
import matplotlib.pyplot as plt
from utils_fit_bp import cal_num_events
from utils_fit_model import event_dict_to_adjacency
from utils_generate_model import simulate_mulch
from dynetworkx import ImpulseDiGraph, count_temporal_motif



# %% motif counts functions
def get_motifs():
    """return (6, 6) list of motifs formations"""
    motifs = [[((1, 2), (3, 2), (1, 2)), ((1, 2), (3, 2), (2, 1)), ((1, 2), (3, 2), (1, 3)),
               ((1, 2), (3, 2), (3, 1)),((1, 2), (3, 2), (2, 3)), ((1, 2), (3, 2), (3, 2))],
              [((1, 2), (2, 3), (1, 2)), ((1, 2), (2, 3), (2, 1)), ((1, 2), (2, 3), (1, 3)),
               ((1, 2), (2, 3), (3, 1)),((1, 2), (2, 3), (2, 3)), ((1, 2), (2, 3), (3, 2))],
              [((1, 2), (3, 1), (1, 2)), ((1, 2), (3, 1), (2, 1)), ((1, 2), (3, 1), (1, 3)),
               ((1, 2), (3, 1), (3, 1)),((1, 2), (3, 1), (2, 3)), ((1, 2), (3, 1), (3, 2))],
              [((1, 2), (1, 3), (1, 2)), ((1, 2), (1, 3), (2, 1)), ((1, 2), (1, 3), (1, 3)),
               ((1, 2), (1, 3), (3, 1)),((1, 2), (1, 3), (2, 3)), ((1, 2), (1, 3), (3, 2))],
              [((1, 2), (2, 1), (1, 2)), ((1, 2), (2, 1), (2, 1)), ((1, 2), (2, 1), (1, 3)),
               ((1, 2), (2, 1), (3, 1)),((1, 2), (2, 1), (2, 3)), ((1, 2), (2, 1), (3, 2))],
              [((1, 2), (1, 2), (1, 2)), ((1, 2), (1, 2), (2, 1)), ((1, 2), (1, 2), (1, 3)),
               ((1, 2), (1, 2), (3, 1)),((1, 2), (1, 2), (2, 3)), ((1, 2), (1, 2), (3, 2))]]
    return motifs


def cal_recip_trans_motif(events_dict, n, motif_delta, verbose=False):
    """
    calculate network's reciprocity, transitivity, and (6, 6) temporal motif matrix

    :param events_dict: dataset formatted as a dictionary {(u, v) node pairs in network : [t1, t2, ...] array of
        events between (u, v)}
    :param n: number of nodes in the network
    :param motif_delta: interval for temporal motifs counts
    :param verbose: (optional) print (6, 6) motif count matrix
    :return: reciprocity, transitivity, (6, 6) temporal motif matrix, number_events
    """
    adj = event_dict_to_adjacency(n, events_dict)
    net = nx.DiGraph(adj)
    recip = nx.overall_reciprocity(net)
    trans = nx.transitivity(net)
    # create ImpulseDiGraph from network
    G_data = ImpulseDiGraph()
    for (u, v) in events_dict:
        events_list_uv = events_dict[u, v]
        for t in events_list_uv:
            G_data.add_edge(u, v, t)
    if verbose:
        print(f"{1:>10}{2:>10}{3:>10}{4:>10}{5:>10}{6:>10}")
    motifs = get_motifs()
    dataset_motif = np.zeros((6, 6), dtype=int)
    for i in range(6):
        for j in range(6):
            dataset_motif[i, j] = count_temporal_motif(G_data, motifs[i][j], motif_delta)
        if verbose:
            print(
                f"{dataset_motif[i, 0]:>10}{dataset_motif[i, 1]:>10}{dataset_motif[i, 2]:>10}{dataset_motif[i, 3]:>10}"
                f"{dataset_motif[i, 4]:>10}{dataset_motif[i, 5]:>10}")
    n_events = cal_num_events(events_dict)
    return recip, trans, dataset_motif, n_events


def simulate_count_motif_experiment(dataset_motif_tuple, params_tup, nodes_mem, K, end_time_sim,
                                    motif_delta, n_sim=10,
                                    verbose=False):
    """
    Simulate networks from MULCH fit parameters and compute temporal motif counts

    :param dataset_motif_tuple: actual dataset (reciprocity, transitivity, motif_counts, dataset_n_events_train)
    :param params_tup: MULCH parameters (mu_bp, alphas_1_bp, .., alpha_s_bp, C_bp, betas )
    :param nodes_mem: block membership for nodes in train dataset
    :param K: number of classes
    :param end_time_sim: Simulation duration
    :param motif_delta: interval for temporal motifs counts
    :param n_sim: number of model's simulations to count motifs on
    :param verbose: if True, print results of each simulation
    :return: dictionary of motif experiment results {"sim_motif_avg": simulations average motif count,
        "sim_recip_avg": simulations average reciprocity, "sim_trans_avg": simulation average transitivity,
        "sim_n_events_avg": average number of events, "mape": MAPE score}
    """
    dataset_recip, dataset_trans, dataset_motif, dataset_n_events = dataset_motif_tuple
    # simulate and count motifs
    n_nodes = len(nodes_mem)
    _, block_count = np.unique(nodes_mem, return_counts=True)
    block_prob = block_count / sum(block_count)
    sim_motif_avg = np.zeros((6, 6))
    sim_motif_all = np.zeros((n_sim, 6, 6))
    sim_mape_all = np.zeros(n_sim)
    sim_n_events_avg, sim_trans_avg, sim_recip_avg = 0, 0, 0
    for run in range(n_sim):
        # simulate using fitted parameters
        if verbose:
            print("\n\tsimulation#", run)
        # simulate from MULCH
        events_dict_sim, _ = simulate_mulch(params_tup, n_nodes, K, block_prob, end_time_sim)
        # count reciprocity, transitivity, motif_counts, #events
        recip_sim, trans_sim, sim_motif, n_evens_sim = cal_recip_trans_motif(events_dict_sim,
                                                                             n_nodes,
                                                                             motif_delta, verbose)
        sim_mape_all[run] = 100 / 36 * np.sum(np.abs(sim_motif - (dataset_motif + 1))
                                              / (dataset_motif + 1))
        sim_motif_avg += sim_motif
        sim_motif_all[run, :, :] = sim_motif
        if verbose:
            print(f"\trecip={recip_sim:.4f}, trans={trans_sim:.4f}, n_events={n_evens_sim}"
                  f", MAPE={sim_mape_all[run]:.1f}")
        sim_recip_avg += recip_sim
        sim_trans_avg += trans_sim
        sim_n_events_avg += n_evens_sim
    # Average simulations results
    sim_motif_avg /= n_sim
    sim_recip_avg /= n_sim
    sim_trans_avg /= n_sim
    sim_n_events_avg /= n_sim
    sim_motif_median = np.median(sim_motif_all, axis=0)

    # calculate MAPE - NOTE: just added 1 to avoid division by 0
    mape = 100 / 36 * np.sum(np.abs(sim_motif_avg - (dataset_motif + 1)) / (dataset_motif + 1))

    # save results
    results_dict = {}
    results_dict["K"] = K
    results_dict["betas"] = params_tup[-1]
    results_dict["n_simulation"] = n_sim
    results_dict["motif_delta"] = motif_delta
    results_dict["parameters"] = params_tup
    results_dict["dataset_motif"] = dataset_motif
    results_dict["dataset_recip"] = dataset_recip
    results_dict["dataset_trans"] = dataset_trans
    results_dict["dataset_n_events"] = dataset_n_events
    results_dict["sim_motif_avg"] = sim_motif_avg
    results_dict["sim_motif_all"] = sim_motif_all
    results_dict["sim_motif_median"] = sim_motif_median
    results_dict["sim_recip_avg"] = sim_recip_avg
    results_dict["sim_trans_avg"] = sim_trans_avg
    results_dict["sim_n_events_avg"] = sim_n_events_avg
    results_dict["mape"] = mape
    results_dict["mape_all"] = sim_mape_all

    return results_dict


# %% link prediction test functions

def mulch_predict_probs_and_actual(n_nodes, t0, delta, events_dict, params_tup, nodes_mem_all):
    """
    For each node pair (u, v), computes probability of an event in interval [t0, t0 + delta)

    (u, v) event probability = 1 - exp( -integral_[t0:t0+delta]{intensity_uv(t)}
    for optimized computations, I defined 3 lists below dia_bp_events_dict, off_bp_events_dict0, off_bp_events_dict1,
    which are used to hold node pairs events computations for intensity_integral function.
    for each node pair (u, v), compute sum_q{C_q * [exp(-beta_q * delta)-1] * sum_t{-beta_q * t0 - t_uv}}.

    if (u, v) in diagonal bp(u_b, u_b):
        - use C parameter of (u_b, u_b), and store in dia_bp_events_dict[u_b]
    if (u, v) in off-diagonal bp(u_b, v_b), then computed twice:
        - one time use C parameter of bp(u_b, v_b), and store in off_bp_events_dict0[u_b][v_b].
        - other use bp(v_b, u_b) C parameter, and store in off_bp_events_dict1[u_b][v_b].


    :param n_nodes: number of nodes in network
    :param t0: start timestamp of test interval
    :param delta: length of test interval
    :param events_dict: full dataset formatted as a dictionary {(u, v) node pairs in network : [t1, t2, ...] array of
        events between (u, v)}
    :param params_tup: MULCH parameters (mu_bp, alphas_1_bp, .., alpha_n_bp, C_bp, betas )
    :param nodes_mem_all: block membership of all nodes in network
    :return: (n, n) array true link, (n, n) array of link prediction probabilities
    """
    n_alpha = len(params_tup) - 3  # number of excitation types
    if n_alpha == 6:
        mu_bp, alpha_s_bp, alpha_r_bp, alpha_tc_bp, alpha_gr_bp, alpha_al_bp, alpha_alr_bp, C_bp, betas = params_tup
    elif n_alpha == 4:
        mu_bp, alpha_s_bp, alpha_r_bp, alpha_tc_bp, alpha_gr_bp, C_bp, betas = params_tup
    else:  # n_alpha = 2
        mu_bp, alpha_s_bp, alpha_r_bp, C_bp, betas = params_tup

    K = len(mu_bp)  # number of blocks
    Q = len(betas)  # number of decays
    # 3 lists of necessary calculation for intensity_integral_function (see function description)
    dia_bp_events_dict = [{}] * K  # (K,) list for the K diagonal block pairs
    off_bp_events_dict0 = [[{}] * K for _ in
                           range(K)]  # (K,K) list for the K*(K-1) off-diagonal block pairs
    off_bp_events_dict1 = [[{}] * K for _ in
                           range(K)]  # (K,K) list for the K*(K-1) off-diagonal block pairs
    for u, v in events_dict:
        # blocks of node u, v
        u_b, v_b = nodes_mem_all[u], nodes_mem_all[v]
        t_uv = np.array(events_dict[(u, v)])
        # index of uv_timestamps < t0
        index = bisect_left(t_uv, t0)
        # array of events t0 - t_uv
        t0_minus_tuv = t0 - t_uv[:index]
        if u_b == v_b:
            exp_sum_q = 0
            C = C_bp[u_b, v_b]
            for q in range(Q):
                exp_sum_q += C[q] * (np.exp(-betas[q] * delta) - 1) * np.sum(
                    np.exp(-betas[q] * t0_minus_tuv))
            dia_bp_events_dict[u_b][(u, v)] = exp_sum_q
        else:
            exp_sum_q0, exp_sum_q1 = 0, 0
            C0, C1 = C_bp[u_b, v_b], C_bp[v_b, u_b]
            for q in range(Q):
                exp_sum_q0 += C0[q] * (np.exp(-betas[q] * delta) - 1) * np.sum(
                    np.exp(-betas[q] * t0_minus_tuv))
                exp_sum_q1 += C1[q] * (np.exp(-betas[q] * delta) - 1) * np.sum(
                    np.exp(-betas[q] * t0_minus_tuv))
            off_bp_events_dict0[u_b][v_b][(u, v)] = exp_sum_q0
            off_bp_events_dict1[u_b][v_b][(u, v)] = exp_sum_q1

    prob_dict = np.zeros((n_nodes, n_nodes))  # Predicted probs that link exists
    y = np.zeros((n_nodes, n_nodes))  # actual link
    for u in range(n_nodes):
        for v in range(n_nodes):
            if u != v:
                u_b, v_b = nodes_mem_all[u], nodes_mem_all[v]
                # block pair fit parameters
                if n_alpha == 6:
                    par = (mu_bp[u_b, v_b], alpha_s_bp[u_b, v_b], alpha_r_bp[u_b, v_b],
                           alpha_tc_bp[u_b, v_b],alpha_gr_bp[u_b, v_b], alpha_al_bp[u_b, v_b],
                           alpha_alr_bp[u_b, v_b],C_bp[u_b, v_b], betas)
                elif n_alpha == 4:
                    par = (mu_bp[u_b, v_b], alpha_s_bp[u_b, v_b], alpha_r_bp[u_b, v_b],
                           alpha_tc_bp[u_b, v_b],alpha_gr_bp[u_b, v_b], C_bp[u_b, v_b], betas)
                else:  # n_alpha =2
                    par = (mu_bp[u_b, v_b], alpha_s_bp[u_b, v_b], alpha_r_bp[u_b, v_b],
                           C_bp[u_b, v_b], betas)
                if u_b == v_b:
                    integral = mulch_uv_intensity_dia_integral(n_alpha, delta, (u, v), par,
                                                               dia_bp_events_dict[u_b])
                else:
                    integral = mulch_uv_intensity_off_integral(n_alpha, delta, (u, v), par,
                                                               off_bp_events_dict0[u_b][v_b],
                                                               off_bp_events_dict1[v_b][u_b])

                prob_dict[u, v] = 1 - np.exp(-integral)

                if (u, v) in events_dict:
                    uv_times = np.array(events_dict[(u, v)])
                    if len(uv_times[np.logical_and(uv_times >= t0, uv_times <= t0 + delta)]) > 0:
                        y[u, v] = 1

    return y, prob_dict


def mulch_uv_intensity_dia_integral(n_alpha, delta, uv, params, events_dict):
    """
    Optimized analytical integral of the intensity function of node pair (u, v) in diagonal block pair (u_b, v_b)

    :param n_alpha: number of excitation types (6, 4, or 2)
    :param delta: link prediction test period
    :param uv: node pair ids tuple (u, v)
    :param params: MULCH block pair (u_b, v_b) parameters tuple
    :param events_dict: dictionary of block pair (u_b, v_b) where {(x, y) node pair is bp(u_b, v_b) : x}
        ,where x = sum_q{C_q * [exp(-beta_q * delta)-1] * sum_txy{-beta_q * t0 - txy}}, and
        C=[C_q1, .., C_Q] is scaling parameter of bp(u_b, v_b)
    :return: (float) integral result
    """
    if n_alpha == 6:
        mu, alpha_s, alpha_r, alpha_tc, alpha_gr, alpha_al, alpha_alr, C, betas = params
    elif n_alpha == 4:
        mu, alpha_s, alpha_r, alpha_tc, alpha_gr, C, betas = params
    else:  # n_alpha =2
        mu, alpha_s, alpha_r, C, betas = params
    u, v = uv
    integral = mu * delta
    # assume all timestamps in events_dict are less than t0
    for (x, y) in events_dict:
        if (u, v) == (x, y):  # same node_pair events (alpha_s)
            integral -= alpha_s * events_dict[(x, y)]
        elif (v, u) == (x, y):  # reciprocal node_pair events (alpha_r)
            integral -= alpha_r * events_dict[(x, y)]
        # br node_pairs events (alpha_tc)
        elif n_alpha > 2 and u == x and v != y:
            integral -= alpha_tc * events_dict[(x, y)]
        # gr node_pairs events (alpha_gr)
        elif n_alpha > 2 and u == y and v != x:
            integral -= alpha_gr * events_dict[(x, y)]
        # alliance np (alpha_al)
        elif n_alpha > 4 and v == y and u != x:
            integral -= alpha_al * events_dict[(x, y)]
        # alliance reciprocal np (alpha_alr)
        elif n_alpha > 4 and v == x and u != y:
            integral -= alpha_alr * events_dict[(x, y)]
    return integral


def mulch_uv_intensity_off_integral(n_alpha, delta, uv, params, events_dict, events_dict_r):
    """
    Optimized analytical integral of the intensity function of node pair (u, v) in off-diagonal block pair (u_b, v_b)

    :param n_alpha: number of excitation types (6, 4, or 2)
    :param delta: link prediction test period
    :param uv: node pair ids tuple (u, v)
    :param params: MULCH block pair (u_b, v_b) parameters tuple
    :param events_dict: dictionary of block pair (u_b, v_b) where {(x, y) node pair in bp(u_b, v_b) : x}
        ,and x = sum_q{C_q * [exp(-beta_q * delta)-1] * sum_txy{-beta_q * t0 - txy}}.
    :param events_dict: dictionary of reciprocal block pair (v_b, u_b) where {(x, y) node pair in bp(v_b, v_b) : x}
        ,and x calculated as above. C=[C_q1, .., C_Q] is scaling parameter of bp(u_b, v_b)
    :return: (float) integral result
    """
    # assume timestamps in events_dict, events_dict_r are less than t0
    if n_alpha == 6:
        mu, alpha_s, alpha_r, alpha_tc, alpha_gr, alpha_al, alpha_alr, C, betas = params
    elif n_alpha == 4:
        mu, alpha_s, alpha_r, alpha_tc, alpha_gr, C, betas = params
    else:  # n_alpha =2
        mu, alpha_s, alpha_r, C, betas = params
    u, v = uv
    integral = mu * delta
    # loop through node pairs in block pair ab
    for (x, y) in events_dict:
        # same node_pair events (alpha_s)
        if (u, v) == (x, y):
            integral -= alpha_s * events_dict[(x, y)]
        # br node_pairs events (alpha_tc)
        elif n_alpha > 2 and u == x and v != y:
            integral -= alpha_tc * events_dict[(x, y)]
        # alliance np (alpha_al)
        elif n_alpha > 4 and v == y and u != x:
            integral -= alpha_alr * events_dict[(x, y)]
    # loop through node pairs in reciprocal block pair ba
    for (x, y) in events_dict_r:
        # reciprocal node_pair events (alpha_r)
        if (v, u) == (x, y):
            integral -= alpha_r * events_dict_r[(x, y)]
        # gr node_pairs events (alpha_gr)
        elif n_alpha > 2 and u == y and v != x:
            integral -= alpha_gr * events_dict_r[(x, y)]
        # alliance reciprocal np (alpha_alr)
        elif n_alpha > 4 and v == x and u != y:
            integral -= alpha_alr * events_dict_r[(x, y)]
    return integral


def mulch_uv_intensity_dia(t, uv, params, events_dict, t0):
    """calculate intensity of node pair (u, v) in a diagonal block pair with 6 excitation types.
        Not used: function is just kept for reference"""
    mu, alpha_s, alpha_r, alpha_tc, alpha_gr, alpha_al, alpha_alr, C, betas = params
    u, v = uv
    Q = len(betas)
    intensity = mu
    for (x, y) in events_dict:
        xy_timestamps_less_t0 = events_dict[(x, y)][events_dict[(x, y)] < t0]
        exp_sum_q = 0
        for q in range(Q):
            exp_sum_q += C[q] * betas[q] * np.sum(np.exp(-betas[q] * (t - xy_timestamps_less_t0)))
        if (u, v) == (x, y):  # same node_pair events (alpha_s)
            intensity += alpha_s * exp_sum_q
        elif (v, u) == (x, y):  # reciprocal node_pair events (alpha_r)
            intensity += alpha_r * exp_sum_q
        # br node_pairs events (alpha_tc)
        elif u == x and v != y:
            intensity += alpha_tc * exp_sum_q
        # gr node_pairs events (alpha_gr)
        elif u == y and v != x:
            intensity += alpha_gr * exp_sum_q
        # alliance np (alpha_al)
        elif v == y and u != x:
            intensity += alpha_al * exp_sum_q
        # alliance reciprocal np (alpha_alr)
        elif v == x and u != y:
            intensity += alpha_alr * exp_sum_q
    return intensity


def mulch_uv_intensity_off(t, uv, params, events_dict, events_dict_r, t0):
    """calculate intensity of node pair (u, v) in a off-diagonal block pair with 6 excitation types.
        Not used: function is just kept for reference"""
    mu, alpha_s, alpha_r, alpha_tc, alpha_gr, alpha_al, alpha_alr, C, betas = params
    u, v = uv
    Q = len(betas)
    intensity = mu
    # loop through node pairs in block pair ab
    for (x, y) in events_dict:
        xy_timestamps_less_t0 = events_dict[(x, y)][events_dict[(x, y)] < t0]
        exp_sum_q = 0
        for q in range(Q):
            exp_sum_q += C[q] * betas[q] * np.sum(np.exp(-betas[q] * (t - xy_timestamps_less_t0)))
        # same node_pair events (alpha_s)
        if (u, v) == (x, y):
            intensity += alpha_s * exp_sum_q
        # br node_pairs events (alpha_tc)
        elif u == x and v != y:
            intensity += alpha_tc * exp_sum_q
        # alliance np (alpha_al)
        elif v == y and u != x:
            intensity += alpha_alr * exp_sum_q
    # loop through node pairs in reciprocal block pair ba
    for (x, y) in events_dict_r:
        xy_timestamps_less_t0 = events_dict_r[(x, y)][events_dict_r[(x, y)] < t0]
        exp_sum_q = 0
        for q in range(Q):
            exp_sum_q += C[q] * betas[q] * np.sum(np.exp(-betas[q] * (t0 - xy_timestamps_less_t0)))
        # reciprocal node_pair events (alpha_r)
        if (v, u) == (x, y):
            intensity += alpha_r * exp_sum_q
        # gr node_pairs events (alpha_gr)
        elif u == y and v != x:
            intensity += alpha_gr * exp_sum_q
        # alliance reciprocal np (alpha_alr)
        elif v == x and u != y:
            intensity += alpha_alr * exp_sum_q
    return intensity


def calculate_auc(y, preds, show_figure=False):
    """return AUC score between true and predicted link probabilities for all node pairs in network"""
    fpr, tpr, thresholds = metrics.roc_curve(y.flatten(), preds.flatten(), pos_label=1)
    roc_auc = metrics.roc_auc_score(y.flatten(), preds.flatten())
    if show_figure == True:
        plt.figure(1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        # plt.legend(loc="lower right")
        plt.show()
    return roc_auc
