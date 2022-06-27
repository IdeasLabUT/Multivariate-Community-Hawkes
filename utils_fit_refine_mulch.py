"""fit MULCH + node membership refinement function"""

import numpy as np
import time
import copy
from sklearn.metrics import adjusted_rand_score
import utils_fit_model as fit_model
import utils_fit_bp as fit_bp
from utils_generate_model import simulate_mulch


def fit_refinement_mulch(events_dict, n, end_time, K, betas, n_alpha=6, max_ref_iter=0,
                         verbose=False,
                         nodes_mem_true=None):
    """
    fit MULCH on the network and refine nodes membership

    1) run spectral clustering, get nodes membership, and fit MULCH.
    2) run refinement algorithm and re-fit MULCH.
    3) repeat step(2) until one of the followings:
        - nodes membership converge
        - model log-likelihood decreases
        - number of classes decreases
        - Maximum number refinement iteration reached

    :param dict events_dict: dataset formatted as a dictionary
        {(u, v) node pairs in network : [t1, t2, ...] array of events between (u, v)}
    :param int n: number of nodes in network
    :param float end_time: duration of network
    :param int K: number of blocks
    :param betas: (Q,) array of decays
    :param int n_alpha: (Optional) number of excitation types. Choose between 2, 4, or 6.
        Default is 6 types (full model)
    :param int max_ref_iter: (Optional) maximum number of refinement iterations.
        Default is 0 (no refinement)
    :param verbose: (Optional) print fitting details
    :param nodes_mem_true: (Optional) only when true nodes membership known (for simulation tests)
    :return: two tuples of MULCH fitting results using both spectral clustering and refined nodes
        membership.
        Each result tuple = (nodes_mem, fit_parameters, log-likelihood, #events, time_to_fit_sec).
        Also return refinement status message.
    :rtype: (tuple, tuple, str)
    """

    # 1) run spectral clustering
    if verbose:
        print("\nRun spectral clustering and fit MULCH")
    start_fit_time = time.time()
    agg_adj = fit_model.event_dict_to_aggregated_adjacency(n, events_dict)
    nodes_mem0 = fit_model.spectral_cluster1(agg_adj, K, n_kmeans_init=500, normalize_z=True,
                                             multiply_s=True)
    if nodes_mem_true is not None and verbose:
        print(f"\tadjusted RI between true and spectral membership = "
              f"{adjusted_rand_score(nodes_mem_true, nodes_mem0):.3f}")

    # if verbose:
    #     MBHP.plot_adj(agg_adj, nodes_mem0, K, f"Spectral membership K={K}")
    #     classes, n_node_per_class = np.unique(nodes_mem0, return_counts=True)
    #     print("\tnodes/class# : ", np.sort(n_node_per_class/np.sum(n_node_per_class)))

    # 2) fit model and get parameters, log-likelihood
    fit_param0, LL_bp0, num_events, events_dict_bp0, N_c0 = fit_model.model_fit(n_alpha,
                                                                                events_dict,
                                                                                nodes_mem0, K,
                                                                                end_time,
                                                                                betas, ref=True)
    spectral_fit_time = time.time() - start_fit_time
    if verbose:
        print("\tlog-likelihood = ", np.sum(LL_bp0))
    # MBHP.print_mulch_param(fit_param0)
    sp_tuple = (nodes_mem0, fit_param0, np.sum(LL_bp0), num_events, spectral_fit_time)

    # no refinement needed if K=1
    if K == 1:
        return sp_tuple, sp_tuple, "No refinement needed at K=1"
    if max_ref_iter == 0:
        return sp_tuple, sp_tuple, "No refinement (MAX refinement iterations = 0)"

    # 3) refinement - stop if node-membership converged, number of blocks decreases,
    # or log-likelihood decreases
    message = f"Max #iterations ({max_ref_iter}) reached"
    for ref_iter in range(max_ref_iter):
        if verbose:
            print("Refinement iteration #", ref_iter + 1)
        nodes_mem1 = get_nodes_mem_refinement(nodes_mem0, events_dict_bp0, fit_param0, end_time,
                                              N_c0, LL_bp0)

        # break if no change in node_membership after a refinement iteration
        if adjusted_rand_score(nodes_mem0, nodes_mem1) == 1:
            message = f"Break: node membership converged at iter# {ref_iter + 1}"
            if verbose:
                print(f"\t--> {message}")
            break

        # break if number of classes decreased (#unique_classes < K)
        classes_ref, n_node_per_class_ref = np.unique(nodes_mem1, return_counts=True)
        if len(classes_ref) < K:
            # print("nodes/class percentage : ", np.sort(n_node_per_class_ref / np.sum(n_node_per_class_ref)))
            message = f"Break: number of classes decreased at iter# {ref_iter + 1}"
            if verbose:
                print(f"\t--> {message}")
            break

        if nodes_mem_true is not None and verbose:
            print(f"\tadjusted RI={adjusted_rand_score(nodes_mem_true, nodes_mem1):.3f}")

        # fit model on refined node membership
        fit_param1, LL_bp1, num_events, events_dict_bp1, N_c1 = fit_model.model_fit(n_alpha,
                                                                                    events_dict,
                                                                                    nodes_mem1,
                                                                                    K, end_time,
                                                                                    betas, ref=True)
        # break if train log-likelihood decreased
        if np.sum(LL_bp1) < np.sum(LL_bp0):
            message = f"Break: train ll decreased from {np.sum(LL_bp0):.1f} to {np.sum(LL_bp1):.1f}" \
                      f" at iter# {ref_iter + 1}"
            if verbose:
                print(f"\t--> {message}")
            break
        if verbose:
            print("\tlog-likelihood = ", np.sum(LL_bp1))

        # MBHP.print_mulch_param(fit_param1)
        # set new argument for next loop
        nodes_mem0 = nodes_mem1
        events_dict_bp0 = events_dict_bp1
        fit_param0 = fit_param1
        N_c0 = N_c1
        LL_bp0 = LL_bp1
    refinement_fit_time = time.time() - start_fit_time
    ref_tuple = (nodes_mem0, fit_param0, np.sum(LL_bp0), num_events, refinement_fit_time)
    return sp_tuple, ref_tuple, message


def cal_new_event_dict_move_node(idx, from_block, to_block, events_dict_bp, K):
    """ calculate new per block_pair events_dict

    (K, K) events dictionary per block pair after moving one node from one block to another """
    events_dict_bp1 = copy.deepcopy(events_dict_bp)
    # move node pair from (a,*) to (b,*)
    for k in range(K):
        for (i, j) in events_dict_bp[from_block][k]:
            if i == idx:
                timestamps = events_dict_bp1[from_block][k].pop((i, j))
                events_dict_bp1[to_block][k][i, j] = timestamps
    # move nodepair from (*,a) to (*,b)
    for k in range(K):
        for (i, j) in events_dict_bp[k][from_block]:
            if j == idx:
                timestamps = events_dict_bp1[k][from_block].pop((i, j))
                events_dict_bp1[k][to_block][i, j] = timestamps
    return events_dict_bp1


def cal_new_LL_move_node(param_tup, T, idx, from_block, to_block, events_dict_bp, n_K, LL_bp,
                         batch=True):
    """ calculate new per block_pair log-likelihoods (K,K)

    new (K, K) log-likelihood for each block pair after moving one node (idx) from one block
    to another """
    K = len(n_K)
    if len(param_tup) == 9:
        mu_bp, alpha_s_bp, alpha_r_bp, alpha_tc_bp, alpha_gr_bp, alpha_al_bp, alpha_alr_bp, C_bp, betas = param_tup
    elif len(param_tup) == 7:
        mu_bp, alpha_s_bp, alpha_r_bp, alpha_tc_bp, alpha_gr_bp, C_bp, betas = param_tup
    else:
        mu_bp, alpha_s_bp, alpha_r_bp, C_bp, betas = param_tup

    # calculate events_dict_bp after moving node_i
    events_dict_bp1 = cal_new_event_dict_move_node(idx, from_block, to_block, events_dict_bp, K)

    # calculate new n_nodes_per_class
    N_c1 = np.copy(n_K)
    N_c1[from_block, 0] -= 1
    N_c1[to_block, 0] += 1
    # calculate new M_bp
    M_bp1 = np.matmul(N_c1, N_c1.T) - np.diagflat(N_c1)

    # calculate new log-likelihood for affected block pairs
    LL_bp1 = np.copy(LL_bp)
    for a in range(K):
        for b in range(K):
            if a == from_block or a == to_block or b == from_block or b == to_block:
                if len(param_tup) == 9:
                    par = (mu_bp[a, b], alpha_s_bp[a, b], alpha_r_bp[a, b], alpha_tc_bp[a, b],
                           alpha_gr_bp[a, b], alpha_al_bp[a, b],
                           alpha_alr_bp[a, b], C_bp[a, b], betas)
                    if a == b:
                        LL_bp1[a, b] = fit_bp.LL_6_alpha_dia_bp(par, events_dict_bp1[a][b], T,
                                                                N_c1[b, 0], M_bp1[a, b])
                    else:
                        LL_bp1[a, b] = fit_bp.LL_6_alpha_off_bp(par, (events_dict_bp1[a][b]),
                                                                (events_dict_bp1[b][a]), T,
                                                                N_c1[b, 0], M_bp1[a, b])
                elif len(param_tup) == 7:
                    par = (mu_bp[a, b], alpha_s_bp[a, b], alpha_r_bp[a, b], alpha_tc_bp[a, b],
                           alpha_gr_bp[a, b], C_bp[a, b], betas)
                    if a == b:
                        LL_bp1[a, b] = fit_bp.LL_4_alpha_dia_bp(par, events_dict_bp1[a][b], T,
                                                                N_c1[b, 0], M_bp1[a, b])
                    else:
                        LL_bp1[a, b] = fit_bp.LL_4_alpha_off_bp(par, (events_dict_bp1[a][b]),
                                                                (events_dict_bp1[b][a]), T,
                                                                N_c1[b, 0], M_bp1[a, b])
                elif len(param_tup) == 5:
                    par = (mu_bp[a, b], alpha_s_bp[a, b], alpha_r_bp[a, b], C_bp[a, b], betas)
                    if a == b:
                        LL_bp1[a, b] = fit_bp.LL_2_alpha_dia_bp(par, events_dict_bp1[a][b], T,
                                                                N_c1[b, 0], M_bp1[a, b])
                    else:
                        LL_bp1[a, b] = fit_bp.LL_2_alpha_off_bp(par, (events_dict_bp1[a][b]),
                                                                (events_dict_bp1[b][a]), T,
                                                                N_c1[b, 0], M_bp1[a, b])
    if batch:
        return (np.sum(LL_bp1))
    else:
        return events_dict_bp1, N_c1, LL_bp1


def get_nodes_mem_refinement(nodes_mem, events_dict_bp, param, end_time, n_K, LL_bp):
    """
    calculate new refined nodes membership given current node membership and MULCH fit parameters

    :param nodes_mem: (n,) array(int) of nodes membership
    :param events_dict_bp: KxK list, each element events_dict[a][b] is events_dict of the block pair (a, b)
    :param tuple param: mulch parameters (m
    :param float end_time: network duration
    :param n_K: (K,) array(int) number of nodes per block
    :param LL_bp: (K, K) array, where LL_bp[a, b] is log-likelihood of block pair (a, b)
    :return: (n,) array(int) refined nodes membership
    """
    K = len(n_K)
    current_ll = np.sum(LL_bp)
    nodes_mem_ref = np.copy(nodes_mem)
    for node_i, from_block in enumerate(nodes_mem):
        # only try to move node if it's not in a block by itself
        if n_K[from_block] > 1:
            # holds log-likelihood scores when assigning node_i to different blocks
            node_i_LL_score = np.zeros(K)
            node_i_LL_score[from_block] = current_ll
            # loop through all block and change membership of node_i
            for to_block in range(K):
                if to_block != from_block:
                    node_i_LL_score[to_block] = cal_new_LL_move_node(param, end_time, node_i,
                                                                     from_block, to_block,
                                                                     events_dict_bp, n_K, LL_bp)
            nodes_mem_ref[node_i] = np.argmax(node_i_LL_score)
    return nodes_mem_ref


# %% Simulate, fit, and refine example

if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    K, N, T_all = 3, 60, 1000  # number of nodes and duration
    n_alpha = 6
    # p = [1 / K] * K  # balanced node membership
    p = [0.70, 0.15, 0.15]
    mu_sim = np.array(
        [[0.0001, 0.0001, 0.0001], [0.0003, 0.0003, 0.0003], [0.0003, 0.0001, 0.0003]])
    alpha_s_sim = np.array([[0.03, 0.03, 0.02], [0.0, 0.1, 0.01], [0.0, 0.03, 0.1]])
    alpha_r_sim = np.array([[0.01, 0.05, 0.07], [0.01, 0.01, 0.01], [0.001, 0.0, 0.35]])
    alpha_tc_sim = np.array([[0.009, 0.001, 0.0001], [0.0, 0.07, 0.0006], [0.0001, 0.01, 0.05]])
    alpha_gr_sim = np.array([[0.001, 0.0, 0.0001], [0.0, 0.008, 0.0001], [0.0, 0.0002, 0.0]])
    alpha_al_sim = np.array([[0.001, 0.0001, 0.0], [0.0, 0.02, 0.0], [0.0001, 0.005, 0.01]])
    alpha_alr_sim = np.array([[0.001, 0.0001, 0.0001], [0.0, 0.001, 0.0006], [0.0001, 0.0, 0.003]])
    C_sim = np.array([[[0.33, 0.34, 0.33], [0.33, 0.34, 0.33], [0.33, 0.34, 0.33]]] * K)
    betas = np.array([0.01, 0.1, 20])
    # 1) simulate from 6-alpha sum of kernels model
    sim_param = (mu_sim, alpha_s_sim, alpha_r_sim, alpha_tc_sim, alpha_gr_sim, alpha_al_sim,
                 alpha_alr_sim, C_sim, betas)
    betas = sim_param[-1]
    print(f"{n_alpha}-alpha Sum of Kernels model simulation at K={K}, N={N}, unbalanced membership")
    print("betas = ", betas)
    events_dict, nodes_mem_true = simulate_mulch(sim_param, N, K, p, T_all)
    n_events_all = fit_bp.cal_num_events(events_dict)
    print("number of simulated events= ", n_events_all)
    agg_adj = fit_model.event_dict_to_aggregated_adjacency(N, events_dict)
    fit_model.plot_adj(agg_adj, nodes_mem_true, K, "True membership")

    MAX_ITER = 10
    fit_refinement_mulch(events_dict, N, T_all, K, betas, n_alpha, MAX_ITER,
                         nodes_mem_true=nodes_mem_true,
                         verbose=True)
