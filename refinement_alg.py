# TODO: refinement function can be moved to sum betas model fit file

import numpy as np
import time
import copy
from sklearn.metrics import adjusted_rand_score
import utils_fit_sum_betas_model as mulch_fit
import utils_sum_betas_bp as mulch_fit_bp
from utils_generate_sum_betas_model import simulate_sum_kernel_model

def model_fit_cal_log_likelihood_sum_betas(train_tup, all_tup, nodes_not_in_train, n_alpha, K, betas, ref_iter, verbose):
    events_dict_train, n_nodes_train, T_train = train_tup
    events_dict_all, n_nodes_all, T_all= all_tup
    sp_tup, ref_tup, ref_message = model_fit_refine_kernel_sum_exact(events_dict_train, n_nodes_train, T_train, K,
                                                                     betas, n_alpha, max_iter=ref_iter, verbose=verbose)

    # spectral clustering fit results
    nodes_mem_train_sp, fit_param_sp, ll_train_sp, n_events_train, fit_time_sp = sp_tup
    node_mem_all_sp = mulch_fit.assign_node_membership_for_missing_nodes(nodes_mem_train_sp, nodes_not_in_train)
    ll_all_sp, n_events_all = mulch_fit.model_LL_kernel_sum(fit_param_sp, events_dict_all, node_mem_all_sp, K, T_all)
    ll_all_event_sp = ll_all_sp / n_events_all
    ll_train_event_sp = ll_train_sp / n_events_train
    ll_test_event_sp = (ll_all_sp - ll_train_sp) / (n_events_all - n_events_train)


    # refinement fit results
    nodes_mem_train_ref, fit_param_ref, ll_train_ref, num_events, fit_time_ref = ref_tup
    nodes_mem_all_ref = mulch_fit.assign_node_membership_for_missing_nodes(nodes_mem_train_ref, nodes_not_in_train)
    ll_all_ref, n_events_all = mulch_fit.model_LL_kernel_sum(fit_param_ref, events_dict_all, nodes_mem_all_ref, K, T_all)
    ll_all_event_ref = ll_all_ref / n_events_all
    ll_train_event_ref = ll_train_ref / n_events_train
    ll_test_event_ref = (ll_all_ref - ll_train_ref) / (n_events_all - n_events_train)


    results_dict = {}
    results_dict["fit_param_ref"] = fit_param_ref
    results_dict["node_mem_train_ref"] = nodes_mem_train_ref
    results_dict["node_mem_all_ref"] = nodes_mem_all_ref
    results_dict["ll_train_ref"] = ll_train_event_ref
    results_dict["ll_all_ref"] = ll_all_event_ref
    results_dict["ll_test_ref"] = ll_test_event_ref
    results_dict["fit_param_sp"] = fit_param_sp
    results_dict["node_mem_train_sp"] = nodes_mem_train_sp
    results_dict["node_mem_all_sp"] = node_mem_all_sp
    results_dict["ll_train_sp"] = ll_train_event_sp
    results_dict["ll_all_sp"] = ll_all_event_sp
    results_dict["ll_test_sp"] = ll_test_event_sp
    results_dict["message"] = ref_message
    results_dict["n_classes"] = K
    results_dict["fit_time_sp(s)"] = fit_time_sp
    results_dict["fit_time_ref(s)"] = fit_time_ref
    results_dict["train_end_time"] = T_train
    results_dict["all_end_time"] = T_all
    return results_dict


#%% Exact

def cal_new_event_dict_bp(idx, from_block, to_block, events_dict_bp, K):
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

def cal_new_LL_kernel_sum_move_node(param_tup, T, idx, from_block, to_block, events_dict_bp, N_c, LL_bp, batch = True):
    K = len(N_c)
    if len(param_tup) == 9:
        mu_bp, alpha_n_bp, alpha_r_bp, alpha_br_bp, alpha_gr_bp, alpha_al_bp, alpha_alr_bp, C_bp, betas = param_tup
    elif len(param_tup) == 7:
        mu_bp, alpha_n_bp, alpha_r_bp, alpha_br_bp, alpha_gr_bp, C_bp, betas = param_tup
    else:
        mu_bp, alpha_n_bp, alpha_r_bp, C_bp, betas = param_tup

    # calculate events_dict_bp after moving node_i
    events_dict_bp1 = cal_new_event_dict_bp(idx, from_block, to_block, events_dict_bp, K)

    # calculate new n_nodes_per_class
    N_c1 = np.copy(N_c)
    N_c1[from_block, 0] -=1
    N_c1[to_block, 0] += 1
    # calculate new M_bp
    M_bp1 = np.matmul(N_c1, N_c1.T)- np.diagflat(N_c1)

    # calculate new log-likelihood for affected block pairs
    LL_bp1 = np.copy(LL_bp)
    for a in range(K):
        for b in range(K):
            if a==from_block or a==to_block or b==from_block or b==to_block:
                if len(param_tup) == 9:
                    par = (mu_bp[a, b], alpha_n_bp[a, b], alpha_r_bp[a, b], alpha_br_bp[a, b], alpha_gr_bp[a, b], alpha_al_bp[a, b],
                           alpha_alr_bp[a, b], C_bp[a,b], betas)
                    if a == b:
                        LL_bp1[a, b] = mulch_fit_bp.LL_6_alpha_kernel_sum_dia(par, events_dict_bp1[a][b], T, N_c1[b, 0], M_bp1[a, b])
                    else:
                        LL_bp1[a, b] = mulch_fit_bp.LL_6_alpha_kernel_sum_off(par, (events_dict_bp1[a][b]), (events_dict_bp1[b][a]), T,
                                                                              N_c1[b, 0], M_bp1[a, b])
                elif len(param_tup) == 7:
                    par = (mu_bp[a, b], alpha_n_bp[a, b], alpha_r_bp[a, b], alpha_br_bp[a, b], alpha_gr_bp[a, b], C_bp[a, b], betas)
                    if a == b:
                        LL_bp1[a, b] = mulch_fit_bp.LL_4_alpha_kernel_sum_dia(par, events_dict_bp1[a][b], T, N_c1[b, 0], M_bp1[a, b])
                    else:
                        LL_bp1[a, b] = mulch_fit_bp.LL_4_alpha_kernel_sum_off(par, (events_dict_bp1[a][b]), (events_dict_bp1[b][a]), T,
                                                                              N_c1[b, 0], M_bp1[a, b])
                elif len(param_tup) == 5:
                    par = (mu_bp[a, b], alpha_n_bp[a, b], alpha_r_bp[a, b], C_bp[a, b], betas)
                    if a == b:
                        LL_bp1[a, b] = mulch_fit_bp.LL_2_alpha_kernel_sum_dia(par, events_dict_bp1[a][b], T, N_c1[b, 0], M_bp1[a, b])
                    else:
                        LL_bp1[a, b] = mulch_fit_bp.LL_2_alpha_kernel_sum_off(par, (events_dict_bp1[a][b]), (events_dict_bp1[b][a]), T,
                                                                              N_c1[b, 0], M_bp1[a, b])
    if batch:
        return(np.sum(LL_bp1))
    else:
        return events_dict_bp1, N_c1, LL_bp1

def nodes_mem_refinement_batch(nodes_mem, events_dict_bp, param, T, N_c, LL_bp):
    K = len(N_c)
    current_ll = np.sum(LL_bp)
    nodes_mem_ref = np.copy(nodes_mem)
    for node_i, from_block in enumerate(nodes_mem):
        # only try to move node if it's not in a block by itself
        if N_c[from_block] > 1:
            # holds log-likelihood scores when assigning node_i to different blocks
            node_i_LL_score = np.zeros(K)
            node_i_LL_score[from_block] = current_ll
            # loop through all block and change membership of node_i
            for to_block in range(K):
                if to_block != from_block:
                    node_i_LL_score[to_block] = cal_new_LL_kernel_sum_move_node(param, T, node_i, from_block, to_block,
                                                                                events_dict_bp, N_c, LL_bp)
            nodes_mem_ref[node_i] = np.argmax(node_i_LL_score)
    return nodes_mem_ref

# Model fit and refine functions
def model_fit_refine_kernel_sum_exact(events_dict, N, end_time, K, betas, n_alpha=6, max_iter=0,verbose=False,
                                      nodes_mem_true=None):
    # nodes_mem_true: only for simulation data

    # 1) run spectral clustering
    if verbose:
        print("\nRun spectral clustering and fit mulch")
    start_fit_time = time.time()
    agg_adj = mulch_fit.event_dict_to_aggregated_adjacency(N, events_dict)
    nodes_mem0 = mulch_fit.spectral_cluster1(agg_adj, K, n_kmeans_init=500, normalize_z=True, multiply_s=True)
    if nodes_mem_true is not None and verbose:
        print(f"\tadusted RI between true and sp = {adjusted_rand_score(nodes_mem_true, nodes_mem0):.3f}")

    # if verbose:
    #     MBHP.plot_adj(agg_adj, nodes_mem0, K, f"Spectral membership K={K}")
    #     classes, n_node_per_class = np.unique(nodes_mem0, return_counts=True)
    #     print("\tnodes/class# : ", np.sort(n_node_per_class/np.sum(n_node_per_class)))

    # 2) fit model and get parameters, LL
    fit_param0, LL_bp0, num_events, events_dict_bp0, N_c0 = mulch_fit.model_fit_kernel_sum(n_alpha, events_dict, nodes_mem0,
                                                                                           K, end_time, betas, ref=True)
    spectral_fit_time = time.time() - start_fit_time
    if verbose:
        print("\tlog-likelihood = ", np.sum(LL_bp0))
    # MBHP.print_model_param_kernel_sum(fit_param0)

    sp_tuple = (nodes_mem0, fit_param0, np.sum(LL_bp0), num_events, spectral_fit_time)
    # no refinement needed if K=1
    if K == 1:
        return sp_tuple, sp_tuple, "No refinement needed at K=1"
    if max_iter == 0:
        return sp_tuple, sp_tuple, "No refinement (MAX refinement iterations = 0)"

    # 3) refinement - stop if converged or #blocks decreased
    message = f"Max #iterations ({max_iter}) reached"
    for ref_iter in range(max_iter):
        if verbose:
            print("Refinement iteration #", ref_iter + 1)
        nodes_mem1 = nodes_mem_refinement_batch(nodes_mem0, events_dict_bp0, fit_param0, end_time, N_c0, LL_bp0)

        # break if no change in node_membership after refinement iteration
        if adjusted_rand_score(nodes_mem0, nodes_mem1) == 1:
            message = f"Break: node membership converged at iter# {ref_iter+1}"
            if verbose:
                print(f"\t--> {message}")
            break

        # break if number of classes decreased - nodes moving into the biggest block
        classes_ref, n_node_per_class_ref = np.unique(nodes_mem1, return_counts=True)
        if len(classes_ref) < K:
            # print("nodes/class percentage : ", np.sort(n_node_per_class_ref / np.sum(n_node_per_class_ref)))
            message = f"Break: number of classes decreased at iter# {ref_iter+1}"
            if verbose:
                print(f"\t--> {message}")
            break

        # MBHP.plot_adj(agg_adj, nodes_mem1, K, f"Refinement membership K={K}, iteration={ref_iter}")
        if nodes_mem_true is not None and verbose:
            print(f"\tadjusted RI={adjusted_rand_score(nodes_mem_true, nodes_mem1):.3f}")

        # fit model on refined node membership
        fit_param1, LL_bp1, num_events, events_dict_bp1, N_c1 = mulch_fit.model_fit_kernel_sum(n_alpha, events_dict, nodes_mem1,
                                                                                               K, end_time, betas, ref=True)
        # break if train log-likelihood decreased
        if np.sum(LL_bp1) < np.sum(LL_bp0):
            message = f"Break: train ll decreased from {np.sum(LL_bp0):.1f} to {np.sum(LL_bp1):.1f} at iter# {ref_iter + 1}"
            if verbose:
                print(f"\t--> {message}")
            break
        if verbose:
            print("\tlog-likelihood = ", np.sum(LL_bp1))

        # MBHP.print_model_param_kernel_sum(fit_param1)
        # set new argument for next loop
        nodes_mem0 = nodes_mem1
        events_dict_bp0 = events_dict_bp1
        fit_param0 = fit_param1
        N_c0 = N_c1
        LL_bp0 = LL_bp1
    refinement_fit_time = time.time() - start_fit_time
    ref_tuple = (nodes_mem0, fit_param0, np.sum(LL_bp0), num_events, refinement_fit_time)
    return sp_tuple, ref_tuple, message


#%% main

if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    K, N, T_all = 3, 60, 1000  # number of nodes and duration
    n_alpha = 6
    # p = [1 / K] * K  # balanced node membership
    p = [0.70, 0.15, 0.15]
    mu_sim = np.array([[0.0001, 0.0001, 0.0001], [0.0003, 0.0003, 0.0003], [0.0003, 0.0001, 0.0003]])
    alpha_n_sim = np.array([[0.03, 0.03, 0.02], [0.0, 0.1, 0.01], [0.0, 0.03, 0.1]])
    alpha_r_sim = np.array([[0.01, 0.05, 0.07], [0.01, 0.01, 0.01], [0.001, 0.0, 0.35]])
    alpha_br_sim = np.array([[0.009, 0.001, 0.0001], [0.0, 0.07, 0.0006], [0.0001, 0.01, 0.05]])
    alpha_gr_sim = np.array([[0.001, 0.0, 0.0001], [0.0, 0.008, 0.0001], [0.0, 0.0002, 0.0]])
    alpha_al_sim = np.array([[0.001, 0.0001, 0.0], [0.0, 0.02, 0.0], [0.0001, 0.005, 0.01]])
    alpha_alr_sim = np.array([[0.001, 0.0001, 0.0001], [0.0, 0.001, 0.0006], [0.0001, 0.0, 0.003]])
    C_sim = np.array([[[0.33, 0.34, 0.33], [0.33, 0.34, 0.33], [0.33, 0.34, 0.33]]] * K)
    betas = np.array([0.01, 0.1, 20])
    # 1) simulate from 6-alpha sum of kernels model
    sim_param = (
    mu_sim, alpha_n_sim, alpha_r_sim, alpha_br_sim, alpha_gr_sim, alpha_al_sim, alpha_alr_sim, C_sim, betas)
    betas = sim_param[-1]
    print(f"{n_alpha}-alpha Sum of Kernels model simultion at K={K}, N={N}, not balanced membership")
    print("betas = ", betas)
    events_dict, nodes_mem_true = simulate_sum_kernel_model(sim_param, N, K, p, T_all)
    n_events_all = mulch_fit_bp.cal_num_events(events_dict)
    print("n_events simulated = ", n_events_all)
    agg_adj = mulch_fit.event_dict_to_aggregated_adjacency(N, events_dict)
    mulch_fit.plot_adj(agg_adj, nodes_mem_true, K, "True membership")

    MAX_ITER = 15
    model_fit_refine_kernel_sum_exact(events_dict, N, T_all, K, betas, n_alpha, MAX_ITER, nodes_mem_true=nodes_mem_true,
                                      verbose=True)




