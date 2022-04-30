# TODO: create refinement functions for one beta model

import numpy as np
import time
import pickle
import utils_one_beta_bp as one_beta_bp
import utils_fit_sum_betas_model as mulch_fit_util


def model_fit_cal_log_likelihood_one_beta(train_tup, all_tup, nodes_not_in_train, n_alpha, n_classes, beta, save_file=""):
    events_dict_train, n_nodes_train, T_train = train_tup
    events_dict_all, n_nodes_all, T_all= all_tup
    start_fit_time = time.time()
    agg_adj = mulch_fit_util.event_dict_to_aggregated_adjacency(n_nodes_train, events_dict_train)
    # run spectral clustering on train dataset
    node_mem_train = mulch_fit_util.spectral_cluster1(agg_adj, n_classes, n_kmeans_init=500)
    node_mem_all = mulch_fit_util.assign_node_membership_for_missing_nodes(node_mem_train, nodes_not_in_train)
    # fit model
    params_est, ll_t, n_events_t = model_fit_single_beta(n_alpha, events_dict_train, node_mem_train, n_classes, T_train,
                                                         beta)
    end_fit_time = time.time()
    # calculate log-likelihood
    ll_all, n_events_all = model_LL_one_beta(params_est, events_dict_all, node_mem_all, n_classes, T_all)
    ll_all_event = ll_all / n_events_all
    ll_train_event = ll_t / n_events_t
    ll_test_event = (ll_all - ll_t) / (n_events_all - n_events_t)
    time_to_fit = end_fit_time - start_fit_time

    results_dict = {}
    results_dict["mu"] = params_est[0]
    results_dict["alpha_n"] = params_est[1]
    results_dict["alpha_r"] = params_est[2]
    if n_alpha > 2:
        results_dict["alpha_br"] = params_est[3]
        results_dict["alpha_gr"] = params_est[4]
        if n_alpha > 4:
            results_dict["alpha_al"] = params_est[5]
            results_dict["alpha_alr"] = params_est[6]
    results_dict["beta"] = beta
    results_dict["n_classes"] = n_classes
    results_dict["node_mem_train"] = node_mem_train
    results_dict["node_mem_all"] = node_mem_all
    results_dict["ll_train"] = ll_train_event
    results_dict["ll_all"] = ll_all_event
    results_dict["ll_test"] = ll_test_event
    results_dict["fit_time(s)"] = time_to_fit
    results_dict["train_end_time"] = T_train
    results_dict["all_end_time"] = T_all

    # save results
    if save_file != "":
        with open(f"{save_file}.p", 'wb') as f:
            pickle.dump(results_dict, f)
    return results_dict

#%% mulch one beta log-likelihood functions

def model_LL_one_beta(param, events_dict, node_mem, k, end_time):
    block_pair_M, n_nodes_c = mulch_fit_util.num_nodes_pairs_per_block_pair(node_mem, k)
    events_dict_block_pair = mulch_fit_util.events_dict_to_events_dict_bp(events_dict, node_mem, k)
    if len(param)==4:
        ll, n_event = model_LL_2_alpha_single_beta(param, events_dict_block_pair, end_time, block_pair_M, n_nodes_c, k)
    elif len(param)==6:
        ll, n_event = model_LL_4_alpha_single_beta(param, events_dict_block_pair, end_time, block_pair_M, n_nodes_c, k)
    elif len(param)==8:
        ll, n_event = model_LL_6_alpha_single_beta(param, events_dict_block_pair, end_time, block_pair_M, n_nodes_c, k)
    return ll, n_event


def model_LL_6_alpha_single_beta(params_tup, events_dict_bp, end_time, M_bp, n_nodes_c, n_classes):
    mu_bp, alpha_n_bp, alpha_r_bp, alpha_br_bp, alpha_gr_bp,alpha_al_bp, alpha_alr_bp, beta = params_tup
    ll_full = 0
    num_events = 0
    for i in range(n_classes):
        for j in range(n_classes):
            par = (mu_bp[i, j], alpha_n_bp[i, j], alpha_r_bp[i, j], alpha_br_bp[i, j], alpha_gr_bp[i, j],
                  alpha_al_bp[i, j], alpha_alr_bp[i, j], beta)
            if i == j: # diagonal block pair
                ll_dia = one_beta_bp.LL_6_alpha_one_beta_dia(par, events_dict_bp[i][j], end_time, n_nodes_c[j,0], M_bp[i,j])
                ll_full += ll_dia
            else: # off-diagonal block pair
                ll_off = one_beta_bp.LL_6_alpha_one_beta_off(par, (events_dict_bp[i][j]), (events_dict_bp[j][i]),
                                                                      end_time, n_nodes_c[j,0],  M_bp[i,j])
                ll_full += ll_off
            # number of event of block_pair
            num_events += one_beta_bp.cal_num_events(events_dict_bp[i][j])
    return ll_full, num_events


def model_LL_4_alpha_single_beta(params_tup, events_dict_bp, end_time, M_bp, n_nodes_c, n_classes):
    mu_bp, alpha_n_bp, alpha_r_bp, alpha_br_bp, alpha_gr_bp, beta = params_tup
    ll_full = 0
    num_events = 0
    for i in range(n_classes):
        for j in range(n_classes):
            par = (mu_bp[i, j], alpha_n_bp[i, j], alpha_r_bp[i, j], alpha_br_bp[i, j], alpha_gr_bp[i, j], beta)
            if i == j: # diagonal block pair
                ll_dia = one_beta_bp.LL_4_alpha_one_beta_dia(par, events_dict_bp[i][j], end_time, n_nodes_c[j,0], M_bp[i,j])
                ll_full += ll_dia
            else: # off-diagonal block pair
                ll_off = one_beta_bp.LL_4_alpha_one_beta_off(par, (events_dict_bp[i][j]), (events_dict_bp[j][i]),
                                                                      end_time, n_nodes_c[j,0],  M_bp[i,j])
                ll_full += ll_off
            # number of event of block_pair
            num_events += one_beta_bp.cal_num_events(events_dict_bp[i][j])
    return ll_full, num_events


def model_LL_2_alpha_single_beta(params_tup, events_dict_bp, end_time, M_bp, n_nodes_c, n_classes):
    mu_bp, alpha_n_bp, alpha_r_bp, beta = params_tup
    ll_full = 0
    num_events = 0
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j: # diagonal block pair
                par = (mu_bp[i,j], alpha_n_bp[i,j], alpha_r_bp[i,j], beta)
                ll_full += one_beta_bp.LL_2_alpha_one_beta_dia(par, events_dict_bp[i][j], end_time, n_nodes_c[j,0], M_bp[i,j])
            else: # off-diagonal block pair
                par = (mu_bp[i,j], alpha_n_bp[i,j], alpha_r_bp[i,j], beta)
                ll_full += one_beta_bp.LL_2_alpha_one_beta_off(par, (events_dict_bp[i][j]), (events_dict_bp[j][i]),
                                                                      end_time, n_nodes_c[j,0],  M_bp[i,j])
            # number of event of block_pair
            num_events += one_beta_bp.cal_num_events(events_dict_bp[i][j])
    return ll_full, num_events




#%% mulch one beta fit functions

def model_fit_single_beta(n_alpha, events_dict, node_mem, n_classes, end_time, beta):
    if n_alpha == 2:
        return fit_2_alpha_single_beta(events_dict, node_mem, n_classes, end_time, beta)
    elif n_alpha == 4:
        return fit_4_alpha_single_beta(events_dict, node_mem, n_classes, end_time, beta)
    elif n_alpha == 6:
        return fit_6_alpha_single_beta(events_dict, node_mem, n_classes, end_time, beta)
    else:
        print(" number of alpha parameter should be 2, 4, or 6")


def fit_2_alpha_single_beta(events_dict, node_mem, n_classes, end_time, beta):
    # return number of node pairs within a block pair, number of nodes per class
    bp_M_train, n_nodes_c = mulch_fit_util.num_nodes_pairs_per_block_pair(node_mem, n_classes)
    bps_train = mulch_fit_util.events_dict_to_events_dict_bp(events_dict, node_mem, n_classes)
    # initialize paramters matrices
    mu_bp = np.zeros((n_classes, n_classes))
    alpha_n_bp = np.zeros((n_classes, n_classes))
    alpha_r_bp = np.zeros((n_classes,n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:  # diagonal block pair
                params_est = one_beta_bp.fit_2_alpha_one_beta_dia(bps_train[i][j], end_time, n_nodes_c[j,0], bp_M_train[i, j], beta)

            else:   # off-diagonal block pair
                params_est = one_beta_bp.fit_2_alpha_one_beta_off(bps_train[i][j], bps_train[j][i], end_time,
                                                           n_nodes_c[j,0], bp_M_train[i, j],beta)
            mu_bp[i, j] = params_est[0]
            alpha_n_bp[i, j] = params_est[1]
            alpha_r_bp[i, j] = params_est[2]
    # calclate log-likelihood on train, test, all datasets
    params_tuple = (mu_bp, alpha_n_bp, alpha_r_bp, beta)
    ll_train_full, num_events_train = model_LL_2_alpha_single_beta(params_tuple, bps_train, end_time, bp_M_train, n_nodes_c, n_classes)
    return params_tuple, ll_train_full, num_events_train


def fit_4_alpha_single_beta(events_dict, node_mem, n_classes, end_time, beta):
    # return number of node pairs within a block pair, number of nodes per class
    block_pair_M_train, n_nodes_c = mulch_fit_util.num_nodes_pairs_per_block_pair(node_mem, n_classes)
    block_pairs_train = mulch_fit_util.events_dict_to_events_dict_bp(events_dict, node_mem, n_classes)
    # initialize paramters matrices
    mu_bp = np.zeros((n_classes, n_classes))
    alpha_n_bp = np.zeros((n_classes, n_classes))
    alpha_r_bp = np.zeros((n_classes,n_classes))
    alpha_br_bp = np.zeros((n_classes, n_classes))
    alpha_gr_bp = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:  # diagonal block pair
                params_est = one_beta_bp.fit_4_alpha_one_beta_dia(block_pairs_train[i][j], end_time,n_nodes_c[j,0],
                                                                 block_pair_M_train[i, j], beta)

            else:   # off-diagonal block pair
                params_est = one_beta_bp.fit_4_alpha_one_beta_off(block_pairs_train[i][j],block_pairs_train[j][i],
                                                       end_time, n_nodes_c[j,0], block_pair_M_train[i, j],beta)
            mu_bp[i, j] = params_est[0]
            alpha_n_bp[i, j] = params_est[1]
            alpha_r_bp[i, j] = params_est[2]
            alpha_br_bp[i, j] = params_est[3]
            alpha_gr_bp[i, j] = params_est[4]
    # calclate log-likelihood on train, test, all datasets
    params_tuple = (mu_bp, alpha_n_bp, alpha_r_bp, alpha_br_bp, alpha_gr_bp, beta)
    ll_train_full, num_events_train = model_LL_4_alpha_single_beta(params_tuple, block_pairs_train, end_time,
                                                                   block_pair_M_train, n_nodes_c, n_classes)
    return params_tuple, ll_train_full, num_events_train


def fit_6_alpha_single_beta(events_dict, node_mem, n_classes, end_time, beta):
    # return number of node pairs within a block pair, number of nodes per class
    block_pair_M_train, n_nodes_c = mulch_fit_util.num_nodes_pairs_per_block_pair(node_mem, n_classes)
    block_pairs_train = mulch_fit_util.events_dict_to_events_dict_bp(events_dict, node_mem, n_classes)
    # initialize paramters matrices
    mu_bp = np.zeros((n_classes, n_classes))
    alpha_n_bp = np.zeros((n_classes, n_classes))
    alpha_r_bp = np.zeros((n_classes,n_classes))
    alpha_br_bp = np.zeros((n_classes, n_classes))
    alpha_gr_bp = np.zeros((n_classes, n_classes))
    alpha_al_bp = np.zeros((n_classes, n_classes))
    alpha_alr_bp = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:  # diagonal block pair
                params_est = one_beta_bp.fit_6_alpha_one_beta_dia(block_pairs_train[i][j], end_time,n_nodes_c[j,0],
                                                                 block_pair_M_train[i, j], beta)

            else:   # off-diagonal block pair
                params_est = one_beta_bp.fit_6_alpha_one_beta_dia(block_pairs_train[i][j],block_pairs_train[j][i],
                                                       end_time, n_nodes_c[j,0], block_pair_M_train[i, j],beta)
            mu_bp[i, j] = params_est[0]
            alpha_n_bp[i, j] = params_est[1]
            alpha_r_bp[i, j] = params_est[2]
            alpha_br_bp[i, j] = params_est[3]
            alpha_gr_bp[i, j] = params_est[4]
            alpha_al_bp[i, j] = params_est[5]
            alpha_alr_bp[i, j] = params_est[6]
    # calclate log-likelihood on train, test, all datasets
    params_tuple = (mu_bp, alpha_n_bp, alpha_r_bp, alpha_br_bp, alpha_gr_bp, alpha_al_bp, alpha_alr_bp, beta)
    ll_train_full, num_events_train = model_LL_6_alpha_single_beta(params_tuple, block_pairs_train, end_time,
                                                                   block_pair_M_train, n_nodes_c, n_classes)
    return params_tuple, ll_train_full, num_events_train

#%% rho version -- DELETE later
def model_LL_2_alpha_rho_single_beta(params_tup, events_dict_bp, end_time, M_bp, n_nodes_c, n_classes):
    mu_bp, alpha_bp, rho_bp, beta = params_tup
    ll_full = 0
    num_events = 0
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j: # diagonal block pair
                par = (mu_bp[i,j], alpha_bp[i,j], alpha_bp[i,j]*rho_bp[i, j], beta)
                ll_full += one_beta_bp.LL_2_alpha_one_beta_dia(par, events_dict_bp[i][j], end_time, n_nodes_c[j,0], M_bp[i,j])
                num_events += one_beta_bp.cal_num_events(events_dict_bp[i][j])
            elif j > i: # off-diagonal block pair & (j > i)
                par = (mu_bp[i,j], mu_bp[j, i], alpha_bp[i,j], alpha_bp[j,i], rho_bp[i, j], beta)
                ll_full += one_beta_bp.LL_2_alpha_one_beta_off_rho(par, (events_dict_bp[i][j]), (events_dict_bp[j][i]),
                                                                      end_time, n_nodes_c[i,0],  n_nodes_c[j,0])
                num_events += one_beta_bp.cal_num_events(events_dict_bp[i][j])
                num_events += one_beta_bp.cal_num_events(events_dict_bp[j][i])
    return ll_full, num_events


def fit_2_alpha_rho_single_beta(events_dict, node_mem, n_classes, end_time, beta):
    # return number of node pairs within a block pair, number of nodes per class
    bp_M_train, n_nodes_c = mulch_fit_util.num_nodes_pairs_per_block_pair(node_mem, n_classes)
    bps_train = mulch_fit_util.events_dict_to_events_dict_bp(events_dict, node_mem, n_classes)
    # initialize paramters matrices
    mu_bp = np.zeros((n_classes, n_classes))
    alpha_bp = np.zeros((n_classes, n_classes))
    rho_bp = np.zeros((n_classes,n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:  # diagonal block pair
                params_est = one_beta_bp.fit_2_alpha_one_beta_dia(bps_train[i][j], end_time, n_nodes_c[j,0], bp_M_train[i, j], beta)
                mu_bp[i, j] = params_est[0]
                alpha_bp[i, j] = params_est[1]
                if params_est[1] == 0:
                    rho_bp[i, j] = 0
                else:
                    rho_bp[i, j] = params_est[2]/params_est[1]
            elif j > i:   # off-diagonal block pair & j>i
                params_est = one_beta_bp.fit_2_alpha_one_beta_off_rho(bps_train[i][j], bps_train[j][i], end_time,
                                                           n_nodes_c[i,0], n_nodes_c[j,0], beta)
                mu_bp[i, j] = params_est[0]
                mu_bp[j, i] = params_est[1]
                alpha_bp[i, j] = params_est[2]
                alpha_bp[j, i] = params_est[3]
                rho_bp[i, j] = params_est[4]
                rho_bp[j, i] = params_est[4]
    # calclate log-likelihood on train, test, all datasets
    params_tuple = (mu_bp, alpha_bp, rho_bp, beta)
    ll_train_full, num_events_train = model_LL_2_alpha_rho_single_beta(params_tuple, bps_train, end_time, bp_M_train, n_nodes_c, n_classes)
    return params_tuple, ll_train_full, num_events_train

