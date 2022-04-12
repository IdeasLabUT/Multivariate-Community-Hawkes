import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import time
import random
import pickle
from sklearn.metrics import adjusted_rand_score
import OneBlockFit
import sys
sys.path.append("./dynetworkx/classes")
from impulsedigraph import ImpulseDiGraph
sys.path.append("./CHIP-Network-Model")
from spectral_clustering import spectral_cluster1

#%% helper function

def plot_adj(agg_adj, node_mem, K, s=""):
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
            adj_ordered[i_a: i_a+n_class[a], i_b:i_b+ n_class[b]] =  agg_adj[nodes_per_class[a], :][:, nodes_per_class[b]]
            i_b += n_class[b]
        i_b = 0
        i_a += n_class[a]
    plt.figure()
    plt.pcolor(adj_ordered)
    plt.title("permuted count matrix "+ s)
    plt.show()

def print_model_param_single_beta(params_est):
    print("mu")
    print(params_est[0])
    print("\nalpha_n")
    print(params_est[1])
    print("\nalpha_r")
    print(params_est[2])
    if len(params_est)==4:
        print("\nbeta")
        print(params_est[3])
    else:
        print("\nalpha_br")
        print(params_est[3])
        print("\nalpha_gr")
        print(params_est[4])
        if len(params_est)==6:
            print("\nbeta")
            print(params_est[5])
        else:
            print("\nalpha_al")
            print(params_est[5])
            print("\nalpha_alr")
            print(params_est[6])
            print("\nbeta")
            print(params_est[7])

def print_model_param_kernel_sum(params_est):
    print("mu")
    print(params_est[0])
    print("\nalpha_n")
    print(params_est[1])
    print("\nalpha_r")
    print(params_est[2])
    classes = np.shape(params_est[0])[0]
    if len(params_est) == 5:
        print("\nC")
        for i in range(classes):
            for j in range(classes):
                print(params_est[3][i, j, :], end='\t')
            print(" ")
    else:
        print("\nalpha_br")
        print(params_est[3])
        print("\nalpha_gr")
        print(params_est[4])
        if len(params_est) == 7:
            print("\nC")
            for i in range(classes):
                for j in range(classes):
                    print(params_est[5][i, j, :], end='\t')
                print(" ")
        elif len(params_est) == 9:
            print("\nalpha_al")
            print(params_est[5])
            print("\nalpha_alr")
            print(params_est[6])
            print("\nC")
            for i in range(classes):
                for j in range(classes):
                    print(params_est[7][i, j, :], end='\t')
                print(" ")

def assign_node_membership_for_missing_nodes(node_membership, missing_nodes):
    """
    Assigns the missing nodes to the largest community
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

def events_dict_to_blocks(events_dict, node_mem, n_classes):
    # each block_pair is a dict of events
    block_pair_events_dict = np.zeros((n_classes, n_classes), dtype=np.int).tolist()
    for i in range(n_classes):
        for j in range(n_classes):
            block_pair_events_dict[i][j] = {}
    for u, v in events_dict:
        block_pair_events_dict[node_mem[u]][node_mem[v]][(u,v)] = np.array(events_dict[u,v])
    return block_pair_events_dict

def num_nodes_pairs_per_block_pair(node_mem, n_classes):
    classes, n_node_per_class = np.unique(node_mem, return_counts=True)
    n_node_per_class = n_node_per_class.reshape((n_classes, 1))
    return (np.matmul(n_node_per_class, n_node_per_class.T)- np.diagflat(n_node_per_class), n_node_per_class)

def split_train(events_dict, split_ratio=0.8):
    events_list = list(events_dict.values())
    # find spliting point
    events_array = np.sort(np.concatenate(events_list))
    split_point = round(events_array.shape[0] * split_ratio)
    split_time = events_array[split_point]
    events_dict_t = {}
    for (u, v) in events_dict:
        p = np.array(events_dict[(u,v)])
        p_train = p[p<split_time]
        if len(p_train)!= 0 :
            events_dict_t[(u,v)] = p_train
    return events_dict_t, split_time

def event_dict_to_aggregated_adjacency(num_nodes, event_dicts, dtype=np.float):
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=dtype)

    for (u, v), event_times in event_dicts.items():
        adjacency_matrix[u, v] = len(event_times)

    return adjacency_matrix

def event_dict_to_adjacency(num_nodes, event_dicts, dtype=np.float):
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=dtype)

    for (u, v), event_times in event_dicts.items():
        if len(event_times) != 0:
            adjacency_matrix[u, v] = 1

    return adjacency_matrix

#%% model simulation functions

def simulate_one_beta_model(sim_param, n_nodes, n_classes, p, duration):
    if len(sim_param) == 4:
        mu_sim, alpha_n_sim, alpha_r_sim, beta_sim = sim_param
    elif len(sim_param) == 6:
        mu_sim, alpha_n_sim, alpha_r_sim, alpha_br_sim, alpha_gr_sim, beta_sim = sim_param
    elif len(sim_param) == 8:
        mu_sim, alpha_n_sim, alpha_r_sim, alpha_br_sim, alpha_gr_sim, alpha_al_sim, alpha_alr_sim, beta_sim = sim_param
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
                    if len(sim_param) == 4:
                        par = (mu_sim[i, j], alpha_n_sim[i, j], alpha_r_sim[i, j], beta_sim)
                    elif len(sim_param) == 6:
                        par = (mu_sim[i, j], alpha_n_sim[i, j], alpha_r_sim[i, j], alpha_br_sim[i, j], alpha_gr_sim[i, j], beta_sim)
                    elif len(sim_param) == 8:
                        par = (mu_sim[i, j], alpha_n_sim[i, j], alpha_r_sim[i, j], alpha_br_sim[i, j], alpha_gr_sim[i, j],
                               alpha_al_sim[i, j], alpha_alr_sim[i, j], beta_sim)
                    events_dict = OneBlockFit.simulate_one_beta_dia_2(par, list(class_nodes_list[i]), duration)
                    events_dict_all.update(events_dict)
            elif i < j:
                if len(sim_param) == 4:
                    par_ab = (mu_sim[i, j], alpha_n_sim[i, j], alpha_r_sim[i, j], beta_sim)
                    par_ba = (mu_sim[j, i], alpha_n_sim[j, i], alpha_r_sim[j, i], beta_sim)
                elif len(sim_param) == 6:
                    par_ab = (mu_sim[i, j], alpha_n_sim[i, j], alpha_r_sim[i, j], alpha_br_sim[i, j], alpha_gr_sim[i, j], beta_sim)
                    par_ba = (mu_sim[j, i], alpha_n_sim[j, i], alpha_r_sim[j, i], alpha_br_sim[j, i], alpha_gr_sim[j, i], beta_sim)
                elif len(sim_param) == 8:
                    par_ab = (mu_sim[i, j], alpha_n_sim[i, j], alpha_r_sim[i, j], alpha_br_sim[i, j], alpha_gr_sim[i, j],
                              alpha_al_sim[i, j], alpha_alr_sim[i, j], beta_sim)
                    par_ba = (mu_sim[j, i], alpha_n_sim[j, i], alpha_r_sim[j, i], alpha_br_sim[j, i], alpha_gr_sim[j, i],
                              alpha_al_sim[j, i], alpha_alr_sim[j, i], beta_sim)
                d_ab, d_ba = OneBlockFit.simulate_one_beta_off_2(par_ab, par_ba, list(class_nodes_list[i]), list(class_nodes_list[j]),
                                                                     duration)
                events_dict_all.update(d_ab)
                events_dict_all.update(d_ba)
    return events_dict_all, node_mem_actual

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
                    events_dict = OneBlockFit.simulate_kernel_sum_dia_2(par, list(class_nodes_list[i]), duration)
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
                d_ab, d_ba = OneBlockFit.simulate_kernel_sum_off_2(par_ab, par_ba, list(class_nodes_list[i]), list(class_nodes_list[j]),
                                                                 duration)
                events_dict_all.update(d_ab)
                events_dict_all.update(d_ba)
    return events_dict_all, node_mem_actual

#%% model log-likelihood functions

""" single beta model versions"""
def model_LL_6_alpha_single_beta(params_tup, events_dict_bp, end_time, M_bp, n_nodes_c, n_classes):
    mu_bp, alpha_n_bp, alpha_r_bp, alpha_br_bp, alpha_gr_bp,alpha_al_bp, alpha_alr_bp, beta = params_tup
    ll_full = 0
    num_events = 0
    for i in range(n_classes):
        for j in range(n_classes):
            par = (mu_bp[i, j], alpha_n_bp[i, j], alpha_r_bp[i, j], alpha_br_bp[i, j], alpha_gr_bp[i, j],
                  alpha_al_bp[i, j], alpha_alr_bp[i, j], beta)
            if i == j: # diagonal block pair
                ll_dia = OneBlockFit.LL_n_r_br_gr_al_alr_one_dia(par, events_dict_bp[i][j], end_time, n_nodes_c[j,0], M_bp[i,j])
                ll_full += ll_dia
            else: # off-diagonal block pair
                ll_off = OneBlockFit.LL_n_r_br_gr_al_alr_one_off(par, (events_dict_bp[i][j]), (events_dict_bp[j][i]),
                                                                      end_time, n_nodes_c[j,0],  M_bp[i,j])
                ll_full += ll_off
            # number of event of block_pair
            num_events += OneBlockFit.cal_num_events_2(events_dict_bp[i][j])
    return ll_full, num_events
def model_LL_4_alpha_single_beta(params_tup, events_dict_bp, end_time, M_bp, n_nodes_c, n_classes):
    mu_bp, alpha_n_bp, alpha_r_bp, alpha_br_bp, alpha_gr_bp, beta = params_tup
    ll_full = 0
    num_events = 0
    for i in range(n_classes):
        for j in range(n_classes):
            par = (mu_bp[i, j], alpha_n_bp[i, j], alpha_r_bp[i, j], alpha_br_bp[i, j], alpha_gr_bp[i, j], beta)
            if i == j: # diagonal block pair
                ll_dia = OneBlockFit.LL_n_r_br_gr_one_dia_2(par, events_dict_bp[i][j], end_time, n_nodes_c[j,0], M_bp[i,j])
                ll_full += ll_dia
            else: # off-diagonal block pair
                ll_off = OneBlockFit.LL_n_r_br_gr_one_off_2(par, (events_dict_bp[i][j]), (events_dict_bp[j][i]),
                                                                      end_time, n_nodes_c[j,0],  M_bp[i,j])
                ll_full += ll_off
            # number of event of block_pair
            num_events += OneBlockFit.cal_num_events_2(events_dict_bp[i][j])
    return ll_full, num_events
def model_LL_2_alpha_single_beta(params_tup, events_dict_bp, end_time, M_bp, n_nodes_c, n_classes):
    mu_bp, alpha_n_bp, alpha_r_bp, beta = params_tup
    ll_full = 0
    num_events = 0
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j: # diagonal block pair
                par = (mu_bp[i,j], alpha_n_bp[i,j], alpha_r_bp[i,j], beta)
                ll_full += OneBlockFit.LL_n_r_one_dia_2(par, events_dict_bp[i][j], end_time, n_nodes_c[j,0], M_bp[i,j])
            else: # off-diagonal block pair
                par = (mu_bp[i,j], alpha_n_bp[i,j], alpha_r_bp[i,j], beta)
                ll_full += OneBlockFit.LL_n_r_one_off_2(par, (events_dict_bp[i][j]), (events_dict_bp[j][i]),
                                                                      end_time, n_nodes_c[j,0],  M_bp[i,j])
            # number of event of block_pair
            num_events += OneBlockFit.cal_num_events_2(events_dict_bp[i][j])
    return ll_full, num_events
def model_LL_2_alpha_rho_single_beta(params_tup, events_dict_bp, end_time, M_bp, n_nodes_c, n_classes):
    mu_bp, alpha_bp, rho_bp, beta = params_tup
    ll_full = 0
    num_events = 0
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j: # diagonal block pair
                par = (mu_bp[i,j], alpha_bp[i,j], alpha_bp[i,j]*rho_bp[i, j], beta)
                ll_full += OneBlockFit.LL_n_r_one_dia_2(par, events_dict_bp[i][j], end_time, n_nodes_c[j,0], M_bp[i,j])
                num_events += OneBlockFit.cal_num_events_2(events_dict_bp[i][j])
            elif j > i: # off-diagonal block pair & (j > i)
                par = (mu_bp[i,j], mu_bp[j, i], alpha_bp[i,j], alpha_bp[j,i], rho_bp[i, j], beta)
                ll_full += OneBlockFit.LL_n_r_one_off_rho(par, (events_dict_bp[i][j]), (events_dict_bp[j][i]),
                                                                      end_time, n_nodes_c[i,0],  n_nodes_c[j,0])
                num_events += OneBlockFit.cal_num_events_2(events_dict_bp[i][j])
                num_events += OneBlockFit.cal_num_events_2(events_dict_bp[j][i])
    return ll_full, num_events
# used to calculate log-likelihood - external call
def model_LL_single_beta_external(param, events_dict, node_mem, k, end_time):
    block_pair_M, n_nodes_c = num_nodes_pairs_per_block_pair(node_mem, k)
    events_dict_block_pair = events_dict_to_blocks(events_dict, node_mem, k)
    if len(param)==4:
        ll, n_event = model_LL_2_alpha_single_beta(param, events_dict_block_pair, end_time, block_pair_M, n_nodes_c, k)
    elif len(param)==6:
        ll, n_event = model_LL_4_alpha_single_beta(param, events_dict_block_pair, end_time, block_pair_M, n_nodes_c, k)
    elif len(param)==8:
        ll, n_event = model_LL_6_alpha_single_beta(param, events_dict_block_pair, end_time, block_pair_M, n_nodes_c, k)
    return ll, n_event

""" sum of kernels model versions """
def model_LL_2_alpha_kernel_sum(params_tup, events_dict_bp, end_time, M_bp, n_nodes_c, n_classes, ref=False):
    mu_bp, alpha_n_bp, alpha_r_bp, C_bp, betas = params_tup
    LL_bp = np.zeros((n_classes, n_classes))
    num_events = 0
    for i in range(n_classes):
        for j in range(n_classes):
            par = (mu_bp[i, j], alpha_n_bp[i, j], alpha_r_bp[i, j], C_bp[i, j, :], betas)
            if i == j:  # diagonal block pair
                ll_dia = OneBlockFit.LL_2_alpha_kernel_sum_dia(par, events_dict_bp[i][j], end_time, n_nodes_c[j, 0], M_bp[i, j])
                LL_bp[i, j] = ll_dia
            else:  # off-diagonal block pair
                ll_off = OneBlockFit.LL_2_alpha_kernel_sum_off(par, events_dict_bp[i][j], events_dict_bp[j][i], end_time,
                                                               n_nodes_c[j, 0], M_bp[i, j])
                LL_bp[i, j] = ll_off
            # number of event of block_pair
            num_events += OneBlockFit.cal_num_events_2(events_dict_bp[i][j])
    if ref:
        return LL_bp, num_events
    else:
        return np.sum(LL_bp), num_events
def model_LL_4_alpha_kernel_sum(params_tup, events_dict_bp, end_time, M_bp, n_nodes_c, n_classes, ref=False):
    mu_bp, alpha_n_bp, alpha_r_bp, alpha_br_bp, alpha_gr_bp, C_bp, betas = params_tup
    LL_bp = np.zeros((n_classes, n_classes))
    num_events = 0
    for i in range(n_classes):
        for j in range(n_classes):
            par = (mu_bp[i, j], alpha_n_bp[i, j], alpha_r_bp[i, j], alpha_br_bp[i, j], alpha_gr_bp[i, j],
                   C_bp[i, j],betas)
            if i == j:  # diagonal block pair
                ll_dia = OneBlockFit.LL_4_alpha_kernel_sum_dia(par, events_dict_bp[i][j], end_time, n_nodes_c[j, 0], M_bp[i, j])
                LL_bp[i, j] = ll_dia
            else:  # off-diagonal block pair
                ll_off = OneBlockFit.LL_4_alpha_kernel_sum_off(par, events_dict_bp[i][j], events_dict_bp[j][i], end_time,
                                                               n_nodes_c[j, 0], M_bp[i, j])
                LL_bp[i, j] = ll_off
            # number of event of block_pair
            num_events += OneBlockFit.cal_num_events_2(events_dict_bp[i][j])
    if ref:
        return LL_bp, num_events
    else:
        return np.sum(LL_bp), num_events
def model_LL_6_alpha_kernel_sum(params_tup, events_dict_bp, end_time, M_bp, n_nodes_c, n_classes, ref=False):
    mu_bp, alpha_n_bp, alpha_r_bp, alpha_br_bp, alpha_gr_bp, alpha_al_bp, alpha_alr_bp, C_bp, betas = params_tup
    LL_bp = np.zeros((n_classes, n_classes))
    num_events = 0
    for i in range(n_classes):
        for j in range(n_classes):
            par = (mu_bp[i, j], alpha_n_bp[i, j], alpha_r_bp[i, j], alpha_br_bp[i, j], alpha_gr_bp[i, j],
                   alpha_al_bp[i, j], alpha_alr_bp[i, j], C_bp[i, j], betas)
            if i == j:  # diagonal block pair
                ll_dia = OneBlockFit.LL_6_alpha_kernel_sum_dia(par, events_dict_bp[i][j], end_time, n_nodes_c[j, 0], M_bp[i, j])
                LL_bp[i, j] = ll_dia
            else:  # off-diagonal block pair
                ll_off = OneBlockFit.LL_6_alpha_kernel_sum_off(par, events_dict_bp[i][j], events_dict_bp[j][i], end_time,
                                                               n_nodes_c[j, 0], M_bp[i, j])
                LL_bp[i, j] = ll_off
            # number of event of block_pair
            num_events += OneBlockFit.cal_num_events_2(events_dict_bp[i][j])
    if ref:
        return LL_bp, num_events
    else:
        return np.sum(LL_bp), num_events
# used to calculate log-likelihood - external call
def model_LL_kernel_sum_external(param, events_dict, node_mem, k, end_time, ref=False):
    block_pair_M, n_nodes_c = num_nodes_pairs_per_block_pair(node_mem, k)
    events_dict_block_pair = events_dict_to_blocks(events_dict, node_mem, k)
    if len(param) == 5:
        return model_LL_2_alpha_kernel_sum(param, events_dict_block_pair, end_time, block_pair_M, n_nodes_c, k, ref)
    elif len(param) == 7:
        return model_LL_4_alpha_kernel_sum(param, events_dict_block_pair, end_time, block_pair_M, n_nodes_c, k, ref)
    elif len(param) == 9:
        return model_LL_6_alpha_kernel_sum(param, events_dict_block_pair, end_time, block_pair_M, n_nodes_c, k, ref)

#%% Model fitting functions

""" single beta model versions """
def fit_2_alpha_single_beta(events_dict, node_mem, n_classes, end_time, beta):
    # return number of node pairs within a block pair, number of nodes per class
    bp_M_train, n_nodes_c = num_nodes_pairs_per_block_pair(node_mem, n_classes)
    bps_train = events_dict_to_blocks(events_dict, node_mem, n_classes)
    # initialize paramters matrices
    mu_bp = np.zeros((n_classes, n_classes))
    alpha_n_bp = np.zeros((n_classes, n_classes))
    alpha_r_bp = np.zeros((n_classes,n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:  # diagonal block pair
                params_est = OneBlockFit.fit_n_r_one_dia_2(bps_train[i][j], end_time, n_nodes_c[j,0], bp_M_train[i, j], beta)

            else:   # off-diagonal block pair
                params_est = OneBlockFit.fit_n_r_one_off_2(bps_train[i][j], bps_train[j][i], end_time,
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
    block_pair_M_train, n_nodes_c = num_nodes_pairs_per_block_pair(node_mem, n_classes)
    block_pairs_train = events_dict_to_blocks(events_dict, node_mem, n_classes)
    # initialize paramters matrices
    mu_bp = np.zeros((n_classes, n_classes))
    alpha_n_bp = np.zeros((n_classes, n_classes))
    alpha_r_bp = np.zeros((n_classes,n_classes))
    alpha_br_bp = np.zeros((n_classes, n_classes))
    alpha_gr_bp = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:  # diagonal block pair
                params_est = OneBlockFit.fit_n_r_br_gr_one_dia_2(block_pairs_train[i][j], end_time,n_nodes_c[j,0],
                                                                 block_pair_M_train[i, j], beta)

            else:   # off-diagonal block pair
                params_est = OneBlockFit.fit_n_r_br_gr_one_off_2(block_pairs_train[i][j],block_pairs_train[j][i],
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
    block_pair_M_train, n_nodes_c = num_nodes_pairs_per_block_pair(node_mem, n_classes)
    block_pairs_train = events_dict_to_blocks(events_dict, node_mem, n_classes)
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
                params_est = OneBlockFit.fit_n_r_br_gr_al_alr_one_dia(block_pairs_train[i][j], end_time,n_nodes_c[j,0],
                                                                 block_pair_M_train[i, j], beta)

            else:   # off-diagonal block pair
                params_est = OneBlockFit.fit_n_r_br_gr_al_alr_one_off(block_pairs_train[i][j],block_pairs_train[j][i],
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
def fit_2_alpha_rho_single_beta(events_dict, node_mem, n_classes, end_time, beta):
    # return number of node pairs within a block pair, number of nodes per class
    bp_M_train, n_nodes_c = num_nodes_pairs_per_block_pair(node_mem, n_classes)
    bps_train = events_dict_to_blocks(events_dict, node_mem, n_classes)
    # initialize paramters matrices
    mu_bp = np.zeros((n_classes, n_classes))
    alpha_bp = np.zeros((n_classes, n_classes))
    rho_bp = np.zeros((n_classes,n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:  # diagonal block pair
                params_est = OneBlockFit.fit_n_r_one_dia_2(bps_train[i][j], end_time, n_nodes_c[j,0], bp_M_train[i, j], beta)
                mu_bp[i, j] = params_est[0]
                alpha_bp[i, j] = params_est[1]
                if params_est[1] == 0:
                    rho_bp[i, j] = 0
                else:
                    rho_bp[i, j] = params_est[2]/params_est[1]
            elif j > i:   # off-diagonal block pair & j>i
                params_est = OneBlockFit.fit_n_r_one_off_rho(bps_train[i][j], bps_train[j][i], end_time,
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
# single beta model fitting funtion - external call - n_alpha specifies version of the model
def model_fit_single_beta(n_alpha, events_dict, node_mem, n_classes, end_time, beta):
    if n_alpha == 2:
        return fit_2_alpha_single_beta(events_dict, node_mem, n_classes, end_time, beta)
    elif n_alpha == 4:
        return fit_4_alpha_single_beta(events_dict, node_mem, n_classes, end_time, beta)
    elif n_alpha == 6:
        return fit_6_alpha_single_beta(events_dict, node_mem, n_classes, end_time, beta)
    else:
        print(" number of alpha parameter should be 2, 4, or 6")

""" sum of kernels model versions """
def fit_2_alpha_kernel_sum(events_dict, node_mem, n_classes, end_time, betas, ref=False):
    # return number of node pairs within a block pair, number of nodes per class
    block_pair_M_train, n_nodes_c = num_nodes_pairs_per_block_pair(node_mem, n_classes)
    block_pairs_train = events_dict_to_blocks(events_dict, node_mem, n_classes)
    # initialize paramters matrices
    mu_bp = np.zeros((n_classes, n_classes))
    alpha_n_bp = np.zeros((n_classes, n_classes))
    alpha_r_bp = np.zeros((n_classes,n_classes))
    C_bp = np.zeros((n_classes, n_classes, np.size(betas)))
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:  # diagonal block pair
                params_est = OneBlockFit.fit_2_alpha_kernel_sum_dia(block_pairs_train[i][j], end_time, n_nodes_c[j, 0],
                                                                    block_pair_M_train[i, j], betas)
            else:   # off-diagonal block pair
                params_est = OneBlockFit.fit_2_alpha_kernel_sum_off(block_pairs_train[i][j], block_pairs_train[j][i],
                                                                    end_time, n_nodes_c[j,0], block_pair_M_train[i, j], betas)
            mu_bp[i, j] = params_est[0]
            alpha_n_bp[i, j] = params_est[1]
            alpha_r_bp[i, j] = params_est[2]
            C_bp[i,j,:] = params_est[3]
    # calclate log-likelihood on train, test, all datasets
    params_tuple = (mu_bp, alpha_n_bp, alpha_r_bp, C_bp, betas)
    ll_train, num_events_train = model_LL_2_alpha_kernel_sum(params_tuple, block_pairs_train, end_time,
                                                                  block_pair_M_train, n_nodes_c, n_classes, ref)
    if ref:
        return params_tuple, ll_train, num_events_train, block_pairs_train, n_nodes_c
    else:
        return params_tuple, ll_train, num_events_train
def fit_4_alpha_kernel_sum(events_dict, node_mem, n_classes, end_time, betas, ref=False):
    # return number of node pairs within a block pair, number of nodes per class
    block_pair_M_train, n_nodes_c = num_nodes_pairs_per_block_pair(node_mem, n_classes)
    block_pairs_train = events_dict_to_blocks(events_dict, node_mem, n_classes)
    # initialize paramters matrices
    mu_bp = np.zeros((n_classes, n_classes))
    alpha_n_bp = np.zeros((n_classes, n_classes))
    alpha_r_bp = np.zeros((n_classes,n_classes))
    alpha_br_bp = np.zeros((n_classes, n_classes))
    alpha_gr_bp = np.zeros((n_classes, n_classes))
    C_bp = np.zeros((n_classes, n_classes, np.size(betas)))
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:  # diagonal block pair
                params_est = OneBlockFit.fit_4_alpha_kernel_sum_dia(block_pairs_train[i][j], end_time, n_nodes_c[j, 0],
                                                                    block_pair_M_train[i, j], betas)
            else:   # off-diagonal block pair
                params_est = OneBlockFit.fit_4_alpha_kernel_sum_off(block_pairs_train[i][j], block_pairs_train[j][i],
                                                                    end_time, n_nodes_c[j,0], block_pair_M_train[i, j], betas)
            mu_bp[i, j] = params_est[0]
            alpha_n_bp[i, j] = params_est[1]
            alpha_r_bp[i, j] = params_est[2]
            alpha_br_bp[i, j] = params_est[3]
            alpha_gr_bp[i, j] = params_est[4]
            C_bp[i,j,:] = params_est[5]
    # calclate log-likelihood on train, test, all datasets
    params_tuple = (mu_bp, alpha_n_bp, alpha_r_bp, alpha_br_bp, alpha_gr_bp, C_bp, betas)
    ll_train, num_events_train = model_LL_4_alpha_kernel_sum(params_tuple, block_pairs_train, end_time,
                                                                  block_pair_M_train, n_nodes_c, n_classes, ref)
    if ref:
        return params_tuple, ll_train, num_events_train, block_pairs_train, n_nodes_c
    else:
        return params_tuple, ll_train, num_events_train
def fit_6_alpha_kernel_sum(events_dict, node_mem, n_classes, end_time, betas, ref=False):
    # return number of node pairs within a block pair, number of nodes per class
    block_pair_M_train, n_nodes_c = num_nodes_pairs_per_block_pair(node_mem, n_classes)
    block_pairs_train = events_dict_to_blocks(events_dict, node_mem, n_classes)
    # initialize paramters matrices
    mu_bp = np.zeros((n_classes, n_classes))
    alpha_n_bp = np.zeros((n_classes, n_classes))
    alpha_r_bp = np.zeros((n_classes,n_classes))
    alpha_br_bp = np.zeros((n_classes, n_classes))
    alpha_gr_bp = np.zeros((n_classes, n_classes))
    alpha_al_bp = np.zeros((n_classes, n_classes))
    alpha_alr_bp = np.zeros((n_classes, n_classes))
    C_bp = np.zeros((n_classes, n_classes, np.size(betas)))
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:  # diagonal block pair
                params_est = OneBlockFit.fit_6_alpha_kernel_sum_dia(block_pairs_train[i][j], end_time, n_nodes_c[j, 0],
                                                                    block_pair_M_train[i, j], betas)
            else:   # off-diagonal block pair
                params_est = OneBlockFit.fit_6_alpha_kernel_sum_off(block_pairs_train[i][j], block_pairs_train[j][i],
                                                                    end_time, n_nodes_c[j,0], block_pair_M_train[i, j], betas)
            mu_bp[i, j] = params_est[0]
            alpha_n_bp[i, j] = params_est[1]
            alpha_r_bp[i, j] = params_est[2]
            alpha_br_bp[i, j] = params_est[3]
            alpha_gr_bp[i, j] = params_est[4]
            alpha_al_bp[i, j] = params_est[5]
            alpha_alr_bp[i, j] = params_est[6]
            C_bp[i,j,:] = params_est[7]
    # calclate log-likelihood on train, test, all datasets
    params_tuple = (mu_bp, alpha_n_bp, alpha_r_bp, alpha_br_bp, alpha_gr_bp, alpha_al_bp, alpha_alr_bp, C_bp, betas)
    ll_train, num_events_train = model_LL_6_alpha_kernel_sum(params_tuple, block_pairs_train, end_time,
                                                                  block_pair_M_train, n_nodes_c, n_classes, ref)
    if ref:
        return params_tuple, ll_train, num_events_train, block_pairs_train, n_nodes_c
    else:
        return params_tuple, ll_train, num_events_train
# sum of kernels model fitting funtion - external call - n_alpha specifies version of the model
def model_fit_kernel_sum(n_alpha, events_dict, node_mem, n_classes, end_time, betas, ref=False):
    if n_alpha == 2:
        return fit_2_alpha_kernel_sum(events_dict, node_mem, n_classes, end_time, betas, ref)
    elif n_alpha == 4:
        return fit_4_alpha_kernel_sum(events_dict, node_mem, n_classes, end_time, betas, ref)
    elif n_alpha == 6:
        return fit_6_alpha_kernel_sum(events_dict, node_mem, n_classes, end_time, betas, ref)
    else:
        print(" number of alpha parameter should be 2, 4, or 6")



#%% motif counts functions

def get_motifs():
    motifs = [[((1, 2), (3, 2), (1, 2)), ((1, 2), (3, 2), (2, 1)), ((1, 2), (3, 2), (1, 3)), ((1, 2), (3, 2), (3, 1)),
               ((1, 2), (3, 2), (2, 3)), ((1, 2), (3, 2), (3, 2))],
              [((1, 2), (2, 3), (1, 2)), ((1, 2), (2, 3), (2, 1)), ((1, 2), (2, 3), (1, 3)), ((1, 2), (2, 3), (3, 1)),
               ((1, 2), (2, 3), (2, 3)), ((1, 2), (2, 3), (3, 2))],
              [((1, 2), (3, 1), (1, 2)), ((1, 2), (3, 1), (2, 1)), ((1, 2), (3, 1), (1, 3)), ((1, 2), (3, 1), (3, 1)),
               ((1, 2), (3, 1), (2, 3)), ((1, 2), (3, 1), (3, 2))],
              [((1, 2), (1, 3), (1, 2)), ((1, 2), (1, 3), (2, 1)), ((1, 2), (1, 3), (1, 3)), ((1, 2), (1, 3), (3, 1)),
               ((1, 2), (1, 3), (2, 3)), ((1, 2), (1, 3), (3, 2))],
              [((1, 2), (2, 1), (1, 2)), ((1, 2), (2, 1), (2, 1)), ((1, 2), (2, 1), (1, 3)), ((1, 2), (2, 1), (3, 1)),
               ((1, 2), (2, 1), (2, 3)), ((1, 2), (2, 1), (3, 2))],
              [((1, 2), (1, 2), (1, 2)), ((1, 2), (1, 2), (2, 1)), ((1, 2), (1, 2), (1, 3)), ((1, 2), (1, 2), (3, 1)),
               ((1, 2), (1, 2), (2, 3)), ((1, 2), (1, 2), (3, 2))]]
    return motifs

def cal_recip_trans_motif(events_dict, N, motif_delta,  dataset="", save=False):
    adj = event_dict_to_adjacency(N, events_dict)
    net = nx.DiGraph(adj)
    recip = nx.overall_reciprocity(net)
    trans = nx.transitivity(net)
    # create ImpulseDiGraph from network
    G_data = ImpulseDiGraph()
    for (u,v) in events_dict:
        events_list_uv = events_dict[u,v]
        for t in events_list_uv:
            G_data.add_edge(u,v,t)
    print(f"{1:>10}{2:>10}{3:>10}{4:>10}{5:>10}{6:>10}")
    motifs = get_motifs()
    dataset_motif = np.zeros((6,6),dtype=int)
    for i in range(6):
        for j in range(6):
            dataset_motif[i,j] = G_data.calculate_temporal_motifs(motifs[i][j], motif_delta)
        print(f"{dataset_motif[i,0]:>10}{dataset_motif[i,1]:>10}{dataset_motif[i,2]:>10}{dataset_motif[i,3]:>10}"
              f"{dataset_motif[i,4]:>10}{dataset_motif[i,5]:>10}")
    n_events = OneBlockFit.cal_num_events_2(events_dict)
    if save:
        results_dict = {}
        results_dict["dataset_motif"] = dataset_motif
        results_dict["dataset_recip"] = recip
        results_dict["dataset_trans"] = trans
        results_dict["dataset_n_events"] = n_events
        with open(f"{dataset}_counts.p", 'wb') as fil:
            pickle.dump(results_dict, fil)
    return recip, trans, dataset_motif
