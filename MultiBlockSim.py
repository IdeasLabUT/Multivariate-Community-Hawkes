import numpy as np
import pandas as pd
import time
import random
import pickle
from scipy.stats import pareto
from bisect import bisect_right
import MultiBlockFit as MBHP
from spectral_clustering import spectral_cluster, LF_spectral_cluster, spectral_cluster1
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
import OneBlockFit

import sys
sys.path.append("./CHIP-Network-Model")
from generative_model_utils import event_dict_to_aggregated_adjacency, event_dict_to_adjacency
sys.path.append("./hawkes")
from MHP import MHP



#%% helper functions
def arrage_fit_param(param, K, node_mem_true, node_mem_fit, betas):
    # arrange fit_parameters before print
    ar = [None] * K
    for a in range(K):
        idx = list(node_mem_true).index(a)
        ar[a] = node_mem_fit[idx]
    arranged_fit_param = []
    for i in range(n_alpha + 2):
        arranged_fit_p = np.zeros_like(param[i])
        for a in range(K):
            for b in range(K):
                arranged_fit_p[a, b] = param[i][ar[a], ar[b]]
        arranged_fit_param.append(arranged_fit_p)
    arranged_fit_param.append(betas)
    return arranged_fit_param

def get_simulation_params(n_classes, level, n_alpha, sum):
    if (n_classes == 2 and level == 1 and n_alpha==6 and sum == True):
        # simulation parameters n_classes=2 , easy
        mu_sim = np.array([[0.0001, 0.0005], [0.0005, 0.0001]])
        alpha_n_sim = np.array([[0.03, 0.2], [0.2, 0.03]])
        alpha_r_sim = np.array([[0.0001, 0.1], [0.1, 0.0001]])
        alpha_br_sim = np.array([[0.0009, 0.002], [0.002, 0.0009]])
        alpha_gr_sim = np.array([[0.0, 0.001], [0.001, 0.0]])
        alpha_al_sim = np.array([[0.0001, 0.003], [0.003, 0.0001]])
        alpha_alr_sim = np.array([[0.0, 0.001], [0.001, 0.0]])
        C_sim = np.array([[[0.33, 0.33, 0.34]] * n_classes for _ in range(n_classes)])
        betas = np.array([0.02, 0.6, 20])
    elif(n_classes == 2 and level == 1000 and n_alpha==6 and sum == True):
        theta_off = [0.0002, 0.3, 0.3, 0.004, 0.001, 0.003, 0.001]
        theta_dia = [0.0002, 0.02, 0.01, 0.0002, 0.0001, 0.0002, 0.00005]
        # dissortative mixing
        mu_sim = np.ones((n_classes, n_classes)) * theta_off[0]
        mu_sim[np.diag_indices_from(mu_sim)] = theta_dia[0]

        alpha_n_sim = np.ones((n_classes, n_classes)) * theta_off[1]
        alpha_n_sim[np.diag_indices_from(mu_sim)] = theta_dia[1]

        alpha_r_sim = np.ones((n_classes, n_classes)) * theta_off[2]
        alpha_r_sim[np.diag_indices_from(mu_sim)] = theta_dia[2]

        alpha_br_sim = np.ones((n_classes, n_classes)) * theta_off[3]
        alpha_br_sim[np.diag_indices_from(mu_sim)] = theta_dia[3]

        alpha_gr_sim = np.ones((n_classes, n_classes)) * theta_off[4]
        alpha_gr_sim[np.diag_indices_from(mu_sim)] = theta_dia[4]

        alpha_al_sim = np.ones((n_classes, n_classes)) * theta_off[5]
        alpha_al_sim[np.diag_indices_from(mu_sim)] = theta_dia[5]

        alpha_alr_sim = np.ones((n_classes, n_classes)) * theta_off[6]
        alpha_alr_sim[np.diag_indices_from(mu_sim)] = theta_dia[6]
        C_sim = np.array([[[0.33, 0.33, 0.34]] * n_classes for _ in range(n_classes)])
        betas_recip = np.array([7 * 2, 1, 1 / 12])  # [2week, 1day, 2hour]
        betas = np.reciprocal(betas_recip)
        param = (mu_sim, alpha_n_sim, alpha_r_sim, alpha_br_sim, alpha_gr_sim, alpha_al_sim, alpha_alr_sim, C_sim, betas)
    elif (n_classes == 3 and level == 1 and n_alpha==2 and sum == False):
        # simulation parameters n_classes=3, level= easy, parameters for new kernel
        mu_sim = np.array([[0.0008, 0.0006, 0.0006], [0.0007, 0.001, 0.0006],[0.0006, 0.0005, 0.001]])
        alpha_n_sim = np.array([[0.2, 0.02, 0.05], [0.07, 0.08, 0.03],[0.05, 0.01, 0.2]])
        rho_sim = np.array([[0.5, 2, 1], [2, 2.5, 1.5],[1, 1.5, 1]])
        alpha_r_sim = alpha_n_sim * rho_sim
        beta_sim = 1
        param = (mu_sim, alpha_n_sim, alpha_r_sim, beta_sim)
    elif (n_classes == 3 and level == 1 and n_alpha==4 and sum == False):
        # simulation parameters n_classes=3, level= easy, parameters for new kernel
        mu_sim = np.array([[0.0015, 0.001, 0.001], [0.0015, 0.0005, 0.001],
                           [0.001, 0.001, 0.002]])
        alpha_n_sim = np.array([[0.8, 0.1, 0.1], [0.07, 0.6, 0.2],
                                [0.09, 0.1, 0.7]])
        alpha_r_sim = np.array([[1, 0.6, 0.6], [0.5, 0.6, 0.4],
                                [0.3, 0.3, 0.8]])
        alpha_br_sim = np.array([[0.009, 0.005, 0.005], [0.01, 0.01, 0.0005],
                                 [0.001, 0.005, 0.01]])
        alpha_gr_sim = np.array([[0.01, 0.005, 0.006], [0.001, 0.01, 0.001],
                                 [0.001, 0.005, 0.009]])
        beta_sim = np.ones((3, 3)) * 5
        param = (mu_sim, alpha_n_sim, alpha_r_sim, alpha_br_sim, alpha_gr_sim,
                 beta_sim)
    elif (n_classes == 3 and level == 100 and n_alpha==4 and sum == False):
        # simulation parameters n_classes=3, level= 100 nodes
        mu_sim = np.array([[0.005, 0.001, 0.002], [0.002, 0.007, 0.001],
                           [0.001, 0.002, 0.006]])
        alpha_n_sim = np.array([[0.08, 0.01, 0.01], [0.07, 0.06, 0.002],
                                [0.09, 0.01, 0.07]])
        alpha_r_sim = np.array([[0.1, 0.06, 0.06], [0.05, 0.06, 0.04],
                                [0.03, 0.03, 0.08]])
        alpha_br_sim = np.array([[0.009, 0.005, 0.005], [0.01, 0.01, 0.0005],
                                 [0.001, 0.005, 0.01]])
        alpha_gr_sim = np.array([[0.01, 0.005, 0.006], [0.001, 0.01, 0.001],
                                 [0.001, 0.005, 0.009]])
        beta_sim = np.array([[4, 6, 7], [6, 4, 5], [7, 5, 5]], dtype=float)
        param = (mu_sim, alpha_n_sim, alpha_r_sim, alpha_br_sim, alpha_gr_sim,
                 beta_sim)
    elif (n_classes == 3 and level == 100 and n_alpha==4 and sum == True):
        # simulation parameters
        mu_sim = np.array([[0.0005, 0.0005, 0.0005], [0.0001, 0.0001, 0.0001],
                           [0.0005, 0.0005, 0.0005]])
        alpha_n_sim = np.array([[0.4, 0.05, 0.7], [0.01, 0.01, 0.01],
                                [0.03, 0.03, 0.01]])
        alpha_r_sim = np.array([[0.3, 0.08, 0.8], [0.01, 0.01, 0.01],
                                [0.03, 0.03, 0.1]])
        alpha_br_sim = np.array([[0.004, 0.001, 0.001], [0.002, 0.003, 0.001],
                                 [0.001, 0.002, 0.002]])
        alpha_gr_sim = np.array([[0.003, 0.001, 0.002], [0.002, 0.001, 0.002],
                                 [0.001, 0.002, 0.0002]])
        C_sim = np.array([[[0.1, 0.6, 0.3], [0.2, 0.2, 0.6], [0.3, 0.3, 0.4]],
                          [[0.1, 0.6, 0.3], [0.1, 0.3, 0.6], [0.1, 0.3, 0.6]],
                          [[0.3, 0.4, 0.3], [0.1, 0.1, 0.8], [0.3, 0.3, 0.4]]])
        betas = np.array([0.02, 0.6, 50])
        param = (mu_sim, alpha_n_sim, alpha_r_sim, alpha_br_sim, alpha_gr_sim,
                 C_sim, betas)
    elif (n_classes == 3 and level == 1 and n_alpha==6 and sum == False):
        # simulation parameters
        mu_sim = np.array([[0.0005, 0.0005, 0.0004],
                           [0.0003, 0.0008, 0.0003],
                           [0.0003, 0.0004, 0.0007]])
        alpha_n_sim = np.array([[0.01, 0.03, 0.02],
                                [0.0, 0.3, 0.01],
                                [0.0, 0.03, 0.1]])
        alpha_r_sim = np.array([[0.1, 0.05, 0.07],
                                [0.01, 0.001, 0.01],
                                [0.001, 0.0, 0.05]])
        alpha_br_sim = np.array([[0.002, 0.0005, 0.0001], [0.0, 0.005, 0.0006], [0.0001, 0.0009, 0.03]])
        alpha_gr_sim = np.array([[0.001, 0.0, 0.0001], [0.0, 0.008, 0.0001], [0.0, 0.0002, 0.0]])
        alpha_al_sim = np.array([[0.001, 0.0001, 0.0], [0.0, 0.002, 0.0], [0.0001, 0.0007, 0.001]])
        alpha_alr_sim = np.array([[0.003, 0.0001, 0.0001], [0.0, 0.001, 0.0006], [0.0001, 0.0, 0.003]])
        beta = 1
        param = (mu_sim, alpha_n_sim, alpha_r_sim, alpha_br_sim, alpha_gr_sim, alpha_al_sim, alpha_alr_sim, beta)
    elif (n_classes == 3 and level == 1 and n_alpha==6 and sum == True):
        # simulation parameters
        mu_sim = np.array([[0.0015, 0.0005, 0.001],
                           [0.0003, 0.0007, 0.0003],
                           [0.0005, 0.0005, 0.001]])
        alpha_n_sim = np.array([[0.1, 0.03, 0.02],
                                [0.0, 0.2, 0.01],
                                [0.0, 0.03, 0.1]])
        alpha_r_sim = np.array([[0.09, 0.05, 0.07],
                                [0.01, 0.1, 0.01],
                                [0.001, 0.0, 0.05]])
        alpha_br_sim = np.array([[0.002, 0.0005, 0.0001], [0.0, 0.005, 0.0006], [0.0001, 0.0009, 0.003]])
        alpha_gr_sim = np.array([[0.001, 0.0, 0.0001], [0.0, 0.008, 0.0001], [0.0, 0.0002, 0.0]])
        alpha_al_sim = np.array([[0.003, 0.0001, 0.0], [0.0, 0.002, 0.0], [0.0001, 0.0007, 0.0001]])
        alpha_alr_sim = np.array([[0.001, 0.0001, 0.0001], [0.0, 0.005, 0.0006], [0.0001, 0.0, 0.003]])
        C_sim = np.array([[[0.5, 0.5]]*n_classes for _ in range(n_classes)])
        # C_sim = np.array([[[0.33, 0.34, 0.33], [0.33, 0.34, 0.33], [0.33, 0.34, 0.33]]] * n_classes)
        betas = np.array([0.01, 20])
        param = (mu_sim, alpha_n_sim, alpha_r_sim, alpha_br_sim, alpha_gr_sim, alpha_al_sim, alpha_alr_sim, C_sim, betas)
    elif (n_classes == 3 and level == 100 and n_alpha==6 and sum == True):
        # simulation parameters
        mu_sim = np.array([[0.0001, 0.0005, 0.001],
                           [0.002, 0.0001, 0.001],
                           [0.004, 0.0005, 0.0001]])
        alpha_n_sim = np.array([[0.001, 0.02, 0.02],
                                [0.01, 0.001, 0.01],
                                [0.02, 0.01, 0.001]])
        alpha_r_sim = np.array([[0.001, 0.01, 0.02],
                                [0.06, 0.0, 0.07],
                                [0.01, 0.001, 0.001]])
        alpha_br_sim = np.array([[0.0001, 0.0005, 0.001], [0.0004, 0.0, 0.001], [0.001, 0.0009, 0.0001]])
        alpha_gr_sim = np.array([[0.0, 0.0, 0.0001], [0.0, 0.0, 0.0001], [0.0002, 0.0, 0.0]])
        alpha_al_sim = np.array([[0.0, 0.0001, 0.0001], [0.001, 0.0, 0.001], [0.0001, 0.0007, 0.0]])
        alpha_alr_sim = np.array([[0.0001, 0.0001, 0.0001], [0.0001, 0.0, 0.0006], [0.001, 0.0001, 0.0]])
        # C_sim = np.array([[[0.1, 0.1, 0.8], [0.8, 0.1, 0.1], [0.33, 0.34, 0.33]]]*n_classes)
        C_sim = np.array([[[0.5, 0.5]] * n_classes for _ in range(n_classes)])
        betas = np.array([0.01, 20])
        param = (mu_sim, alpha_n_sim, alpha_r_sim, alpha_br_sim, alpha_gr_sim, alpha_al_sim, alpha_alr_sim, C_sim, betas)
    elif (n_classes == 4 and level == 100 and n_alpha==4 and sum == False):
        # simulation parameters - for 100 nodes trials
        mu_sim = np.array([[0.001, 0.001, 0.0001, 0.0001],
                           [0.001, 0.001, 0.0001, 0.0001],
                           [0.001, 0.001, 0.0001, 0.0001],
                           [0.0001, 0.0001, 0.0001, 0.0001]])
        alpha_n_sim = np.array([[0.1, 0.1, 0.01, 0.01],
                                [0.01, 0.01, 0.01, 0.009],
                                [0.03, 0.03, 0.003, 0.0],
                                [0.02, 0.02, 0.02, 0.02]])
        alpha_r_sim = np.array([[0.3, 0.2, 0.01, 0.01],
                                [0.01, 0.01, 0.01, 0.01],
                                [0.03, 0.03, 0.01, 0.0], [0.01, 0, 0.01,
                                                          0.01]])
        alpha_br_sim = np.array([[0.004, 0.001, 0.001, 0.001],
                                 [0.002, 0.003, 0.001, 0.001],
                                 [0.001, 0.002, 0.002, 0.001],
                                 [0.001, 0.002, 0.002, 0.001]])
        alpha_gr_sim = np.array([[0.003, 0.001, 0.002, 0.001],
                                 [0.002, 0.001, 0.002, 0.001],
                                 [0.001, 0.002, 0.0002, 0.001],
                                 [0.001, 0.001, 0.001, 0.001]])
        beta_sim = 5.0
        param = (mu_sim, alpha_n_sim, alpha_r_sim, alpha_br_sim, alpha_gr_sim,
                 beta_sim)
    elif (n_classes == 4 and level == 1 and n_alpha==6 and sum == True):
        # simulation parameters
        mu_sim = np.array([[0.0005, 0.0001, 0.0001, 0.0001],
                           [0.0001, 0.0004, 0.0001, 0.00001],
                           [0.0001, 0.0001, 0.0003, 0.00005],
                           [0.00001, 0.0001, 0.0001, 0.0002]])
        alpha_n_sim = np.array([[0.4, 0.09, 0.03, 0.0],[0.01, 0.25, 0.01, 0.009],
                                [0.03, 0.03, 0.1, 0.0],[0.0, 0.0, 0.02, 0.01]])
        alpha_r_sim = np.array([[0.1, 0.05, 0.02, 0.01],[0.01, 0.25, 0.01, 0.01],
                                [0.03, 0.03, 0.4, 0.0],[0.0, 0.0, 0.01, 0.1]])
        alpha_br_sim = np.array([[0.002, 0.0001, 0.001, 0.0001],
                                 [0.0001, 0.0002, 0.0007, 0.0003],
                                 [0.0001, 0.0002, 0.003, 0.001],
                                 [0.0, 0.0, 0.0, 0.001]])
        alpha_gr_sim = np.array([[0.001, 0.0001, 0.0002, 0.0001],
                                 [0.0001, 0.0, 0.0002, 0.0001],
                                 [0.0001, 0.0002, 0.001, 0.0001],
                                 [0.0001, 0.0001, 0.0001, 0.001]])
        alpha_al_sim = np.array([[0.0005, 0.0001, 0.0001, 0.0001],
                                 [0.0001, 0.0005, 0.0006, 0.0001],
                                  [0.009, 0.0001, 0.0005, 0.0001],
                                 [0.001, 0.0001, 0.0, 0.001]])
        alpha_alr_sim = np.array([[0.0001, 0.0001, 0.0001, 0.0001], [0.0001, 0.0, 0.0006, 0.0001],
                                  [0.001, 0.0001, 0.0, 0.0001], [0.0, 0.0, 0.0, 0.0001]])
        C_sim = np.array([[[0.33, 0.33, 0.34]] * n_classes for _ in range(n_classes)])
        betas = np.array([0.02, 0.6, 20])
        param = (mu_sim, alpha_n_sim, alpha_r_sim, alpha_br_sim, alpha_gr_sim, alpha_al_sim, alpha_alr_sim, C_sim, betas)
    elif (n_classes == 4 and level == 1000 and n_alpha==6 and sum == True):
        theta_dia = [0.0002, 0.3, 0.3, 0.004, 0.001, 0.003, 0.001]
        theta_off = [0.0002, 0.02, 0.01, 0.0002, 0.0001, 0.0002, 0.00005]
        # assortative mixing
        mu_sim = np.ones((n_classes, n_classes)) * theta_off[0]
        mu_sim[np.diag_indices_from(mu_sim)] = theta_dia[0]

        alpha_n_sim = np.ones((n_classes, n_classes)) * theta_off[1]
        alpha_n_sim[np.diag_indices_from(mu_sim)] = theta_dia[1]

        alpha_r_sim = np.ones((n_classes, n_classes)) * theta_off[2]
        alpha_r_sim[np.diag_indices_from(mu_sim)] = theta_dia[2]

        alpha_br_sim = np.ones((n_classes, n_classes)) * theta_off[3]
        alpha_br_sim[np.diag_indices_from(mu_sim)] = theta_dia[3]

        alpha_gr_sim = np.ones((n_classes, n_classes)) * theta_off[4]
        alpha_gr_sim[np.diag_indices_from(mu_sim)] = theta_dia[4]

        alpha_al_sim = np.ones((n_classes, n_classes)) * theta_off[5]
        alpha_al_sim[np.diag_indices_from(mu_sim)] = theta_dia[5]

        alpha_alr_sim = np.ones((n_classes, n_classes)) * theta_off[6]
        alpha_alr_sim[np.diag_indices_from(mu_sim)] = theta_dia[6]
        C_sim = np.array([[[0.33, 0.33, 0.34]] * n_classes for _ in range(n_classes)])
        betas_recip = np.array([7 * 2, 1, 1 / 12]) # [2week, 1day, 2hour]
        betas = np.reciprocal(betas_recip)
        param = (mu_sim, alpha_n_sim, alpha_r_sim, alpha_br_sim, alpha_gr_sim, alpha_al_sim, alpha_alr_sim, C_sim, betas)
    return param

#%% degree corrected simulation functions for sinlge beta model

def simulate_one_beta_dia_2_corrected(params_sim, a_nodes, end_time, theta_in_a, theta_out_a):
    if len(params_sim) == 4:
        mu_array, alpha_matrix, beta = OneBlockFit.get_array_param_n_r_dia(params_sim, len(a_nodes))
    elif len(params_sim) == 6:
        mu_array, alpha_matrix, beta = OneBlockFit.get_array_param_n_r_br_gr_dia(params_sim, len(a_nodes))
    elif len(params_sim) == 8:
        mu_array, alpha_matrix, beta = OneBlockFit.get_array_param_n_r_br_gr_al_alr_dia(params_sim, len(a_nodes))
    # multibly mu with correction terms
    theta_out_in_a = np.repeat(theta_out_a, len(a_nodes)) * np.tile(theta_in_a, len(a_nodes))
    theta_corrected = np.delete(theta_out_in_a, np.arange(0, len(a_nodes) ** 2, len(a_nodes) + 1))
    mu_corrected = mu_array * theta_corrected
    # multivariate hawkes process object [NOTE: alpha=jump_size/beta, omega=beta]
    P = MHP(mu=mu_corrected, alpha=alpha_matrix, omega=beta)
    P.generate_seq(end_time)
    # assume that timestamps list is ordered ascending with respect to u then v [(0,1), (0,2), .., (1,0), (1,2), ...]
    events_list = []
    for m in range(len(mu_array)):
        i = np.nonzero(P.data[:, 1] == m)[0]
        events_list.append(P.data[i, 0])
    events_dict = OneBlockFit.events_list_to_events_dict_remove_empty_np(events_list, a_nodes)
    return events_dict
def simulate_one_beta_off_2_corrected(param_ab, param_ba, a_nodes, b_nodes, end_time, theta_in_a, theta_out_a, theta_in_b, theta_out_b):
    if len(param_ab) == 4:
        mu_array, alpha_matrix, beta = OneBlockFit.get_array_param_n_r_off(param_ab, param_ba, len(a_nodes), len(b_nodes))
    elif len(param_ab) == 6:
        mu_array, alpha_matrix, beta = OneBlockFit.get_array_param_n_r_br_gr_off(param_ab, param_ba, len(a_nodes), len(b_nodes))
    if len(param_ab) == 8:
        mu_array, alpha_matrix, beta = OneBlockFit.get_array_param_n_r_br_gr_al_alr_off(param_ab, param_ba, len(a_nodes), len(b_nodes))
    # multibly mu with correction terms
    theta_out_a_in_b = np.repeat(theta_out_a, len(b_nodes)) * np.tile(theta_in_b, len(a_nodes))
    theta_out_b_in_a = np.repeat(theta_out_b, len(a_nodes)) * np.tile(theta_in_a, len(b_nodes))
    theta_corrected = np.r_[theta_out_a_in_b, theta_out_b_in_a]
    mu_corrected = mu_array * theta_corrected
    # multivariate hawkes process object [NOTE: alpha=jump_size/beta, omega=beta]
    P = MHP(mu=mu_corrected, alpha=alpha_matrix, omega=beta)
    P.generate_seq(end_time)
    # assume that timestamps list is ordered ascending with respect to u then v [(0,1), (0,2), .., (1,0), (1,2), ...]
    events_list = []
    for m in range(len(mu_array)):
        i = np.nonzero(P.data[:, 1] == m)[0]
        events_list.append(P.data[i, 0])
    M = len(a_nodes) * len(b_nodes)
    events_list_ab = events_list[:M]
    events_list_ba = events_list[M:]
    events_dict_ab = OneBlockFit.events_list_to_events_dict_remove_empty_np_off(events_list_ab, a_nodes, b_nodes)
    events_dict_ba = OneBlockFit.events_list_to_events_dict_remove_empty_np_off(events_list_ba, b_nodes, a_nodes)
    return events_dict_ab, events_dict_ba
# full model simulation
def simulate_one_beta_model_2_corrected(sim_param, n_nodes, n_classes, p, duration):
    if len(sim_param) == 4:
        mu_sim, alpha_n_sim, alpha_r_sim, beta_sim = sim_param
    elif len(sim_param) == 6:
        mu_sim, alpha_n_sim, alpha_r_sim, alpha_br_sim, alpha_gr_sim, beta_sim = sim_param
    elif len(sim_param) == 8:
        mu_sim, alpha_n_sim, alpha_r_sim, alpha_br_sim, alpha_gr_sim, alpha_al_sim, alpha_alr_sim, beta_sim = sim_param
    # Generate theta_in & theta_out from pareto distribution
    theta_in = pareto.rvs(1.5, size=n_nodes)
    theta_out = pareto.rvs(1.5, size=n_nodes)
    # normalize step
    theta_in = theta_in * n_nodes / np.sum(theta_in)
    theta_out = theta_out * n_nodes / np.sum(theta_out)
    # list (n_classes) elements, each element is array of nodes that belong to same class
    nodes_list = list(range(n_nodes))
    random.shuffle(nodes_list)
    p = np.round(np.cumsum(p) * n_nodes).astype(int)
    class_nodes_list = np.array_split(nodes_list, p[:-1])
    class_theta_in_list, class_theta_out_list = [], []
    node_mem_actual = np.zeros((n_nodes,), dtype=int)
    for c in range(n_classes):
        node_mem_actual[class_nodes_list[c]] = c
        # normalize degree paramters of each block
        theta_in_block = theta_in[class_nodes_list[c]]
        # theta_in_block_norm = theta_in_block * len(theta_in_block) / np.sum(theta_in_block)
        class_theta_in_list.append(theta_in_block)
        theta_out_block = theta_out[class_nodes_list[c]]
        # theta_out_block_norm = theta_out_block * len(theta_out_block) / np.sum(theta_out_block)
        class_theta_out_list.append(theta_out_block)
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
                    events_dict = simulate_one_beta_dia_2_corrected(par, list(class_nodes_list[i]), duration, class_theta_in_list[i],
                                                                    class_theta_out_list[i])
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
                d_ab, d_ba = simulate_one_beta_off_2_corrected(par_ab, par_ba, list(class_nodes_list[i]), list(class_nodes_list[j]),
                                                               duration, class_theta_in_list[i], class_theta_out_list[i],
                                                               class_theta_in_list[j], class_theta_out_list[j])
                events_dict_all.update(d_ab)
                events_dict_all.update(d_ba)
    return events_dict_all, node_mem_actual

#%% Main

if __name__ == "__main__":
    SP_RI_test = False
    MSE_test = True
    rho_test = False
    np.set_printoptions(suppress=True)

#%% test LL values, paramters estimation and motif counts and bad and good SP run.
    """ """
    # K, N, T_all = 3, 60, 2000  # number of nodes and duration
    # n_alpha = 6
    # p = [1 / K] * K  # nodes percentage membership
    # split_ratio = 0.8
    #
    # # # # single beta model
    # # sim_param = get_simulation_params(K, 100, n_alpha=4, sum=False)
    # # beta = sim_param[n_alpha+1]
    # # print(f"Simulation: singel beta model model at beta=", beta)
    # # events_dict_all, node_mem_true = MBHP.simulate_one_beta_model(sim_param, N, K, p, T_all)
    #
    # # ## sum of kernels model
    # sim_param = get_simulation_params(K, 1, n_alpha=n_alpha, sum=True)
    # betas = sim_param[n_alpha + 2]
    # print(f"Simulation: sum of kerenels model at betas=", betas)
    # events_dict_all, node_mem_true = MBHP.simulate_sum_kernel_model(sim_param, N, K, p, T_all)
    #
    # n_events_all = OneBlockFit.cal_num_events_2(events_dict_all)
    # print(f"at K={K}, N={N}, balanced membership, split ratio={split_ratio}")
    # print(f"T_all={T_all}, events_all={n_events_all}")
    # # split simulted events into train & all
    # events_dict_train, T_train = MBHP.split_train(events_dict_all, split_ratio=split_ratio)
    # n_event_train = OneBlockFit.cal_num_events_2(events_dict_train)
    # print(f"T_train={T_train:.2f}, train_events={n_event_train}")
    #
    # # compute log-likelihood on true parameters
    # # ll_true_train, _ = MBHP.model_LL_single_beta_external(sim_param, events_dict_train, node_mem_true, K, T_train)
    # # ll_true_all, _ = MBHP.model_LL_single_beta_external(sim_param, events_dict_all, node_mem_true, K, T_all)
    # ll_true_train, _ = MBHP.model_LL_kernel_sum_external(sim_param, events_dict_train, node_mem_true, K, T_train)
    # ll_true_all, _ = MBHP.model_LL_kernel_sum_external(sim_param, events_dict_all, node_mem_true, K, T_all)
    # print(f"LL(true param): train={ll_true_train/n_event_train:.3f}, all={ll_true_all/n_events_all:.3f},"
    #     f" test={(ll_true_all-ll_true_train)/(n_events_all-n_event_train):.3f}")
    # agg_adj = event_dict_to_aggregated_adjacency(N, events_dict_train)
    # MBHP.plot_adj(agg_adj, node_mem_true, K, "True membership")
    #
    # for K_test in range(3, 4):
    #     print(f"\nfitting test at K={K_test}")
    #     # compute node membership using spectral clustering
    #     node_mem_spectral = spectral_cluster1(agg_adj, K_test, n_kmeans_init=500, normalize_z=True, multiply_s=True)
    #     MBHP.plot_adj(agg_adj, node_mem_spectral, K_test, f"Spectral membership K={K_test}")
    #     rand_i = adjusted_rand_score(node_mem_true, node_mem_spectral)
    #     print(f"\tRI score : {rand_i:.3f}")
    #     ## # fit single beta full model at K_test classes
    #     # fit_param, ll_train, _ = MBHP.model_fit_single_beta(n_alpha, events_dict_train, node_mem_spectral, K_test, T_train, beta)
    #     # ll_all, _ = MBHP.model_LL_single_beta_external(fit_param, events_dict_all, node_mem_spectral, K_test, T_all)
    #     ## # fit sum of kernels full model at K_test classes
    #     fit_param, ll_train, _ = MBHP.model_fit_kernel_sum(n_alpha, events_dict_train, node_mem_spectral, K_test, T_train, betas)
    #     ll_all, _ = MBHP.model_LL_kernel_sum_external(fit_param, events_dict_all, node_mem_spectral, K_test, T_all)
    #     print(f"\tLL(fit param): train={ll_train / n_event_train:.3f}, all={ll_all / n_events_all:.3f},"
    #           f" test={(ll_all - ll_train) / (n_events_all - n_event_train):.3f}")
    #
    #     # arrange fit_parameters before print
    #     arranged_fit_param = arrage_fit_param(fit_param, K_test, node_mem_true, node_mem_spectral, betas)
    #     MBHP.print_model_param_kernel_sum(arranged_fit_param)
    #
    #     # simulate from parameter fit and compare two datasets motif counts
    #     events_dict_sim, node_mem_sim = MBHP.simulate_sum_kernel_model(fit_param, N, K, p, T_all)
    #     n_events_sim = OneBlockFit.cal_num_events_2(events_dict_sim)
    #     print(f"events_sim={n_events_sim}, events_all={n_events_all}")
    #     recip, trans, motif_count = MBHP.cal_recip_trans_motif(events_dict_all, N, 10)
    #     recip_sim, trans_sim, motif_count_sim = MBHP.cal_recip_trans_motif(events_dict_sim, N, 10)
    #     print(f"\noriginal sim: recip={recip:.3f} trans={trans:.3f}")
    #     print(f"second sim: recip={recip_sim:.3f} trans={trans_sim:.3f}")

#%% model mismatch test
    """
    Goal: test assumption that all nodes start from same baseline intensity is causing
          model mismatch - second part will have lower adjusted_RI
    1) simulate from 6-alpha sum of kernel model 
    2) split simulated  data into 2 equal parts
    3) run SP on both parts
    Notes: only for a few simulation run second had lower adjusted_RI
    """
    # K, N, T_all = 3, 70, 1500  # number of nodes and duration
    # n_alpha = 6
    # percent = [1 / K] * K  # nodes percentage membership
    # split_ratio = 0.5
    # # simulate from 6-alpha sum of kernels model
    # sim_param = get_simulation_params(K, 1, n_alpha=n_alpha, sum=True)
    # betas = sim_param[n_alpha + 2]
    # print("Sum of kernels model at betas ", betas)
    # events_dict_all, node_mem_true = MBHP.simulate_sum_kernel_model(sim_param, N, K, percent, T_all)
    # n_events_all = OneBlockFit.cal_num_events_2(events_dict_all)
    # print(f"Simultion at K={K}, N={N}, balanced membership, split ratio={split_ratio}")
    # print(f"T_all={T_all}, events_all={n_events_all}")
    # # print("\nTrue simulation parameters:")
    # # MBHP.print_model_param_kernel_sum(sim_param)
    #
    # # split data into two
    # events_list = list(events_dict_all.values())
    # # find spliting point
    # events_array = np.sort(np.concatenate(events_list))
    # split_point = round(events_array.shape[0] * split_ratio)
    # split_time = events_array[split_point]
    # # split simulated data into 2 datasets below and above split point
    # events_dict_1 = {}
    # events_dict_2 = {}
    # for (u, v) in events_dict_all:
    #     events = np.array(events_dict_all[(u, v)])
    #     bisect_point = bisect_right(events, split_time)
    #     events1 = events[:bisect_point]
    #     events2 = events[bisect_point:] - split_time
    #     if len(events1) != 0:
    #         events_dict_1[(u, v)] = events1
    #     if len(events2) != 0:
    #         if np.any(events2<=0):
    #             print("warning :(", events2)
    #         events_dict_2[(u, v)] = events2
    # print("split dataset into 2 equal parts and scaled second part")
    # n_event_1 = OneBlockFit.cal_num_events_2(events_dict_1)
    # n_event_2 = OneBlockFit.cal_num_events_2(events_dict_2)
    # print(f"n_events_1={n_event_1} , n_event_2={n_event_2}\n")
    #
    # # run SP on full dataset then calculate RI and LL
    # agg_adj = event_dict_to_aggregated_adjacency(N, events_dict_all)
    # MBHP.plot_adj(agg_adj, node_mem_true, K, "Full dataset")
    # node_mem = spectral_cluster1(agg_adj, K, n_kmeans_init=500, normalize_z=True, multiply_s=True)
    # rand_i_full = adjusted_rand_score(node_mem_true, node_mem)
    # ll_true_all, _ = MBHP.model_LL_kernel_sum_external(sim_param, events_dict_all, node_mem_true, K, T_all)
    # print(f"Full simulation:\n\tRI score={rand_i_full:.3f}"
    #       f"\n\tLL(true param)={ll_true_all / n_events_all:.3f}")
    #
    # #First part: then fit
    # adj1 = event_dict_to_aggregated_adjacency(N, events_dict_1)
    # node_mem_1 = spectral_cluster1(adj1, K, n_kmeans_init=500, normalize_z=True, multiply_s=True)
    # MBHP.plot_adj(adj1, node_mem_1, K, "1st dataset")
    # rand_i_1 = adjusted_rand_score(node_mem_true, node_mem_1)
    # # fit_param1, ll_train1, _ = MBHP.model_fit_kernel_sum(n_alpha, events_dict_1, node_mem_1, K, split_time, betas)
    # # arranged_fit_param1 = arrage_fit_param(fit_param1, K, node_mem_true, node_mem_1, betas)
    # # MBHP.print_model_param_kernel_sum(arranged_fit_param1)
    # print(f"First part:RI score={rand_i_1:.3f}")
    #
    # adj2 = event_dict_to_aggregated_adjacency(N, events_dict_2)
    # node_mem_2 = spectral_cluster1(adj2, K, n_kmeans_init=500, normalize_z=True, multiply_s=True)
    # MBHP.plot_adj(adj2, node_mem_2, K, "2nd dataset")
    # rand_i_2 = adjusted_rand_score(node_mem_true, node_mem_2)
    # # fit_param2, ll_train2, _ = MBHP.model_fit_kernel_sum(n_alpha, events_dict_2, node_mem_2, K, T_all - split_time, betas)
    # # arranged_fit_param2 = arrage_fit_param(fit_param2, K, node_mem_true, node_mem_2, betas)
    # # MBHP.print_model_param_kernel_sum(arranged_fit_param2)
    # print(f"Second part:RI score : {rand_i_2:.3f}")
    """v"""

#%% Degree Corrected simulation test
    """
    - simulate data with degree corrected baseline intensity (used pareto distribution)
    - run spectral clustering and calculate average RI over 10 run
    Notes: decree corrected affect spectral clustering accuracy a little
    """
    # K, N, T_all = 3, 60, 1500  # number of nodes and duration
    # n_alpha = 2
    # p = [1 / K] * K  # balanced node membership
    # # # single beta model
    # sim_param = get_simulation_params(K, 1, n_alpha=2, sum=False)
    # beta = sim_param[n_alpha+1]
    # print(f"singel beta model simultion at K={K}, N={N}, balanced membership")
    # print("simulation with correction step")
    # RI_runs = 0
    # n_events_runs = 0
    # n_runs = 10
    # for run in range(n_runs):
    #     events_dict_all, node_mem_true = simulate_one_beta_model_2_corrected(sim_param, N, K, p, T_all)
    #     n_events_all = OneBlockFit.cal_num_events_2(events_dict_all)
    #     print(f"Sim: Duration={T_all}, Total Events={n_events_all}")
    #     agg_adj = event_dict_to_aggregated_adjacency(N, events_dict_all)
    #     MBHP.plot_adj(agg_adj, node_mem_true, K, "True membership")
    #     # compute node membership using spectral clustering
    #     node_mem_spectral = spectral_cluster1(agg_adj, K, n_kmeans_init=500, normalize_z=True, multiply_s=True)
    #     # MBHP.plot_adj(agg_adj, node_mem_spectral, K, f"Spectral membership K={K}")
    #     rand_i = adjusted_rand_score(node_mem_true, node_mem_spectral)
    #     print(f"adjusted RI={rand_i:.3f}")
    #     RI_runs += rand_i
    #     n_events_runs += n_events_all
    # print(f"Over {n_runs} runs average events={int(n_events_runs/n_runs)} & RI={RI_runs/n_runs:.3f}")
    """ """
#%% simulate from K=4 and fit to different range of K
    # K, N, T_all = 4, 90, 1500  # number of nodes and duration
    # n_alpha = 6
    # percent = [1 / K] * K  # nodes percentage membership
    # split_ratio = 0.8
    # # simulate from 6-alpha sum of kernels model
    # sim_param = get_simulation_params(K, level=1, n_alpha=n_alpha, sum=True)
    # betas = sim_param[-1]
    # print("Sum of kernels model at betas ", betas)
    # events_dict_all, node_mem_true = MBHP.simulate_sum_kernel_model(sim_param, N, K, percent, T_all)
    # n_events_all = OneBlockFit.cal_num_events_2(events_dict_all)
    # print(f"Simultion at K={K}, N={N}, balanced membership, split ratio={split_ratio}")
    # print(f"T_all={T_all}, events_all={n_events_all}")
    #
    # # split into train and all
    # events_dict_train, T_train = MBHP.split_train(events_dict_all, split_ratio = split_ratio)
    # n_event_train = OneBlockFit.cal_num_events_2(events_dict_train)
    # print(f"T_train={T_train:.2f}, train_events={n_event_train}")
    # ll_true_all, _ = MBHP.model_LL_kernel_sum_external(sim_param, events_dict_all, node_mem_true, K, T_all)
    # ll_true_train, _ = MBHP.model_LL_kernel_sum_external(sim_param, events_dict_train, node_mem_true, K, T_train)
    # print(f"LL(true): train={ll_true_train/n_event_train:.3f}, all={ll_true_all/n_events_all:.3f}")
    #
    # agg_adj = event_dict_to_aggregated_adjacency(N, events_dict_train)
    # MBHP.plot_adj(agg_adj, node_mem_true, K, "True membership - train")
    #
    # # loop
    # for K_test in range(1, 21):
    #     # print(f"\nfitting test at K={K_test}")
    #     # compute node membership using spectral clustering
    #     node_mem_spectral = spectral_cluster1(agg_adj, K_test, n_kmeans_init=500, normalize_z=True, multiply_s=True)
    #     # plot_adj(agg_adj, node_mem_spectral, K_test, f"Spectral membership K={K_test}")
    #     rand_i = adjusted_rand_score(node_mem_true, node_mem_spectral)
    #     # print(f"\tRI score : {rand_i:.3f}")
    #     ## # fit sum of kernels 6-alpha model at K_test classes
    #     fit_param, ll_train, _ = MBHP.model_fit_kernel_sum(n_alpha, events_dict_train, node_mem_spectral, K_test, T_train, betas)
    #     ll_all, _ = MBHP.model_LL_kernel_sum_external(fit_param, events_dict_all, node_mem_spectral, K_test, T_all)
    #     # print(f"\tLL(fit): train={ll_train / n_event_train:.3f}, all={ll_all / n_events_all:.3f},"
    #     #       f" test={(ll_all - ll_train) / (n_events_all - n_event_train):.3f}")
    #     print(f"{rand_i:.3f}\t{ll_train / n_event_train:.3f}\t{ll_all / n_events_all:.3f}\t"
    #           f"{(ll_all - ll_train) / (n_events_all - n_event_train):.3f}")

#%% spectral clustering tests on 6-alpha model while increasing T, N
    if SP_RI_test:
        n_run = 10
        K, n_alpha = 4, 6
        percent = [1 / K] * K  # nodes percentage membership
        sim_param = get_simulation_params(K, level=1000, n_alpha=n_alpha, sum=True)
        save = True

        N_range = np.arange(40, 101, 15)
        T_range = np.arange(600, 1401, 200)
        RI = np.zeros((len(N_range), len(T_range)))  # hold RI scores while varying n_nodes & duration
        n_events_matrix = np.zeros((len(N_range), len(T_range)))  # hold simulated n_events while varying n_nodes & duration
        for T_idx, T in enumerate(T_range):
            for N_idx, N in enumerate(N_range):
                print("\nduration: ", T, "n_nodes: ", N)
                RI_avg = 0
                n_events_avg = 0
                for it in range(n_run):
                    events_dict, node_mem_true = MBHP.simulate_sum_kernel_model(sim_param, N, K, percent, T)
                    n_events = OneBlockFit.cal_num_events_2(events_dict)
                    agg_adj = event_dict_to_aggregated_adjacency(N, events_dict)
                    if it == 0:
                        MBHP.plot_adj(agg_adj, node_mem_true, K, f"N={N}, T={T}")
                    node_mem_spectral = spectral_cluster1(agg_adj, K, n_kmeans_init=500, normalize_z=True, multiply_s=True)
                    rand_i = adjusted_rand_score(node_mem_true, node_mem_spectral)
                    print(f"\titer# {it}: RI={rand_i:.3f}, #events={n_events}")
                    RI_avg += rand_i
                    n_events_avg += n_events
                # average over runs
                RI_avg = RI_avg/n_run
                n_events_avg = n_events_avg/n_run
                print("\tIteration average: ", RI_avg)
                RI[N_idx, T_idx] = RI_avg
                n_events_matrix[N_idx, T_idx] = n_events_avg
        print(n_events_matrix)
        if save:
            results_dict = {}
            results_dict["sim_param"] = sim_param
            results_dict["RI"] = RI
            results_dict["n_events_matrix"] = n_events_matrix
            results_dict["N_range"] = N_range
            results_dict["T_range"] = T_range
            results_dict["n_run"] = n_run
            with open(f"spectral_clustering_RI_K={K}_last.p", 'wb') as fil:
                pickle.dump(results_dict, fil)
#%% MSE parameter accuracy estimeation
    if MSE_test:
        K = 2
        n_alpha = 6
        percent = [1 / K] * K  # nodes percentage membership
        sim_param = get_simulation_params(K, level=1000, n_alpha=n_alpha, sum=True)
        betas = sim_param[-1]
        save = True

        N_range = np.arange(40, 101, 15) # np.arange(40, 101, 15) np.array([70])
        T_range = np.array([1000]) # np.arange(600, 1401, 200) np.array([1000])
        mMSE_mu = np.zeros((len(N_range), len(T_range)))
        mMSE_alpha_n = np.zeros((len(N_range), len(T_range)))
        mMSE_alpha_r = np.zeros((len(N_range), len(T_range)))
        mMSE_alpha_br = np.zeros((len(N_range), len(T_range)))
        mMSE_alpha_gr = np.zeros((len(N_range), len(T_range)))
        mMSE_alpha_al = np.zeros((len(N_range), len(T_range)))
        mMSE_alpha_alr = np.zeros((len(N_range), len(T_range)))
        mMSE_C = np.zeros((len(N_range), len(T_range)))
        sMSE_mu = np.zeros((len(N_range), len(T_range)))
        sMSE_alpha_n = np.zeros((len(N_range), len(T_range)))
        sMSE_alpha_r = np.zeros((len(N_range), len(T_range)))
        sMSE_alpha_br = np.zeros((len(N_range), len(T_range)))
        sMSE_alpha_gr = np.zeros((len(N_range), len(T_range)))
        sMSE_alpha_al = np.zeros((len(N_range), len(T_range)))
        sMSE_alpha_alr = np.zeros((len(N_range), len(T_range)))
        sMSE_C = np.zeros((len(N_range), len(T_range)))
        n_run = 10
        for T_idx, T in enumerate(T_range):
            for N_idx, N in enumerate(N_range):
                print("\nduration: ", T, "n_nodes: ", N)
                MSE_mu = np.zeros(n_run)
                MSE_alpha_n = np.zeros(n_run)
                MSE_alpha_r = np.zeros(n_run)
                MSE_alpha_br = np.zeros(n_run)
                MSE_alpha_gr = np.zeros(n_run)
                MSE_alpha_al = np.zeros(n_run)
                MSE_alpha_alr = np.zeros(n_run)
                MSE_C = np.zeros(n_run)
                for it in range(n_run):
                    events_dict, node_mem_true = MBHP.simulate_sum_kernel_model(sim_param, N, K, percent, T)
                    n_events = OneBlockFit.cal_num_events_2(events_dict)
                    agg_adj = event_dict_to_aggregated_adjacency(N, events_dict)
                    if it == 0:
                        MBHP.plot_adj(agg_adj, node_mem_true, K, f"N={N}, T={T}")
                    node_mem_spectral = spectral_cluster1(agg_adj, K, n_kmeans_init=500, normalize_z=True, multiply_s=True)
                    rand_i = adjusted_rand_score(node_mem_true, node_mem_spectral)
                    print(f"\titer# {it}: RI={rand_i:.3f}, #events={n_events}")
                    fit_param, ll_train, _ = MBHP.model_fit_kernel_sum(n_alpha, events_dict, node_mem_spectral, K, T,
                                                                       betas)
                    MSE_mu[it] = np.sum(np.square(sim_param[0]-fit_param[0]))
                    MSE_alpha_n[it] = np.sum(np.square(sim_param[1]-fit_param[1]))
                    MSE_alpha_r[it] = np.sum(np.square(sim_param[2]-fit_param[2]))
                    MSE_alpha_br[it] = np.sum(np.square(sim_param[3]-fit_param[3]))
                    MSE_alpha_gr[it] = np.sum(np.square(sim_param[4]-fit_param[4]))
                    MSE_alpha_al[it] = np.sum(np.square(sim_param[5]-fit_param[5]))
                    MSE_alpha_alr[it] = np.sum(np.square(sim_param[6]-fit_param[6]))
                    MSE_C[it] = np.sum(np.square(sim_param[7][:,:,:-1]-fit_param[7][:,:,:-1]))
                    print(f"\t\t{MSE_mu[it]:.4f}, {MSE_alpha_r[it]:.4f}, {MSE_alpha_br[it]:.4f}, {MSE_C[it]:.4f}")
                mMSE_mu[N_idx, T_idx] = np.mean(MSE_mu)
                mMSE_alpha_n[N_idx, T_idx] = np.mean(MSE_alpha_n)
                mMSE_alpha_r[N_idx, T_idx] = np.mean(MSE_alpha_r)
                mMSE_alpha_br[N_idx, T_idx] = np.mean(MSE_alpha_br)
                mMSE_alpha_gr[N_idx, T_idx] = np.mean(MSE_alpha_gr)
                mMSE_alpha_al[N_idx, T_idx] = np.mean(MSE_alpha_al)
                mMSE_alpha_alr[N_idx, T_idx] = np.mean(MSE_alpha_alr)
                mMSE_C[N_idx, T_idx] = np.mean(MSE_C)
                sMSE_mu[N_idx, T_idx] = np.mean(MSE_mu)
                sMSE_alpha_n[N_idx, T_idx] = np.std(MSE_alpha_n)
                sMSE_alpha_r[N_idx, T_idx] = np.std(MSE_alpha_r)
                sMSE_alpha_br[N_idx, T_idx] = np.std(MSE_alpha_br)
                sMSE_alpha_gr[N_idx, T_idx] = np.std(MSE_alpha_gr)
                sMSE_alpha_al[N_idx, T_idx] = np.std(MSE_alpha_al)
                sMSE_alpha_alr[N_idx, T_idx] = np.std(MSE_alpha_alr)
                sMSE_C[N_idx, T_idx] = np.std(MSE_C)
                print(f"Average: {mMSE_mu[N_idx, T_idx]:.4f}, {mMSE_alpha_r[N_idx, T_idx]:.4f},"
                      f"{mMSE_alpha_br[N_idx, T_idx]:.4f}, {mMSE_C[N_idx, T_idx]:.4f}")
        if save:
            results_dict = {}
            results_dict["sim_param"] = sim_param
            results_dict["MSE_mean"] = (mMSE_mu, mMSE_alpha_n, mMSE_alpha_r, mMSE_alpha_br, mMSE_alpha_gr, mMSE_alpha_al
                                        , mMSE_alpha_alr, mMSE_C)
            results_dict["MSE_std"] = (sMSE_mu, sMSE_alpha_n, sMSE_alpha_r, sMSE_alpha_br, sMSE_alpha_gr, sMSE_alpha_al
                                        , sMSE_alpha_alr, sMSE_C)
            results_dict["N_range"] = N_range
            results_dict["T_range"] = T_range
            results_dict["n_run"] = n_run
            with open(f"parameter_accuracy_K{K}_T1000_last.p", 'wb') as fil:
                pickle.dump(results_dict, fil)
#%% rho restriction test
    if rho_test:
        N, K, T = 80, 3, 2000
        n_alpha = 2
        percent = [1 / K] * K  # nodes percentage membership
        sim_param = get_simulation_params(K, level=1, n_alpha=n_alpha, sum=False)
        print("simulation parameters")
        MBHP.print_model_param_single_beta(sim_param)
        beta = sim_param[-1]
        events_dict, node_mem_true = MBHP.simulate_one_beta_model(sim_param, N, K, percent, T)
        n_events = OneBlockFit.cal_num_events_2(events_dict)
        agg_adj = event_dict_to_aggregated_adjacency(N, events_dict)
        MBHP.plot_adj(agg_adj, node_mem_true, K, f"N={N}, T={T}")
        print("\nfit method 1")
        fit_param, ll_train, n_ev = MBHP.model_fit_single_beta(n_alpha, events_dict, node_mem_true, K, T, beta)
        print("estimated parameters")
        MBHP.print_model_param_single_beta(fit_param)
        print(f"LL={ll_train:.1f}, n_ev={n_ev}")
        print("\nfit method 2")
        fit_param_rho, ll_train_rho, n_ev = MBHP.fit_2_alpha_rho_single_beta(events_dict, node_mem_true, K, T, beta)
        print("estimated parameters")
        MBHP.print_model_param_single_beta(fit_param_rho)
        print(f"LL={ll_train_rho:.1f}, n_ev={n_ev}")


