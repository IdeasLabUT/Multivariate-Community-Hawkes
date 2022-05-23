import random
import numpy as np
from hawkes.MHP import MHP
import utils_generate_model as gen


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
                    events_dict = gen.simulate_one_beta_dia_2(par, list(class_nodes_list[i]), duration)
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
                d_ab, d_ba = gen.simulate_one_beta_off_2(par_ab, par_ba, list(class_nodes_list[i]), list(class_nodes_list[j]),
                                                                     duration)
                events_dict_all.update(d_ab)
                events_dict_all.update(d_ba)
    return events_dict_all, node_mem_actual

def simulate_one_beta_dia(par, a_nodes, end_time, return_list=False):
    if len(par) == 4:
        mu_array, alpha_matrix = gen.get_mu_array_alpha_matrix_dia_bp(par[0], par[1:3], len(a_nodes))
    elif len(par) == 6:
        mu_array, alpha_matrix = gen.get_mu_array_alpha_matrix_dia_bp(par[0], par[1:5], len(a_nodes))
    elif len(par) == 8:
        mu_array, alpha_matrix = gen.get_mu_array_alpha_matrix_dia_bp(par[0], par[1:7], len(a_nodes))
    # multivariate hawkes process object [NOTE: alpha=jump_size/beta, omega=beta]
    P = MHP(mu=mu_array, alpha=alpha_matrix, omega=par[-1])
    P.generate_seq(end_time)
    # assume that timestamps list is ordered ascending with respect to u then v [(0,1), (0,2), .., (1,0), (1,2), ...]
    events_list = []
    for m in range(len(mu_array)):
        i = np.nonzero(P.data[:, 1] == m)[0]
        events_list.append(P.data[i, 0])
    events_dict = gen.events_list_to_events_dict_remove_empty_np(events_list, a_nodes)
    if return_list:
        return events_list, events_dict
    return events_dict

def simulate_one_beta_off(par_ab, par_ba, a_nodes, b_nodes, end_time, return_list=False):
    if len(par_ab) == 4:
        mu_array, alpha_matrix = gen.get_mu_array_alpha_matrix_off_bp(par_ab[0], par_ab[1:3], par_ba[0], par_ba[1:3],
                                                                  len(a_nodes), len(b_nodes))
    elif len(par_ab) == 6:
        mu_array, alpha_matrix = gen.get_mu_array_alpha_matrix_off_bp(par_ab[0], par_ab[1:5], par_ba[0], par_ba[1:5],
                                                                  len(a_nodes), len(b_nodes))
    else:
        mu_array, alpha_matrix = gen.get_mu_array_alpha_matrix_off_bp(par_ab[0], par_ab[1:7], par_ba[0], par_ba[1:7],
                                                                  len(a_nodes), len(b_nodes))
    # multivariate hawkes process object [NOTE: alpha=jump_size/beta, omega=beta]
    P = MHP(mu=mu_array, alpha=alpha_matrix, omega=par_ab[-1])
    P.generate_seq(end_time)
    # assume that timestamps list is ordered ascending with respect to u then v [(0,1), (0,2), .., (1,0), (1,2), ...]
    events_list = []
    for m in range(len(mu_array)):
        i = np.nonzero(P.data[:, 1] == m)[0]
        events_list.append(P.data[i, 0])
    M = len(a_nodes) * len(b_nodes)
    events_list_ab = events_list[:M]
    events_list_ba = events_list[M:]
    events_dict_ab = gen.events_list_to_events_dict_remove_empty_np_off(events_list_ab, a_nodes, b_nodes)
    events_dict_ba = gen.events_list_to_events_dict_remove_empty_np_off(events_list_ba, b_nodes, a_nodes)
    if return_list:
        return events_list, events_dict_ab, events_dict_ba
    return events_dict_ab, events_dict_ba


