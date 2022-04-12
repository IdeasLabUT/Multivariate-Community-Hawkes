# -*- coding: utf-8 -*-
"""
Created on Wed May 19 13:39:53 2021

@author: kevin
"""
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from generative_model_utils import event_dict_to_aggregated_adjacency, event_dict_to_adjacency
from spectral_clustering import spectral_cluster
from MultiBlockFit import fit_kernel_sum_model, fit_n_r_br_gr_fixed_beta_model, print_paramters_kernel

n_classes = 6
timestamp_max = 1000
kernel = "single"
# kernel = "sum"
beta = 1
betas = np.array([0.02, 0.2, 2, 20])


def create_incident_event_dict(dnx_pickle_file_name, timestamp_max=1000):
    incident_dnx = pickle.load(open(dnx_pickle_file_name, 'rb'))
    event_dict = {}

    timestamp_first = incident_dnx.edges()[0][2]
    timestamp_last = incident_dnx.edges()[-1][2]
    duration = timestamp_last - timestamp_first

    if timestamp_max is None:
        timestamp_max = timestamp_last - timestamp_first

    node_set = set(incident_dnx.nodes())
    node_id_map, id_node_map = get_node_id_maps(node_set)

    for edge in incident_dnx.edges():
        sender_id = node_id_map[edge[0]]
        receiver_id = node_id_map[edge[1]]
        timestamp = (edge[2] - timestamp_first) / duration * timestamp_max
        if timestamp < 0:
            print(edge)

        if (sender_id, receiver_id) not in event_dict:
            event_dict[(sender_id, receiver_id)] = []

        event_dict[(sender_id, receiver_id)].append(timestamp)

    return event_dict, len(incident_dnx.nodes()), duration, id_node_map


def get_node_id_maps(node_set):
    nodes = list(node_set)
    nodes.sort()

    node_id_map = {}
    id_node_map = {}
    for i, n in enumerate(nodes):
        node_id_map[n] = i
        id_node_map[i] = n

    return node_id_map, id_node_map


#%% Load MID incident data and fit multivariate block Hawkes model
if __name__ == "__main__":
    dnx_pickle_file_name = 'incident_impulse.pckl'
    events_dict, n_nodes, duration, id_node_map = create_incident_event_dict(
        dnx_pickle_file_name, timestamp_max)

    start_fit_time = time.time()
    agg_adj = event_dict_to_aggregated_adjacency(n_nodes, events_dict)
    unweight_adj = event_dict_to_adjacency(n_nodes, events_dict)
    node_membership = spectral_cluster(agg_adj,
                                       n_classes,
                                       verbose=False,
                                       plot_eigenvalues=False)

    #%% Analyze blocks
    print(np.histogram(node_membership, bins=n_classes))
    for i in range(n_classes):
        print(f"Class {i}")
        nodes_in_class_i = np.where(node_membership == i)[0]
        for id in nodes_in_class_i:
            print(id_node_map[id], end=' ')
        print()

    #%% Fit multiblock Hawkes using sum of kernels
    if kernel == "sum":
        params_est, ll, n_events = fit_kernel_sum_model(
            events_dict, node_membership, n_classes, timestamp_max, betas)
        mu_bp, alpha_n_bp, alpha_r_bp, alpha_br_bp, alpha_gr_bp, C_bp, betas = params_est
        end_fit_time = time.time()
        print(f'Fit time: {(end_fit_time - start_fit_time) / 60} min')

        scaling_betas = np.sum(C_bp / betas, 2)

        #print_paramters_6(params_est)

        #%% Plot parameters
        plt.figure()
        plt.imshow(mu_bp)
        plt.colorbar()
        plt.title("mu")

        plt.figure()
        plt.imshow(alpha_n_bp * scaling_betas)
        plt.colorbar()
        plt.title("alpha_n")

        plt.figure()
        plt.imshow(alpha_r_bp * scaling_betas)
        plt.colorbar()
        plt.title("alpha_r")

        plt.figure()
        plt.imshow(alpha_br_bp * scaling_betas)
        plt.colorbar()
        plt.title("alpha_br")

        plt.figure()
        plt.imshow(alpha_gr_bp * scaling_betas)
        plt.colorbar()
        plt.title("alpha_gr")

    #%% Fit multiblock Hawkes using fixed beta
    if kernel == "single":
        beta_bp = np.ones((n_classes, n_classes)) * beta
        params_est, ll, n_events = fit_n_r_br_gr_fixed_beta_model(
            events_dict, node_membership, n_classes, timestamp_max, beta_bp)
        end_fit_time = time.time()
        #print_paramters_6(params_est)

        #%% Plot parameters
        plt.figure()
        plt.imshow(params_est[0])
        plt.colorbar()
        plt.title("mu")

        plt.figure()
        plt.imshow(params_est[1])
        plt.colorbar()
        plt.title("alpha_n")

        plt.figure()
        plt.imshow(params_est[2])
        plt.colorbar()
        plt.title("alpha_r")

        plt.figure()
        plt.imshow(params_est[3])
        plt.colorbar()
        plt.title("alpha_br")

        plt.figure()
        plt.imshow(params_est[4])
        plt.colorbar()
        plt.title("alpha_gr")
