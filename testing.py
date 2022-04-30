import numpy as np
import pickle
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from sklearn import metrics
sys.path.append("./CHIP-Network-Model")
from dataset_utils import load_and_combine_nodes_for_test_train
from utils_sum_betas_bp import cal_num_events


        
def load_data_train_all(dnx_pickle_file_name, split_ratio=0.8):
    incident_dnx_list = pickle.load(open(dnx_pickle_file_name, 'rb'))
    digraph1 = incident_dnx_list[0]
    scale = 1000

    small_comp_countries_train = ['GUA', 'BLZ', 'GAM', 'SEN', 'SAF', 'LES', 'SWA', 'MZM', 'GNB']

    # print("n_events before removing small components: ", len(digraph1.edges()))
    nodes_before = set(digraph1.nodes())
    for country in small_comp_countries_train:
        for node in nodes_before:
            digraph1.remove_edge(country, node)
            digraph1.remove_edge(node, country)

    # find train splitting point
    n_events_all = len(digraph1.edges())
    split_point = int(n_events_all * split_ratio)
    timestamp_last_train = digraph1.edges()[split_point - 1][2]  # time of last event included in train dataset
    timestamp_last_all = digraph1.edges()[-1][2]  # time of last event in all dataset
    timestamp_first = digraph1.edges()[0][2]
    n_events_train = len(digraph1.edges(end=timestamp_last_train))
    duration = timestamp_last_all - timestamp_first
    print("duration = ", int(duration/(60*60*24)), " days")

    # get train and all nodes id map
    node_set_all = set(digraph1.nodes(end=timestamp_last_all))
    node_set_train = set(digraph1.nodes(end=timestamp_last_train))
    nodes_not_in_train = node_set_all - node_set_train # {'SSD', 'PAN'}
    print("nodes not in train ", nodes_not_in_train)
    n_nodes_train = len(node_set_train)
    print("#nodes train = ", n_nodes_train)
    node_id_map_train, id_node_map_train = get_node_id_maps(node_set_train)

    # create event dictionary of train and all dataset
    data_all = []
    data_train = []
    data_test = []
    for edge in digraph1.edges():
        # ignore nodes not in train
        if edge[0] in node_id_map_train and edge[1] in node_id_map_train:
            sender_id, receiver_id = int(node_id_map_train[edge[0]]), int(node_id_map_train[edge[1]])
            timestamp = (edge[2] - timestamp_first) / duration * scale
            if timestamp < 0:
                print(edge)
            data_all.append([sender_id, receiver_id, timestamp])
            if edge[2] <= timestamp_last_train:
                data_train.append([sender_id, receiver_id, timestamp])
            else:
                data_test.append([sender_id, receiver_id, timestamp])
    # data_train = np.array(data_train)
    # data_test = np.array(data_test)
    # data_all = np.array(data_all)
    T_all = (timestamp_last_all - timestamp_first) / duration * scale
    T_train = (timestamp_last_train - timestamp_first) / duration * scale

    n_events_all = len(data_all)
    n_events_train = len(data_train)
    tuple_train = data_train, T_train, n_nodes_train, n_events_train, id_node_map_train
    tuple_all = data_all, T_all, n_nodes_train, n_events_all, id_node_map_train
    return tuple_train, tuple_all, data_test

def get_node_id_maps(node_set):
    nodes = list(node_set)
    nodes.sort()
    node_id_map = {}
    id_node_map = {}
    for i, n in enumerate(nodes):
        node_id_map[n] = int(i)
        id_node_map[i] = n
    return node_id_map, id_node_map




# # ROC curves
# results_path = "/shared/Results/MultiBlockHawkesModel/LSH_tests/BHM"
# datasets = ["RealityMining", "MID", "Enron-2", "Enron-15", "fb-forum"]
# Ks = [50, 95, 16, 14, 57]
# auc_dict = {}
# for dataset, K in zip(datasets, Ks):
#     with open(f'{results_path}/{dataset}_auc_K_{K}.p', 'rb') as file:
#         result_dict = pickle.load(file)
#     auc_dict[dataset] = result_dict["auc"]
#     print(dataset, result_dict["avg"])
#     y_runs = result_dict["y__runs"]
#     pred_runs = result_dict["pred_runs"]
#     fig, ax = plt.subplots(figsize=(5, 4))
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     for r in range(100):
#         fpr, tpr, thresholds = metrics.roc_curve(y_runs[:,:,r].flatten(), pred_runs[:,:,r].flatten(), pos_label=1)
#         plt.plot(fpr, tpr, color='darkorange', lw=2)
#         plt.xlabel('False Positive Rate', fontsize=12)
#         plt.ylabel('True Positive Rate', fontsize=12)
#         # plt.title('Receiver operating characteristic')
#         # plt.legend(loc="lower right")
#         plt.tight_layout()
#     plt.show()
#     fig.savefig(f"{results_path}/{dataset}_BHM_ROC.pdf")






