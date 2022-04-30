# TODO one beta has no refinement <-- maybe remove
# TODO SUBMISSION: remove docker and all saving or read saved options
# TODO maybe change MID formate to be csv file
# TODO remove saved motif of dataset

import numpy as np
import pickle
import matplotlib.pyplot as plt

import utils_accuracy_tests as accuracy_test
from utils_fit_one_beta_model import model_fit_cal_log_likelihood_one_beta
from refinement_alg import model_fit_cal_log_likelihood_sum_betas
from utils_sum_betas_bp import cal_num_events
from utils_fit_sum_betas_model import get_node_id_maps
from Read_results import analyze_block
import networkx as nx
import os



#%% helper functions
def load_data_train_all(dnx_pickle_file_name, split_ratio=0.8, scale=1000, remove_small_comp=True):
    incident_dnx_list = pickle.load(open(dnx_pickle_file_name, 'rb'))
    digraph1 = incident_dnx_list[0]

    small_comp_countries_train = ['GUA', 'BLZ', 'GAM', 'SEN', 'SAF', 'LES', 'SWA', 'MZM', 'GNB']
    small_comp_countres_full = ['BLZ', 'GUA', 'MZM', 'SWA', 'SAF', 'LES']

    if remove_small_comp:
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
    # print("duration = ", int(duration/(60*60*24)), " days")

    # get train and all nodes id map
    node_set_all = set(digraph1.nodes(end=timestamp_last_all))
    n_nodes_all = len(node_set_all)
    node_id_map_all, id_node_map_all = get_node_id_maps(node_set_all)
    node_set_train = set(digraph1.nodes(end=timestamp_last_train))
    n_nodes_train = len(node_set_train)
    node_id_map_train, id_node_map_train = get_node_id_maps(node_set_train)

    # create event dictionary of train and all dataset
    event_dict_all = {}
    event_dict_train = {}
    for edge in digraph1.edges():
        sender_id, receiver_id = node_id_map_all[edge[0]], node_id_map_all[edge[1]]
        if scale == 1000: # scale timestamp in range [0 : 1000]
            timestamp = (edge[2] - timestamp_first) / duration * scale
        else:
            timestamp = (edge[2] - timestamp_first) / scale
        if timestamp < 0:
            print(edge)
        if (sender_id, receiver_id) not in event_dict_all:
            event_dict_all[(sender_id, receiver_id)] = []
        event_dict_all[(sender_id, receiver_id)].append(timestamp)
        if edge[2] <= timestamp_last_train:
            sender_id_t, receiver_id_t = node_id_map_train[edge[0]], node_id_map_train[edge[1]]
            if (sender_id_t, receiver_id_t) not in event_dict_train:
                event_dict_train[(sender_id_t, receiver_id_t)] = []
            event_dict_train[(sender_id_t, receiver_id_t)].append(timestamp)
    # train and all end time
    if scale == 1000:
        T_all = (timestamp_last_all - timestamp_first) / duration * scale
        T_train = (timestamp_last_train - timestamp_first) / duration * scale
    else:
        T_all = (timestamp_last_all - timestamp_first) / scale
        T_train = (timestamp_last_train - timestamp_first) / scale
    # node not in train list
    nodes_not_in_train = []
    for n in (node_set_all - node_set_train):
        nodes_not_in_train.append(node_id_map_all[n])

    tuple_train = event_dict_train, T_train, n_nodes_train, n_events_train, id_node_map_train
    tuple_all = event_dict_all, T_all, n_nodes_all, n_events_all, id_node_map_all
    return tuple_train, tuple_all, nodes_not_in_train


def get_list_counties_small_comp(adj, id_node_map):
    net = nx.DiGraph(adj)
    components = nx.weakly_connected_components(net)
    nodes_small_comp = []
    countries_small_comp = []
    for nodes_com in components:
        # small component
        if len(nodes_com) < 3:
            for n in nodes_com:
                nodes_small_comp.append(n)
                countries_small_comp.append(id_node_map[n])
    # remove countries in small components
    net.remove_nodes_from(nodes_small_comp)
    # check if removing nodes caused other small components
    components = nx.weakly_connected_components(net)
    for nodes_com in components:
        print(nodes_com)
        if len(nodes_com) < 3:
            for n in nodes_com:
                print(f"Disconnected node: id:{n} -> country:{id_node_map_train[n]}")
    return countries_small_comp


#%% Load MID incident data and fit multivariate block Hawkes model
if __name__ == "__main__":
    docker = False
    if docker:
        save_path = f"/data"  # when called from docker
    else:
        save_path = f'/shared/Results/MultiBlockHawkesModel'

    """ sum of kernel refinement fitting """
    sum_betas_K_range = [4]  # ex: list(range(1,11))
    n_alpha = 6
    fit_model = True  # either fit mulch or read saved fit
    save_fit = True  # save fitted model - specify path
    REF_ITER = 3  # maximum refinement interation - set to 0 for no refinement
    betas_recip = np.array([2 * 30, 2 * 7, 1 / 2]) * (1000 / 8380)  # [2 month, 2 week, 12 hours]
    betas = np.reciprocal(betas_recip)

    # OTHER betas options
    # betas_recip = np.array([4*30, 30, 1]) * (1000 / 8380)  # [4 month, 1 month, 1 day]
    # betas_recip = np.array([2*30, 7, 1/12]) * (1000 / 8380)  # [2 month, 1week, 2 hour]
    # betas_recip = np.array([30, 7, 1]) * (1000 / 8380)  # [1 month, 1 week, 1 day]

    """ motif simulation """
    motif_experiment = False
    n_motif_simulations = 5
    save_motif = True  # save simulation motif counts - specify path

    """ link prediction """
    link_prediction_experiment = True
    save_link = True  # specify path in code

    """ single beta"""
    block_range = []  # ex: list(range(1,11))




    print("Load MID dataset - timestampes scaled [0:1000]")
    print("Added random Gaussian noise (mean=0, std=1hour) to events with same timestamp")
    pickle_file = os.path.join(os.getcwd(), "storage", "datasets", "MID", "MID_std1hour.p")
    train_tup, all_tup, nodes_not_in_train = load_data_train_all(pickle_file)
    events_dict_train, T_train, n_nodes_train, n_events_train, id_node_map_train = train_tup
    events_dict_all, T_all, n_nodes_all, n_events_all, id_node_map_all = all_tup
    dataset = "MID"
    link_pred_delta = 7.15
    motif_delta_month = 4



#%% refinement - implemented for sum of kernel versions
    if len(sum_betas_K_range) !=0:
        print(f"Fit MID using {n_alpha}-alpha (sum of kernel) at betas={betas}")
        train_tup = events_dict_train, n_nodes_train, T_train
        all_tup = events_dict_all, n_nodes_all, T_all
    for K in sum_betas_K_range:
        if fit_model:
            print("\nfit MULCH + refinement at K=", K)
            fit_dict = model_fit_cal_log_likelihood_sum_betas(train_tup, all_tup, nodes_not_in_train, n_alpha, K, betas,
                                                              REF_ITER, verbose=True)
            # NOTE: if max_iter=0 no refinement tuple is returned
            print(f"spectral log-likelihood:\ttrain={fit_dict['ll_train_sp']:.3f}\tall={fit_dict['ll_all_sp']:.3f}"
                  f"\ttest={fit_dict['ll_test_sp']:.3f}")
            print(f"refine log-likelihood:  \ttrain={fit_dict['ll_train_ref']:.3f}\tall={fit_dict['ll_all_ref']:.3f}"
                  f"\ttest={fit_dict['ll_test_ref']:.3f}")

            print("\nAnalyze countries per block membership")
            analyze_block(fit_dict["node_mem_train_ref"], K, id_node_map_train)

            if save_fit:
                fit_dict["id_node_map_train"] = id_node_map_train
                fit_dict["id_node_map_all"] = id_node_map_all
                full_fit_path = f'{save_path}/MID/test'
                pickle_file_name = f"{full_fit_path}/k_{K}.p"
                with open(pickle_file_name, 'wb') as f:
                    pickle.dump(fit_dict, f)

        # read saved fit
        else:
            full_fit_path = f"{save_path}/MID/6alpha_KernelSum_Ref_batch/2month1week2hour"
            with open(f"{full_fit_path}/k_{K}.p", 'rb') as f:
                fit_dict = pickle.load(f)
            print(f"spectral log-likelihood:\ttrain={fit_dict['ll_train_sp']:.3f}\tall={fit_dict['ll_all_sp']:.3f}"
                  f"\ttest={fit_dict['ll_test_sp']:.3f}")
            print(f"refine log-likelihood:  \ttrain={fit_dict['ll_train_ref']:.3f}\tall={fit_dict['ll_all_ref']:.3f}"
                  f"\ttest={fit_dict['ll_test_ref']:.3f}")

        # Simulation and motif experiments
        if motif_experiment:

            # refinement parameters
            fit_param_ref = fit_dict["fit_param_ref"]
            nodes_mem_train_ref = fit_dict["node_mem_train_ref"]

            # ---> Either run motif counts on dataset
            # dataset_recip, dataset_trans, dataset_motif_month = accuracy_test.cal_recip_trans_motif(events_dict_train,
            #                                                                                         n_nodes_train,
            #                                                                                         motif_delta_month,
            #                                                                                         verbose=False)
            # dataset_n_events_train = cal_num_events(events_dict_train)
            # dataset_motif_tup = (dataset_recip, dataset_trans, dataset_motif_month, dataset_n_events_train)

            # ---> OR read networks recip, trans, motifs count from saved pickle
            with open(f"storage/datasets_motif_counts/month_MID_counts.p", 'rb') as f:
                dataset_motif_dict = pickle.load(f)
            dataset_motif_month = dataset_motif_dict["dataset_motif"]
            recip = dataset_motif_dict["dataset_recip"]
            trans = dataset_motif_dict["dataset_trans"]
            n_events_train = dataset_motif_dict["dataset_n_events"]
            print(f"{dataset}: reciprocity={recip:.4f}, transitivity={trans:.4f}, #events:{n_events_train}")
            dataset_motif_tup = (recip, trans, dataset_motif_month, n_events_train)

            # run simulation and count motifs
            motif_test_dict = accuracy_test.simulate_count_motif_experiment(dataset_motif_tup, fit_param_ref,
                                                                            nodes_mem_train_ref, K, T_train,
                                                                            motif_delta_month,
                                                                            n_sim=n_motif_simulations,
                                                                            verbose=True)
            if save_motif:
                full_motif_path = f"{save_path}/MotifCounts/{dataset}/test" # 2month1week2hour
                pickle_file_name = f"{full_motif_path}/k{K}.p"
                with open(pickle_file_name, 'wb') as f:
                    pickle.dump(motif_test_dict, f)


        # Link prediction experiment -- NOTE: didn't remove nodes from train
        if link_prediction_experiment and n_alpha == 6:

            fit_params_tup = fit_dict["fit_param_ref"]
            nodes_mem_all = fit_dict["node_mem_all_ref"]  # <--- using full node membership
            t0s = np.loadtxt(f"storage/t0/{dataset}_t0.csv", delimiter=',', usecols=1)
            runs = len(t0s)
            auc = np.zeros(runs)
            y_runs = np.zeros((n_nodes_all, n_nodes_all, runs))
            pred_runs = np.zeros((n_nodes_all, n_nodes_all, runs))
            for i, t0 in enumerate(t0s):
                # t0 = np.random.uniform(low=end_time_train, high=end_time_all - delta, size=None)
                y_mulch, pred_mulch = accuracy_test.mulch_predict_probs_and_actual(n_nodes_all, t0,
                                                                                   link_pred_delta,
                                                                                   events_dict_all,
                                                                                   fit_params_tup,
                                                                                   nodes_mem_all)
                y_runs[:, :, i] = y_mulch
                pred_runs[:, :, i] = pred_mulch
                auc[i] = accuracy_test.calculate_auc(y_mulch, pred_mulch, show_figure=False)
                print(f"at i={i} -> auc={auc[i]}")

            print(f"{fit_dict['ll_test_ref']:.5f}\t{K}\t{np.average(auc):.5f}\t{auc.std():.3f}")
            if save_link:
                full_link_path = f"{save_path}/AUC/{dataset}"
                pickle_file_name = f"{full_link_path}/auc_k{K}.p"
                auc_dict = {"t0": t0s, "auc": auc, "avg": np.average(auc), "std": auc.std(), "y__runs": y_runs,
                            "pred_runs": pred_runs, "ll_test": fit_dict['ll_test_ref']}
                with open(pickle_file_name, 'wb') as f:
                    pickle.dump(auc_dict, f)

#%% Fit multiblock Hawkes using one beta model
    beta = 3.57 # 1 month
    if len(block_range) !=0:
        print(f"Fit {n_alpha}-alpha single beta={beta}")
        train_tup = events_dict_train, n_nodes_train, T_train
        all_tup = events_dict_all, n_nodes_all, T_all
    for n_classes in block_range:
        fit_dict = model_fit_cal_log_likelihood_one_beta(train_tup, all_tup, nodes_not_in_train, n_alpha,
                                                         n_classes, beta, save_file="")
        print(f"log-likelihood:train={fit_dict['ll_train']:.3f}\tall={fit_dict['ll_all']:.3f}"
              f"\ttest={fit_dict['ll_test']:.3f}")