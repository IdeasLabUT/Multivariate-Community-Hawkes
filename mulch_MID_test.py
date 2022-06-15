"""MID dataset MULCH Experiment (Section 5.2)

This script runs MULCH on MID dataset. Then, evaluate the model's predictive and generative accuracy.

For performance evaluation:
    1. test-loglikelihood = (full dataset - train dataset log-likelihood) / #test events.

    2. motif counts of simulation: generate networks from MULCH fit parameters, then
    count motifs and compare to actual network's motif counts. We use MAPE score for comparison.

    3. dynamic link prediction: sample a timestamp (t0) in test duration. For each node pair
    in network, predict probability of an event in interval [t0: t0+delta]. We compute
    AUC between actual and predicted events.

Instructions for running script - the following variables can be changed:
 - dataset: choose which dataset to run MULCH on
 - K_range: # number of blocks (K) range - ex: range(1,11)
 - n_alpha:  # number of excitations types - choose between 2, 4, or 6 (default=6)
 - REF_ITER: # maximum refinement interation - set to 0 for no refinement (default=15)
 - motif_experiment: (bool) if True, run motif count experiment
 - n_motif_simulations: # number of simulations for motif count experiment (default=10)
 - link_prediction_experiment: (bool) if True, run link prediction experiment

Other dataset-specific variables:
 - betas: np.array of MULCH decays
 - motif_delta: (float) for motif count experiment
 - link_pred_delta: (float) delta for link prediction experiment

@author: Hadeel Soliman
"""

# TODO SUBMISSION: remove docker and all saving or read saved options
# TODO - remove read graph function --> load MID function from graph is not used
# TODO remove saved motif of dataset
# TODO add analyze blocks and plot fit parameters

import numpy as np
import pickle
import networkx as nx
import os
import matplotlib.pyplot as plt
import utils_accuracy_tests as accuracy_test
from utils_fit_refine_mulch import fit_refinement_mulch
import utils_fit_model as fit_model





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
    node_id_map_all, id_node_map_all = fit_model.get_node_id_maps(node_set_all)
    node_set_train = set(digraph1.nodes(end=timestamp_last_train))
    n_nodes_train = len(node_set_train)
    node_id_map_train, id_node_map_train = fit_model.get_node_id_maps(node_set_train)

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
    docker = True
    if docker:
        save_path = f"/result"  # when called from docker
    else:
        save_path = f'/shared/Results/MultiBlockHawkesModel'

    PRINT_DETAILS = True   # print intermediate details of fitting and other test experiments

    """ Model Fitting """
    FIT_MODEL = False  # either fit mulch or read saved fit
    K_range = range(1,11)  # number of blocks (K) range ex: range(1,11)
    n_alpha = 6 # number of excitations types choose between 2, 4, or 6
    save_fit = False  # save fitted model - specify path
    REF_ITER = 0  # maximum refinement interation - set to 0 for no refinement

    # OTHER betas options
    # betas_recip = np.array([4*30, 30, 1]) * (1000 / 8380)  # [4 month, 1 month, 1 day]
    # betas_recip = np.array([2*30, 7, 1/12]) * (1000 / 8380)  # [2 month, 1week, 2 hour]
    # betas_recip = np.array([30, 7, 1]) * (1000 / 8380)  # [1 month, 1 week, 1 day]

    betas_recip = np.array([2 * 30, 2 * 7, 1 / 2]) * (1000 / 8380)  # [2 month, 2 week, 12 hours]
    # betas_recip = np.array([30])* (1000 / 8380)
    betas = np.reciprocal(betas_recip)



    """ Simulation from fit parameters and count motifs experiment"""
    motif_experiment = False
    n_motif_simulations = 10 # number of simulations to count motifs on
    save_motif = False # save simulation motif counts - specify save path in code

    """ link prediction experiment"""
    link_prediction_experiment = True
    save_link = True  # save link prediction results specify path in code

    # read MID csv file. Data stored as 3 columns (attacker country, attacked country, timestamp)
    print("Load MID dataset - timestampes scaled [0:1000]")
    
    # print("Added random Gaussian noise (mean=0, std=1hour) to events with same timestamp")
    # pickle_file = os.path.join(os.getcwd(), "storage", "datasets", "MID", "MID_std1hour.p")
    # train_tup, all_tup, nodes_not_in_train1 = load_data_train_all(pickle_file)
    # events_dict_train1, T_train1, n_nodes_train1, n_events_train1, id_node_map_train1 = train_tup
    # events_dict_all1, T_all1, n_nodes_all1, n_events_all1, id_node_map_all1 = all_tup

    file_path_csv = os.path.join(os.getcwd(), "storage", "datasets", "MID", "MID.csv")
    # read full data set and use 0.8 as train
    train_tup, all_tup, nodes_not_in_train = fit_model.read_csv_split_train(file_path_csv, delimiter=',',
                                                                            remove_not_train=False)
    # train and full dataset tuples
    events_dict_train, n_nodes_train, T_train, n_events_train, id_node_map_train = train_tup
    events_dict_all, n_nodes_all, T_all, n_events_all, id_node_map_all = all_tup

    dataset = "MID"
    link_pred_delta = 7.15 # two month link prediction delta
    motif_delta_month = 4 # around one month motif delta



#%% fit MULCH with refinement
    if len(K_range) !=0:
        print(f"Fit MID using {n_alpha}-alpha MULCH at betas={betas}, max #refinement iterations={REF_ITER}")
    for K in K_range:
        if FIT_MODEL:
            print("\nFit MULCH at K=", K)
            sp_tup, ref_tup, ref_message = fit_refinement_mulch(events_dict_train, n_nodes_train, T_train, K,
                                                                betas, n_alpha, max_ref_iter=REF_ITER, verbose=PRINT_DETAILS)

            # Fit results using spectral clustering for node membership
            nodes_mem_train_sp, fit_param_sp, ll_train_sp, n_events_train, fit_time_sp = sp_tup
            # full dataset nodes membership
            node_mem_all_sp = fit_model.assign_node_membership_for_missing_nodes(nodes_mem_train_sp, nodes_not_in_train)
            ll_all_sp, n_events_all = fit_model.log_likelihood_mulch(fit_param_sp, events_dict_all, node_mem_all_sp, K,
                                                                     T_all)
            # train, full, test log-likelihoods per event
            ll_all_event_sp = ll_all_sp / n_events_all
            ll_train_event_sp = ll_train_sp / n_events_train
            ll_test_event_sp = (ll_all_sp - ll_train_sp) / (n_events_all - n_events_train)

            # Fit results after nodes membership refinement iterations
            nodes_mem_train_ref, fit_param_ref, ll_train_ref, num_events, fit_time_ref = ref_tup
            # full dataset nodes membership
            nodes_mem_all_ref = fit_model.assign_node_membership_for_missing_nodes(nodes_mem_train_ref,
                                                                                   nodes_not_in_train)
            ll_all_ref, n_events_all = fit_model.log_likelihood_mulch(fit_param_ref, events_dict_all, nodes_mem_all_ref,
                                                                      K, T_all)
            # train, full, test log-likelihoods per event
            ll_all_event_ref = ll_all_ref / n_events_all
            ll_train_event_ref = ll_train_ref / n_events_train
            ll_test_event_ref = (ll_all_ref - ll_train_ref) / (n_events_all - n_events_train)

            print(f"->Spectral log-likelihood:\ttrain={ll_train_event_sp:.3f}\tall={ll_all_event_sp:.3f}"
                  f"\ttest={ll_test_event_sp:.3f}")
            print(f"->Refinement log-likelihood:  \ttrain={ll_train_event_ref:.3f}\tall={ll_all_event_ref:.3f}"
                  f"\ttest={ll_test_event_ref:.3f}")

            print("\n->Analyzing refinement node membership: Counties in each block")
            fit_model.analyze_block(nodes_mem_train_ref, K, id_node_map_train)
            print("Plotting fit parameters")
            fit_model.plot_mulch_param(fit_param_ref, n_alpha)

            if save_fit:
                fit_dict = {}
                fit_dict["fit_param_ref"] = fit_param_ref
                fit_dict["node_mem_train_ref"] = nodes_mem_train_ref
                fit_dict["node_mem_all_ref"] = nodes_mem_all_ref
                fit_dict["ll_train_ref"] = ll_train_event_ref
                fit_dict["ll_all_ref"] = ll_all_event_ref
                fit_dict["ll_test_ref"] = ll_test_event_ref
                fit_dict["fit_param_sp"] = fit_param_sp
                fit_dict["node_mem_train_sp"] = nodes_mem_train_sp
                fit_dict["node_mem_all_sp"] = node_mem_all_sp
                fit_dict["ll_train_sp"] = ll_train_event_sp
                fit_dict["ll_all_sp"] = ll_all_event_sp
                fit_dict["ll_test_sp"] = ll_test_event_sp
                fit_dict["message"] = ref_message
                fit_dict["n_classes"] = K
                fit_dict["fit_time_sp(s)"] = fit_time_sp
                fit_dict["fit_time_ref(s)"] = fit_time_ref
                fit_dict["train_end_time"] = T_train
                fit_dict["all_end_time"] = T_all
                fit_dict["id_node_map_train"] = id_node_map_train
                fit_dict["id_node_map_all"] = id_node_map_all
                full_fit_path = f'{save_path}/MID/test'
                pickle_file_name = f"{full_fit_path}/k_{K}.p"
                with open(pickle_file_name, 'wb') as f:
                    pickle.dump(fit_dict, f)

        # read saved fit
        else:
            full_fit_path = f"{save_path}/MID/6alpha/2month2week1_2day"
            with open(f"{full_fit_path}/k_{K}.p", 'rb') as f:
                fit_dict = pickle.load(f)
            # refinement parameters
            fit_param_ref = fit_dict["fit_param_ref"]
            nodes_mem_train_ref = fit_dict["node_mem_train_ref"]
            nodes_mem_all_ref = fit_dict["node_mem_all_ref"]
            ll_test_event_ref = fit_dict["ll_test_ref"]
            print(f"\nSpectral log-likelihood:\ttrain={fit_dict['ll_train_sp']:.3f}\tall={fit_dict['ll_all_sp']:.3f}"
                  f"\ttest={fit_dict['ll_test_sp']:.3f}")
            print(f"Refinement log-likelihood:  \ttrain={fit_dict['ll_train_ref']:.3f}\tall={fit_dict['ll_all_ref']:.3f}"
                  f"\ttest={fit_dict['ll_test_ref']:.3f}")

        # Simulation and motif experiments
        if motif_experiment:
            print(f"\n\nMotifs Count Experiment at delta={motif_delta_month} (#simulations={n_motif_simulations})")
            # ---> Either run motif counts on dataset
            # # compute dataset's reciprocity, transitivity, and (6, 6) temporal motifs counts matrix
            # recip, trans, dataset_motif, n_events_train = \
            #     accuracy_test.cal_recip_trans_motif(events_dict_train, n_nodes_train, motif_delta_month, verbose=PRINT_DETAILS)
            # print(f"->actual {dataset}: reciprocity={recip:.4f}, transitivity={trans:.4f}, #events:{n_events_train}")
            # if save_dataset_motif:
            #     save_dataset_motif_path = ""
            #     results_dict = {}
            #     results_dict["dataset_motif"] = dataset_motif
            #     results_dict["dataset_recip"] = recip
            #     results_dict["dataset_trans"] = trans
            #     results_dict["dataset_n_events"] = n_events_train
            #     with open(f"{save_dataset_motif_path}.p", 'wb') as fil:
            #         pickle.dump(results_dict, fil)
            # dataset_motif_tup = (recip, trans, dataset_motif, n_events_train)

            # ---> OR read networks recip, trans, motifs count from saved pickle
            with open(f"storage/datasets_motif_counts/month_MID_counts.p", 'rb') as f:
                dataset_motif_dict = pickle.load(f)
            dataset_motif = dataset_motif_dict["dataset_motif"]
            recip = dataset_motif_dict["dataset_recip"]
            trans = dataset_motif_dict["dataset_trans"]
            n_events_train = dataset_motif_dict["dataset_n_events"]
            print(f"->actual {dataset}: reciprocity={recip:.4f}, transitivity={trans:.4f}, #events:{n_events_train}")
            dataset_motif_tup = (recip, trans, dataset_motif, n_events_train)

            # Simulate networks using MULCH fit parameters and compute reciprocity,  motif counts
            motif_test_dict = accuracy_test.simulate_count_motif_experiment(dataset_motif_tup, fit_param_ref,
                                                                            nodes_mem_train_ref, K, T_train,
                                                                            motif_delta_month,
                                                                            n_sim=n_motif_simulations,
                                                                            verbose=PRINT_DETAILS)
            print("\n->actual dataset motifs count at delta=", motif_delta_month)
            print(np.asarray(motif_test_dict["dataset_motif"], dtype=int))
            print("->average motifs count over ", n_motif_simulations, " simulations")
            print(np.asarray(motif_test_dict["sim_motif_avg"], dtype=int))
            print(f'-> at K={K}: MAPE = {motif_test_dict["mape"]:.2f}')

            if save_motif:
                full_motif_path = f"{save_path}/MotifCounts/{dataset}/test" # 2month1week2hour
                pickle_file_name = f"{full_motif_path}/k{K}.p"
                with open(pickle_file_name, 'wb') as f:
                    pickle.dump(motif_test_dict, f)


        # Link prediction experiment -- NOTE: used nodes in full dataset
        if link_prediction_experiment:
            print("\n\nLink Prediction Experiment at delta=", link_pred_delta)

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
                                                                                   fit_param_ref,
                                                                                   nodes_mem_all_ref)
                y_runs[:, :, i] = y_mulch
                pred_runs[:, :, i] = pred_mulch
                auc[i] = accuracy_test.calculate_auc(y_mulch, pred_mulch, show_figure=False)
                if PRINT_DETAILS:
                    print(f"at i={i} -> auc={auc[i]}")
            print(f"-> at K={K}: average AUC={np.average(auc):.5f}, std={auc.std():.3f}")

            if save_link:
                full_link_path = f"{save_path}/AUC/{dataset}/{n_alpha}alpha"
                pickle_file_name = f"{full_link_path}/auc_k{K}.p"
                auc_dict = {"t0": t0s, "auc": auc, "avg": np.average(auc), "std": auc.std(), "y__runs": y_runs,
                            "pred_runs": pred_runs, "ll_test": ll_test_event_ref}
                with open(pickle_file_name, 'wb') as f:
                    pickle.dump(auc_dict, f)
