import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import time
import sys
sys.path.append("./CHIP-Network-Model")
import dataset_utils
import generative_model_utils as utils
import model_fitting_utils as model_utils
import bhm_parameter_estimation as bhm_utils
from dataset_utils import load_enron_train_test, load_reality_mining_test_train
# from model_fitting_utils import fit_community_model
# from generative_model_utils import event_dict_to_aggregated_adjacency, event_dict_to_adjacency, num_events_in_event_dict
from chip_generative_model import community_generative_model
from bhm_generative_model import block_generative_model
# from spectral_clustering import spectral_cluster
import copy
import time
from utils_sum_betas_bp import cal_num_events
from Analyze_MID_test import load_data_train_all
from MBHP_datasets_fit import load_facebook_chip
import MultiBlockFit as MBHP
import pandas as pd
import pickle


CHIP = True
BHM = False
SIM_FIT = False

# # # load Dataset
dataset = "MID"

if dataset == "RealityMining" or dataset =="Enron-2":
    if dataset =="Enron-2":
        train_tuple, test_tuple, all_tuple, nodes_not_in_train = load_enron_train_test(remove_nodes_not_in_train=False)
    else:
        train_tuple, test_tuple, all_tuple, nodes_not_in_train = load_reality_mining_test_train(remove_nodes_not_in_train=False)
    events_dict_train, n_nodes_train, T_train = train_tuple
    events_dict_all, n_nodes_all, T_all = all_tuple
    n_events_train = cal_num_events(events_dict_train)
    n_events_all = cal_num_events(events_dict_all)
elif dataset == "MID":
    f = './storage/datasets/MID/MID_std1hour.p'
    train_tup, all_tup, nodes_not_in_train = load_data_train_all(f, split_ratio=0.8, scale=1000, remove_small_comp=True)
    events_dict_train, T_train, n_nodes_train, n_events_train, id_node_map_train = train_tup
    events_dict_all, T_all, n_nodes_all, n_events_all, id_node_map_all = all_tup
elif dataset == "fb-forum":
    data_path = "/nethome/hsolima/MultivariateBlockHawkesProject/MultivariateBlockHawkes/storage/datasets"
    save_path = "/shared/Results/MultiBlockHawkesModel/LSH_tests"
    train_path = "fb-forum/fb_forum_train.csv"
    test_path = "fb-forum/fb_forum_test.csv"
    # read source, target, timestamp of train dataset - ignore heades
    train_data = np.loadtxt(f"{data_path}/{train_path}", delimiter=",", skiprows=1, usecols=[0, 1, 2])
    test = np.loadtxt(f"{data_path}/{test_path}", delimiter=",", skiprows=1, usecols=[0, 1, 2])

    # train node id map
    nodes_train_set = set(np.r_[train_data[:, 0], train_data[:, 1]])
    node_id_map = {}
    for i, n in enumerate(nodes_train_set):
        node_id_map[n] = i

    # create events_dict_train
    events_dict_train = {}
    for i in range(len(train_data)):
        sender_id, receiver_id = node_id_map[train_data[i,0]], node_id_map[train_data[i,1]]
        if (sender_id, receiver_id) not in events_dict_train:
            events_dict_train[(sender_id, receiver_id)] = []
        events_dict_train[(sender_id, receiver_id)].append(train_data[i,2])

    # remove nodes in test not in train
    n_events_test = 0
    events_dict_all = copy.deepcopy(events_dict_train)
    for i in range(len(test)):
        if test[i, 0] in node_id_map and test[i, 1] in node_id_map:
            n_events_test +=1
            sender_id, receiver_id = node_id_map[test[i,0]], node_id_map[test[i,1]]
            if (sender_id, receiver_id) not in events_dict_all:
                events_dict_all[(sender_id, receiver_id)] = []
            events_dict_all[(sender_id, receiver_id)].append(test[i, 2])
    nodes_not_in_train = []
    n_nodes_train = n_nodes_all = len(nodes_train_set)
    T_train = train_data[-1, 2]
    T_all = test[-1, 2]
    n_events_train = len(train_data)
    n_events_all = n_events_train + n_events_test
    n_train_events = 0
    for (u, v) in events_dict_train:
        n_train_events += len(events_dict_train[(u, v)])
    print(n_train_events)
elif dataset == "Enron-15":
    save_path = "/shared/Results/MultiBlockHawkesModel/LSH_tests"
    with open('storage/datasets/enron2/enron-events.pckl', 'rb') as f:
        n_nodes_all, T_all, enron_all = pickle.load(f)
    n_nodes_train = n_nodes_all
    T_train = 316

    events_dict_train = {}
    events_dict_all = {}
    n_events_train = 0
    n_events_all = len(enron_all)
    for u, v, t in enron_all:
        if t <= T_train:
            n_events_train += 1
            if (u, v) not in events_dict_train:
                events_dict_train[(u, v)] = []
            events_dict_train[(u, v)].append(t)
        if (u, v) not in events_dict_all:
            events_dict_all[(u, v)] = []
        events_dict_all[(u, v)].append(t)
    nodes_not_in_train = []
elif dataset == "FacebookFiltered":
    #### Filtered Facebook dataset
    events_dict_all, n_nodes_all, T_all = load_facebook_chip('./storage/datasets/facebook_filtered/facebook-wall-filtered.txt')
    events_dict_train, T_train = MBHP.split_train(events_dict_all)
    nodes_not_in_train = np.array([])
    dataset = "FacebookFiltered"
    # assuming the nodes of train are the same as all
    n_nodes_train = n_nodes_all
    n_events_train = cal_num_events(events_dict_train)
    n_events_all = cal_num_events(events_dict_all)


for K in range(1, 11):
    if CHIP:
        start_fit_time = time.time()
        node_mem_train, bp_mu_t, bp_alpha_t, bp_beta_t, events_dict_bp_train = model_utils.fit_community_model(
            events_dict_train, n_nodes_train, T_train, K, 0, -1, verbose=False)
        end_time_fit = time.time()
        # Add nodes that were not in train to the largest block
        node_mem_all = model_utils.assign_node_membership_for_missing_nodes(node_mem_train, nodes_not_in_train)
        # Calculate log-likelihood given the entire dataset
        events_dict_bp_all = utils.event_dict_to_block_pair_events(events_dict_all, node_mem_all, K)
        ll_all = model_utils.calc_full_log_likelihood(events_dict_bp_all, node_mem_all, bp_mu_t, bp_alpha_t, bp_beta_t,
                                                      T_all, K)
        # Calculate log-likelihood given the train dataset
        ll_train = model_utils.calc_full_log_likelihood(events_dict_bp_train, node_mem_train, bp_mu_t, bp_alpha_t, bp_beta_t,
                                                        T_train, K)
        fit_time = end_time_fit - start_fit_time
        ll_all_event = ll_all / n_events_all
        ll_train_event = ll_train / n_events_train
        ll_test_event = (ll_all - ll_train) / (n_events_all - n_events_train)
        # print(f"K={K}:\ttrain={ll_train_event:.3f}\tall={ll_all_event:.3f}\ttest={ll_test_event:.3f}")
        print(f"{ll_train_event:.3f}\t{ll_all_event:.3f}\t{ll_test_event:.3f}\t{fit_time:.3f}")
    elif BHM:
        print("K = ", K)
        try:
            # Fitting the model to the train data
            node_mem_train, bp_mu_t, bp_alpha_t, bp_beta_t, events_dict_bp_train = bhm_utils.fit_block_model(events_dict_train,
                n_nodes_train, T_train, K, local_search_max_iter=200, local_search_n_cores=0, verbose=True)
            # Add nodes that were not in train to the largest block
            node_mem_all = model_utils.assign_node_membership_for_missing_nodes(node_mem_train, nodes_not_in_train)

            # Calculate log-likelihood given the entire dataset
            events_dict_bp_all = bhm_utils.event_dict_to_combined_block_pair_events(events_dict_all, node_mem_all, K)

            ll_all = bhm_utils.calc_full_log_likelihood(events_dict_bp_all, node_mem_all, bp_mu_t, bp_alpha_t, bp_beta_t, T_all, K,
                                                        add_com_assig_log_prob=True)

            # Calculate log-likelihood given the train dataset
            ll_train = bhm_utils.calc_full_log_likelihood(events_dict_bp_train, node_mem_train, bp_mu_t, bp_alpha_t, bp_beta_t, T_train, K,
                                                          add_com_assig_log_prob=True)

            ll_all_event = ll_all / n_events_all
            ll_train_event = ll_train / n_events_train
            ll_test_event = (ll_all - ll_train) / (n_events_all - n_events_train)
            print(f"K={K}:\ttrain={ll_train_event:.3f}\tall={ll_all_event:.3f}\ttest={ll_test_event:.3f}")
            # print(f"{ll_train_event:.3f}\t{ll_all_event:.3f}\t{ll_test_event:.3f}")

            results_dict = {}
            results_dict["param"] = (bp_mu_t, bp_alpha_t, bp_beta_t)
            results_dict["node_mem_train"] = node_mem_train
            results_dict["node_mem_all"] = node_mem_all
            results_dict["ll_train"] = ll_train_event
            results_dict["ll_all"] = ll_all_event
            results_dict["ll_test"] = ll_test_event
            results_dict["local_search"] = 200
            # results_path = '/shared/Results/MultiBlockHawkesModel/BHM_MID'
            # results_path = "/data/BHM_MID"  #when called from docker
            # results_path = "/shared/Results/MultiBlockHawkesModel/MotifCounts/BHM/RealityMining/param_fit"
            file_name = f"{dataset}_k_{K}.p"
            pickle_file_name = f"{save_path}/{file_name}"
            with open(pickle_file_name, 'wb') as f:
                pickle.dump(results_dict, f)
        except Exception:
            pass


#%% simulate from CHIP using scaled parameters
if SIM_FIT:
    print(f"Fit and simulate from {dataset}")
    nRuns = 10
    motif_delta = 45  # week - reality
    if dataset == "Enron":
        motif_delta = 100  # week - enron
    elif dataset == "MID":
        motif_delta = 4
    T_sim = T_train  # only train dataset

    # read dataset motif counts
    with open(f"Datasets_motif_counts/month_{dataset}_counts.p", 'rb') as f:
        Mcounts = pickle.load(f)
    dataset_motif_month = Mcounts["dataset_motif"]
    dataset_recip = Mcounts["dataset_recip"]
    dataset_trans = Mcounts["dataset_trans"]
    dataset_n_events_train = Mcounts["dataset_n_events"]
    print(f"{dataset}: reciprocity={dataset_recip:.4f}, transitivity={dataset_trans:.4f}, #events:{dataset_n_events_train}")

    # sim_results_path = f"/shared/Results/MultiBlockHawkesModel/MotifCounts/BHM/{dataset}"
    # # sim_results_path = f"/data/MotifCounts/BHM/{dataset}"
    # read_path = f"/shared/Results/MultiBlockHawkesModel/MotifCounts/BHM/{dataset}/param_fit"
    # # read_path = f"/data/MotifCounts/BHM/{dataset}/param_fit"
    # k_range = list(range(14,15)) + [20, 30, 40, 45]
    # for K in k_range:
    #     # 1) fit BHM
    #     try:
    #         # # Fitting the model to the train data
    #         # node_mem_train, bp_mu_t, bp_alpha_t, bp_beta_t, events_dict_bp_train = bhm_utils.fit_block_model(events_dict_train,
    #         #                                                                                                  n_nodes_train, T_train, K,
    #         #                                                                                                  local_search_max_iter=100,
    #         #                                                                                                  local_search_n_cores=0,
    #         #                                                                                                  verbose=True)
    #         # # Add nodes that were not in train to the largest block
    #         # node_mem_all = model_utils.assign_node_membership_for_missing_nodes(node_mem_train, nodes_not_in_train)
    #         #
    #         # # Calculate log-likelihood given the entire dataset
    #         # events_dict_bp_all = bhm_utils.event_dict_to_combined_block_pair_events(events_dict_all, node_mem_all, K)
    #         #
    #         # ll_all = bhm_utils.calc_full_log_likelihood(events_dict_bp_all, node_mem_all, bp_mu_t, bp_alpha_t, bp_beta_t, T_all, K,
    #         #                                             add_com_assig_log_prob=True)
    #         #
    #         # # Calculate log-likelihood given the train dataset
    #         # ll_train = bhm_utils.calc_full_log_likelihood(events_dict_bp_train, node_mem_train, bp_mu_t, bp_alpha_t, bp_beta_t, T_train, K,
    #         #                                               add_com_assig_log_prob=True)
    #         #
    #         # ll_all_event = ll_all / n_events_all
    #         # ll_train_event = ll_train / n_events_train
    #         # ll_test_event = (ll_all - ll_train) / (n_events_all - n_events_train)
    #         # print(f"K={K}:\ttrain={ll_train_event:.3f}\tall={ll_all_event:.3f}\ttest={ll_test_event:.3f}")
    #         # # print(f"{ll_train_event:.3f}\t{ll_all_event:.3f}\t{ll_test_event:.3f}")
    #         #
    #         # results_dict = {}
    #         # results_dict["param"] = (bp_mu_t, bp_alpha_t, bp_beta_t)
    #         # results_dict["node_mem_train"] = node_mem_train
    #         # results_dict["node_mem_all"] = node_mem_all
    #         # results_dict["ll_train"] = ll_train_event
    #         # results_dict["ll_all"] = ll_all_event
    #         # results_dict["ll_test"] = ll_test_event
    #         # results_dict["local_search"] = 500
    #         # file_name = f"k_{K}.p"
    #         # pickle_file_name = f"{sim_results_path}/param_fit/{file_name}"
    #         # with open(pickle_file_name, 'wb') as f:
    #         #     pickle.dump(results_dict, f)
    #
    #         # read fit
    #         pickle_file_name = f"{read_path}/k_{K}.p"
    #         with open(pickle_file_name, 'rb') as f:
    #             results_dict_s = pickle.load(f)
    #         bp_mu_t, bp_alpha_t, bp_beta_t = results_dict_s["param"]
    #         node_mem_train = results_dict_s["node_mem_train"]
    #         print(f"K={K}:\ttrain={results_dict_s['ll_train']:.3f}\tall={results_dict_s['ll_all']:.3f}"
    #               f"\ttest={results_dict_s['ll_test']:.3f}")
    #
    #         # 2) nRuns simulations
    #         print("\nsimulate at K=", K, " motif delta=", motif_delta)
    #         _, block_count = np.unique(node_mem_train, return_counts=True)
    #         block_prob = block_count / sum(block_count)
    #         sim_motif_avg_month = np.zeros((6, 6))
    #         sim_n_events_avg, sim_trans_avg, sim_recip_avg = 0, 0, 0
    #         for run in range(nRuns):
    #             # simulate using fitted parameters
    #             print("simulation ", run)
    #             _, events_dict_sim = block_generative_model(n_nodes_train, block_prob, bp_mu_t, bp_alpha_t, bp_beta_t, T_sim)
    #             n_evens_sim = cal_num_events(events_dict_sim)
    #             # print(n_evens_sim)
    #             recip_sim, trans_sim, sim_motif_month = MBHP.cal_recip_trans_motif(events_dict_sim, n_nodes_train, motif_delta)
    #             sim_motif_avg_month += sim_motif_month
    #             print(f"n_events={n_evens_sim}, recip={recip_sim:.4f}, trans={trans_sim:.4f}")
    #             sim_recip_avg += recip_sim
    #             sim_trans_avg += trans_sim
    #             sim_n_events_avg += n_evens_sim
    #         # simulation runs at a certain K is done
    #         sim_motif_avg_month /= nRuns
    #         sim_recip_avg /= nRuns
    #         sim_trans_avg /= nRuns
    #         sim_n_events_avg /= nRuns
    #
    #         # calculate MAPE
    #         mape = 100 / 36 * np.sum(np.abs(sim_motif_avg_month - (dataset_motif_month+1)) / (dataset_motif_month+1))
    #
    #         # save results
    #         results_dict = {}
    #         results_dict["K"] = K
    #         results_dict["nRuns"] = nRuns
    #         results_dict["parameters"] = (bp_mu_t, bp_alpha_t, bp_beta_t)
    #         results_dict["motif_delta"] = motif_delta
    #         results_dict["dataset_motif_month"] = dataset_motif_month
    #         results_dict["dataset_recip"] = dataset_recip
    #         results_dict["dataset_trans"] = dataset_trans
    #         results_dict["dataset_n_events"] = dataset_n_events_train
    #         results_dict["sim_motif_avg_month"] = sim_motif_avg_month
    #         results_dict["sim_recip_avg"] = sim_recip_avg
    #         results_dict["sim_trans_avg"] = sim_trans_avg
    #         results_dict["sim_n_events_avg"] = sim_n_events_avg
    #         results_dict["mape"] = mape
    #
    #         print(np.asarray(results_dict["dataset_motif_month"], dtype=int))
    #         print("")
    #         print(np.asarray(results_dict["sim_motif_avg_month"], dtype=int))
    #
    #         file_name = f"k{K}.p"
    #         pickle_file_name = f"{sim_results_path}/{file_name}"
    #         with open(pickle_file_name, 'wb') as f:
    #             pickle.dump(results_dict, f)
    #     except Exception as e:
    #         print(e)
    #         pass

    par_results_path = f'/shared/Results/MultiBlockHawkesModel/MotifCounts/CHIP/{dataset}/param_fit'
    # par_results_path = f'/data/MotifCounts/CHIP/{dataset}/param_fit'
    sim_results_path = f"/shared/Results/MultiBlockHawkesModel/MotifCounts/CHIP/{dataset}"
    for K in range(10,15):
        # 1) fit CHIP
        node_mem_train, bp_mu_t, bp_alpha_t, bp_beta_t, events_dict_bp_train = model_utils.fit_community_model(events_dict_train,
            n_nodes_train, T_train, K, 0, -1, verbose=False)
        # Add nodes that were not in train to the largest block
        node_mem_all = model_utils.assign_node_membership_for_missing_nodes(node_mem_train, nodes_not_in_train)
        # Calculate log-likelihood given the entire dataset
        events_dict_bp_all = utils.event_dict_to_block_pair_events(events_dict_all, node_mem_all, K)
        ll_all = model_utils.calc_full_log_likelihood(events_dict_bp_all, node_mem_all, bp_mu_t, bp_alpha_t, bp_beta_t, T_all, K)
        # Calculate log-likelihood given the train dataset
        ll_train = model_utils.calc_full_log_likelihood(events_dict_bp_train, node_mem_train, bp_mu_t, bp_alpha_t, bp_beta_t, T_train, K)

        ll_all_event = ll_all / n_events_all
        ll_train_event = ll_train / n_events_train
        ll_test_event = (ll_all - ll_train) / (n_events_all - n_events_train)
        print(f"K={K}:\ttrain={ll_train_event:.3f}\tall={ll_all_event:.3f}\ttest={ll_test_event:.3f}")

        # save fit parameters
        results_dict = {}
        results_dict["param"] = (bp_mu_t, bp_alpha_t, bp_beta_t)
        results_dict["node_mem_train"] = node_mem_train
        results_dict["node_mem_all"] = node_mem_all
        results_dict["ll_train"] = ll_train_event
        results_dict["ll_all"] = ll_all_event
        results_dict["ll_test"] = ll_test_event
        file_name = f"k_{K}.p"
        pickle_file_name = f"{par_results_path}/{file_name}"
        with open(pickle_file_name, 'wb') as f:
            pickle.dump(results_dict, f)

        # 2) nRuns simulations
        print("\nsimulate at K=", K, " motif delta=", motif_delta)
        _, block_count = np.unique(node_mem_train, return_counts=True)
        block_prob = block_count / sum(block_count)
        sim_motif_avg_month = np.zeros((6, 6))
        sim_n_events_avg, sim_trans_avg, sim_recip_avg = 0, 0, 0
        for run in range(nRuns):
            # simulate using fitted parameters
            print("simulation ", run)
            _, events_dict_sim = community_generative_model(n_nodes_train, block_prob, bp_mu_t, bp_alpha_t, bp_beta_t, T_sim)
            n_evens_sim = cal_num_events(events_dict_sim)
            # print(n_evens_sim)
            recip_sim, trans_sim, sim_motif_month = MBHP.cal_recip_trans_motif(events_dict_sim, n_nodes_train, motif_delta)
            sim_motif_avg_month += sim_motif_month
            print(f"n_events={n_evens_sim}, recip={recip_sim:.4f}, trans={trans_sim:.4f}")
            sim_recip_avg += recip_sim
            sim_trans_avg += trans_sim
            sim_n_events_avg += n_evens_sim
        # simulation runs at a certain K is done
        sim_motif_avg_month /= nRuns
        sim_recip_avg /= nRuns
        sim_trans_avg /= nRuns
        sim_n_events_avg /= nRuns

        # calculate MAPE
        mape = 100 / 36 * np.sum(np.abs(sim_motif_avg_month - (dataset_motif_month+1)) / (dataset_motif_month+1))

        # save results
        results_dict = {}
        results_dict["K"] = K
        results_dict["nRuns"] = nRuns
        results_dict["parameters"] = (bp_mu_t, bp_alpha_t, bp_beta_t)
        results_dict["motif_delta"] = motif_delta
        results_dict["dataset_motif_month"] = dataset_motif_month
        results_dict["dataset_recip"] = dataset_recip
        results_dict["dataset_trans"] = dataset_trans
        results_dict["dataset_n_events"] = dataset_n_events_train
        results_dict["sim_motif_avg_month"] = sim_motif_avg_month
        results_dict["sim_recip_avg"] = sim_recip_avg
        results_dict["sim_trans_avg"] = sim_trans_avg
        results_dict["sim_n_events_avg"] = sim_n_events_avg
        results_dict["mape"] = mape

        print(np.asarray(results_dict["dataset_motif_month"], dtype=int))
        print("")
        print(np.asarray(results_dict["sim_motif_avg_month"], dtype=int))

        file_name = f"k{K}.p"
        pickle_file_name = f"{sim_results_path}/{file_name}"
        with open(pickle_file_name, 'wb') as f:
            pickle.dump(results_dict, f)





