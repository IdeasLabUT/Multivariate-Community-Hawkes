import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import time
import random
import pickle
from sklearn.metrics import adjusted_rand_score
import OneBlockFit
import MultiBlockFit as MBHP
from refinement_alg import model_fit_refine_kernel_sum_exact, model_fit_refine_kernel_sum_relative

import sys
sys.path.append("./dynetworkx/classes")
from impulsedigraph import ImpulseDiGraph

sys.path.append("./CHIP-Network-Model")
from dataset_utils import load_enron_train_test, load_reality_mining_test_train, get_node_map, load_facebook_wall
from generative_model_utils import event_dict_to_aggregated_adjacency, event_dict_to_adjacency
from spectral_clustering import spectral_cluster1


"""
Functions for working with Community Hawkes Independent Pairs (CHIP) models.

@author: Kevin S. Xu
"""
def load_facebook_chip(data_file_name, timestamp_max=1000):
    # sender_id receiver_id sender_id
    data = np.loadtxt(data_file_name, np.float)

    # Sorting by unix_timestamp and adjusting first timestamp to start from 0
    data = data[data[:, 2].argsort()]
    data[:, 2] = data[:, 2] - data[0, 2]

    if timestamp_max is not None:
        # Scale timestamps to 0 to timestamp_max
        data[:, 2] = (data[:, 2] - min(data[:, 2])) / (max(data[:, 2])
            - min(data[:, 2])) * timestamp_max

    duration = data[-1, 2]

    node_set = set(data[:, 0].astype(np.int)).union(data[:, 1].astype(np.int))
    node_id_map = get_node_map(node_set)

    event_dict = {}
    for i in range(data.shape[0]):
        receiver_id = node_id_map[np.int(data[i, 0])]
        sender_id = node_id_map[np.int(data[i, 1])]

        if (sender_id, receiver_id) not in event_dict:
            event_dict[(sender_id, receiver_id)] = []

        event_dict[(sender_id, receiver_id)].append(data[i, 2])

    return event_dict, len(node_set), duration


if __name__ == "__main__":
    FIT_MODEl = False
    FIT_MODEl_RHO = False
    SIM_COUNT_MOTIF = False
    FIT_REF_MODEL = False
    SIM_COUNT_MOTIF_REF = False
    SIM_COUNT_MOTIF_TEMP = True
    docker = False


    np.set_printoptions(suppress=True)  # always print floating point numbers using fixed point notation

    # read Reality Mining dataset
    dataset = "RealityMining"
    print("Reality Mining Dataset - multiblock HawkesProcess")
    train_tuple, test_tuple, all_tuple, nodes_not_in_train = load_reality_mining_test_train(remove_nodes_not_in_train=False)

    # ### read Enron dataset
    # dataset = "Enron"
    # print("Enron Dataset - multiblock HawkesProcess")
    # train_tuple, test_tuple, all_tuple, nodes_not_in_train = load_enron_train_test(remove_nodes_not_in_train=False)

    # # Entire Facebook Dataset
    # dataset = "Facebook"
    # print("Facebook wall-post dataset")
    # train_tuple, test_tuple, all_tuple, nodes_not_in_train = load_facebook_wall(timestamp_max=1000,
    #                                                                             largest_connected_component_only=True,
    #                                                                             train_percentage=0.8)

    # dataset tuple to events_list_full, num_nodes, duration
    events_dict_train, n_nodes_train, end_time_train = train_tuple
    events_dict_all, n_nodes_all, end_time_all = all_tuple

    # #### Filtered Facebook dataset
    # events_dict_all, n_nodes_all, end_time_all = load_facebook_chip('./storage/datasets/facebook_filtered/facebook-wall-filtered.txt')
    # events_dict_train, end_time_train = MBHP.split_train(events_dict_all)
    # nodes_not_in_train = np.array([])
    # dataset = "FacebookFiltered"
    # # assuming the nodes of train are the same as all
    # n_nodes_train = n_nodes_all
    """"""


#%% fit multiblock Hawkes using One beta & sum of kernels methods
    if FIT_MODEl:
        n_alpha = 4 # model version 6 or 4 or 2
        kernel = "sum" # model version "sum" or "single"
        save_param_fit = True

        agg_adj = event_dict_to_aggregated_adjacency(n_nodes_train, events_dict_train)

        if dataset == "Facebook":
            betas = np.array([0.02, 0.2, 20])  # [2 month , 1 week , 2 hours]
        elif dataset == "RealityMining":
            betas_recip = np.array([7, 1, 1/24]) * (1000 / 150)  # [1week, 2day, 1hour]
            betas = np.reciprocal(betas_recip)
        elif dataset == "Enron":
            betas_recip = np.array([7, 2, 1 / 4]) * (1000 / 60)  # [1week, 2days, 6 hour]
            betas = np.reciprocal(betas_recip)
        beta = 0.02
        np.set_printoptions(suppress=True)  # always print floating point numbers using fixed point notation

        for n_classes in range(1,11):
            # community detection using spectral clustering on weighted adj
            start_fit_time = time.time()
            node_mem_train = spectral_cluster1(agg_adj, n_classes, n_kmeans_init=100, normalize_z=True, multiply_s=True)
            node_mem_all = MBHP.assign_node_membership_for_missing_nodes(node_mem_train, nodes_not_in_train)

            # fit model
            if kernel == "sum" :
                params_est, ll_t, n_events_t = MBHP.model_fit_kernel_sum(n_alpha, events_dict_train, node_mem_train,
                                                                         n_classes, end_time_train, betas)
                end_fit_time = time.time()
                ll_all, n_events_all = MBHP.model_LL_kernel_sum_external(params_est, events_dict_all, node_mem_all, n_classes, end_time_all)
            else:
                params_est, ll_t, n_events_t = MBHP.model_fit_single_beta(n_alpha, events_dict_train, node_mem_train,
                                                                          n_classes, end_time_train, beta)
                end_fit_time = time.time()
                ll_all, n_events_all = MBHP.model_LL_single_beta_external(params_est, events_dict_all, node_mem_all, n_classes, end_time_all)


            # calculate log-likelihood per event
            ll_all_event = ll_all / n_events_all
            ll_train_event = ll_t / n_events_t
            ll_test_event = (ll_all - ll_t) / (n_events_all - n_events_t)
            time_to_fit = end_fit_time - start_fit_time
            # print(f"K={n_classes}:\ttrain={ll_train_event:.3f}\tall={ll_all_event:.3f}\ttest={ll_test_event:.3f}")
            print(f"{ll_train_event:.3f}\t{ll_all_event:.3f}\t{ll_test_event:.3f}\t{time_to_fit/60:.3f}")

            # save results
            if save_param_fit:
                results_dict = {}
                results_dict["fit_param"] = params_est
                results_dict["n_classes"] = n_classes
                results_dict["node_mem_train"] = node_mem_train
                results_dict["node_mem_all"] = node_mem_all
                results_dict["ll_train"] =  ll_train_event
                results_dict["ll_all"] = ll_all_event
                results_dict["ll_test"] = ll_test_event
                results_dict["fit_time(s)"] = time_to_fit
                results_dict["train_end_time"] = end_time_train
                results_dict["all_end_time"] = end_time_all
                # open file in write+binary mode
                if docker:
                    results_path = f"/data/{dataset}/6alpha_KernelSum"  # when called from docker
                else:
                    results_path = f'/shared/Results/MultiBlockHawkesModel/{dataset}/no_ref_{n_alpha}alpha'
                file_name = f"k_{n_classes}.p"
                pickle_file_name = f"{results_path}/{file_name}"
                with open(pickle_file_name, 'wb') as f:
                    pickle.dump(results_dict, f)

# %% fit multiblock Hawkes using One beta & rho restriction
    if FIT_MODEl_RHO:
        save_param_fit = True

        agg_adj = event_dict_to_aggregated_adjacency(n_nodes_train, events_dict_train)
        beta = 0.02
        np.set_printoptions(suppress=True)  # always print floating point numbers using fixed point notation

        for n_classes in range(1, 11):
            # community detection using spectral clustering on weighted adj
            node_mem_train = spectral_cluster1(agg_adj, n_classes, n_kmeans_init=100, normalize_z=True, multiply_s=True)
            node_mem_all = MBHP.assign_node_membership_for_missing_nodes(node_mem_train, nodes_not_in_train)

            # fit model
            start_fit_time = time.time()
            params_est, ll_t, n_events_t = MBHP.model_fit_single_beta(2, events_dict_train, node_mem_train, n_classes,
                                                                      end_time_train, beta)
            ll_all, n_events_all = MBHP.model_LL_single_beta_external(params_est, events_dict_all, node_mem_all, n_classes,
                                                                      end_time_all)
            end_fit_time = time.time()

            # # fit model rho
            # start_fit_time = time.time()
            # params_est, ll_t, n_events_t = MBHP.fit_2_alpha_rho_single_beta(events_dict_train, node_mem_train, n_classes, end_time_train, beta)
            # M_bp, n_nodes_c = MBHP.num_nodes_pairs_per_block_pair(node_mem_all, n_classes)
            # events_dict_bp_all = MBHP.events_dict_to_blocks(events_dict_all, node_mem_all, n_classes)
            # ll_all, n_events_all = MBHP.model_LL_2_alpha_rho_single_beta(params_est, events_dict_bp_all, end_time_all,
            #                                                              M_bp, n_nodes_c, n_classes)
            # end_fit_time = time.time()


            # calculate log-likelihood per event
            ll_all_event = ll_all / n_events_all
            ll_train_event = ll_t / n_events_t
            ll_test_event = (ll_all - ll_t) / (n_events_all - n_events_t)
            time_to_fit = end_fit_time - start_fit_time
            # print(f"K={n_classes}:\ttrain={ll_train_event:.3f}\tall={ll_all_event:.3f}\ttest={ll_test_event:.3f}")
            print(f"{ll_train_event:.3f}\t{ll_all_event:.3f}\t{ll_test_event:.3f}\t{time_to_fit / 60:.3f}")

            # save results
            if save_param_fit:
                results_dict = {}
                results_dict["fit_param"] = params_est
                results_dict["n_classes"] = n_classes
                results_dict["node_mem_train"] = node_mem_train
                results_dict["node_mem_all"] = node_mem_all
                results_dict["ll_train"] = ll_train_event
                results_dict["ll_all"] = ll_all_event
                results_dict["ll_test"] = ll_test_event
                results_dict["fit_time(s)"] = time_to_fit
                results_dict["train_end_time"] = end_time_train
                results_dict["all_end_time"] = end_time_all
                # open file in write+binary mode
                if docker:
                    results_path = f"/data/{dataset}/6alpha_KernelSum"  # when called from docker
                else:
                    results_path = f'/shared/Results/MultiBlockHawkesModel/{dataset}/2alpha_single'
                file_name = f"k_{n_classes}.p"
                pickle_file_name = f"{results_path}/{file_name}"
                with open(pickle_file_name, 'wb') as f:
                    pickle.dump(results_dict, f)
#%% fit and simulate multiblock Hawkes (prediction check)
    if SIM_COUNT_MOTIF:
        motif_delta = 20
        n_alpha = 6  # model version 6 or 4 or 2 alpha
        kernel = "sum"  # model version "sum" or "single"
        betas = np.array([0.01, 0.4, 20])
        beta = 0.2
        docker = True
        save_motif_counts = False

        motifs = MBHP.get_motifs()

        # # # # calculate network reciprocity, transitivity, motifs
        # recip, trans, dataset_motif = MBHP.cal_recip_trans_motif(events_dict_train, n_nodes_train, motif_delta, dataset, save=True)

        # # # # read networks recip, trans, motifs count from saved pickle
        with open(f"Datasets_motif_counts/{dataset}_counts.p", 'rb') as f:
            Mcounts = pickle.load(f)
        dataset_motif = Mcounts["dataset_motif"]
        recip = Mcounts["dataset_recip"]
        trans = Mcounts["dataset_trans"]
        n_events_train = Mcounts["dataset_n_events"]
        print(f"{dataset}: reciprocity={recip:.4f}, transitivity={trans:.4f}, #events:{n_events_train}")


        np.set_printoptions(suppress=True)  # always print floating point numbers using fixed point notation
        for n_classes in [4]:
            print("number of classes = ", n_classes)

            ### Option 1: fit model on train dataset
            agg_adj = event_dict_to_aggregated_adjacency(n_nodes_train, events_dict_train)
            node_mem_train = spectral_cluster1(agg_adj, n_classes)
            if kernel == "sum":
                print(f"fit using Kernels sum = {betas}")
                params_est, ll_t, n_events_t = MBHP.model_fit_kernel_sum(n_alpha, events_dict_train, node_mem_train,
                                                                         n_classes, end_time_train, betas)
            else:
                params_est, ll_t, n_events_t = MBHP.model_fit_single_beta(n_alpha, events_dict_train, node_mem_train,
                                                                      n_classes, end_time_train, beta)

            # ### Option 2: read saved fitting results
            # if docker:
            #     res_path = f"/data/{dataset}/Block_n_r_br_gr_KernelSum2[0.01]"
            # else:
            #     res_path = f"/shared/Results/MultiBlockHawkesModel/{dataset}/Block_n_r_br_gr_KernelSum2[0.01]"
            # file_name = f"k_4.p"
            # with open(f"{res_path}/{file_name}", 'rb') as f:
            #     res = pickle.load(f)
            # if kernel == "sum" :
            #     params_est = (res["mu"], res["alpha_n"], res["alpha_r"], res["alpha_br"], res["alpha_gr"], res["alpha_al"],
            #                   res["alpha_alr"], res["C"], res["betas"])
            #     # params_est = (res["mu"], res["alpha_n"], res["alpha_r"], res["alpha_br"], res["alpha_gr"], res["C"], res["betas"])
            #     betas = res["betas"]
            # else:
            #     params_est = (res["mu"], res["alpha_n"], res["alpha_r"], res["alpha_br"], res["alpha_gr"],
            #                   res["alpha_al"], res["alpha_alr"], res["beta"])
            #     # params_est = (res["mu"], res["alpha_n"], res["alpha_r"], res["alpha_br"], res["alpha_gr"], res["beta"])
            #     beta = res["beta"]
            # node_mem_train = res["node_mem_train"]

            #  nodes/block probabilities
            _, block_count = np.unique(node_mem_train, return_counts=True)
            block_prob = block_count / sum(block_count)


            # simulate from estimated parameters
            duration = end_time_train
            nRuns = 15
            sim_motif_avg = np.zeros((6,6))
            sim_n_events_avg, sim_trans_avg, sim_recip_avg = 0, 0, 0

            for run in range(nRuns):
                # simulate using fitted parameters
                print("simulation ", run)
                if kernel == "sum":
                    events_dict_sim, _ = MBHP.simulate_sum_kernel_model(params_est, n_nodes_train, n_classes, block_prob, duration)
                else:
                    events_dict_sim, _ = MBHP.simulate_one_beta_model(params_est, n_nodes_train, n_classes, block_prob, duration)


                recip_sim, trans_sim, sim_motif_run = MBHP.cal_recip_trans_motif(events_dict_sim, n_nodes_train, motif_delta)
                n_evens_sim = OneBlockFit.cal_num_events_2(events_dict_sim)
                print(f"n_events={n_evens_sim}, recip={recip_sim:.4f}, trans={trans_sim:.4f}")

                sim_motif_avg += sim_motif_run
                sim_recip_avg += recip_sim
                sim_trans_avg += trans_sim
                sim_n_events_avg += n_evens_sim

            # simulation runs at a certain K is done
            sim_motif_avg /= nRuns
            sim_recip_avg /= nRuns
            sim_trans_avg /= nRuns
            sim_n_events_avg /= nRuns

            # save results
            if save_motif_counts:
                results_dict = {}
                results_dict["K"] = n_classes
                if kernel == "sum":
                    results_dict["betas"] = betas
                else:
                    results_dict["beta"] = beta
                results_dict["nRuns"] = nRuns
                results_dict["parameters"] = params_est
                # results_dict['ll_train'] = ll_t/n_events_train
                # results_dict['ll_all'] = ll_all/n_events_all
                # results_dict['ll_test'] = ll_test_event

                results_dict["dataset_motif"] = dataset_motif
                results_dict["dataset_recip"] = recip
                results_dict["dataset_trans"] = trans
                results_dict["dataset_n_events"] = n_events_t

                results_dict["sim_motif_avg"] = sim_motif_avg
                results_dict["sim_recip_avg"] = sim_recip_avg
                results_dict["sim_trans_avg"] = sim_trans_avg
                results_dict["sim_n_events_avg"] = sim_n_events_avg

                print(np.asarray(results_dict["dataset_motif"], dtype=int))
                print("")
                print(np.asarray(results_dict["sim_motif_avg"], dtype=int))

                # open file in write+binary mode
                if docker:
                    results_path = f"/data"  # when called from docker
                else:
                    results_path = f'/shared/Results/MultiBlockHawkesModel/MotifCounts/{dataset}'
                file_name = f"Sim_KernelSum4_{dataset}_on_k_{n_classes}[0.01].p"
                pickle_file_name = f"{results_path}/{file_name}"
                with open(pickle_file_name, 'wb') as f:
                    pickle.dump(results_dict, f)
#%% fit MBHP with refinement
    if FIT_REF_MODEL:
        MAX_ITER = 0
        ref_batch = True
        n_alpha = 6
        if dataset == "RealityMining":
            # betas_recip = np.array([7, 1/2, 1 / 24]) * (1000 / 150)  # [1week, 1/2day, 1hour]
            # betas_recip = np.array([7*2, 1, 1/12]) * (1000 / 150)  # [2week, 1day, 2hour]
            betas_recip = np.array([7, 1, 1/24]) * (1000 / 150)  # [1week, 2day, 1hour]
            betas = np.reciprocal(betas_recip)
        if dataset == "Enron":
            # betas_recip = np.array([7, 1 / 2, 1 / 24]) * (1000 / 60)  # [1week, 1/2day, 1hour]
            # betas_recip = np.array([7*2, 1, 1/12]) * (1000 / 60)  # [2week, 1day, 2hour]
            betas_recip = np.array([7, 2, 1 / 4]) * (1000 / 60)  # [1week, 2days, 6 hour]
            betas = np.reciprocal(betas_recip)
        if dataset == "FacebookFiltered":
            days = (1196972372-1168985687)/60/60/24 # dataset lasted for (324 days)
            betas_recip = np.array([2*7, 2, 1 / 4]) * (1000 / days)  # [2week, 2days, 6 hour]
            betas = np.reciprocal(betas_recip)
        for K in range(1,11):
            print("fit refine at K=", K)
            # run one iteration of refinement algorithm
            start_fit_time = time.time()
            # sp_tup, ref_tup, message, ref_time = model_fit_refine_kernel_sum_relative(1, n_alpha, events_dict_train, n_nodes_train, K,
            #                                                                 end_time_train, betas, batch=ref_batch)
            sp_tup, ref_tup, message = model_fit_refine_kernel_sum_exact(MAX_ITER, n_alpha, events_dict_train, n_nodes_train,
                                                                                      K, end_time_train, betas, batch=ref_batch)
            end_fit_time = time.time()
            time_to_fit = end_fit_time - start_fit_time

            # spectral clustering fit results
            nodes_mem_train_sp, fit_param_sp, ll_train_sp, n_events_train = sp_tup
            node_mem_all_sp = MBHP.assign_node_membership_for_missing_nodes(nodes_mem_train_sp, nodes_not_in_train)
            ll_all_sp, n_events_all = MBHP.model_LL_kernel_sum_external(fit_param_sp, events_dict_all, node_mem_all_sp, K, end_time_all)
            ll_all_event_sp = ll_all_sp / n_events_all
            ll_train_event_sp = ll_train_sp / n_events_train
            ll_test_event_sp = (ll_all_sp - ll_train_sp) / (n_events_all - n_events_train)
            print("No Refinement fit time = ", time_to_fit)
            print(f"spectral:\ttrain={ll_train_event_sp:.3f}\tall={ll_all_event_sp:.3f}\ttest={ll_test_event_sp:.3f}")
            # analyze_block(nodes_mem_train_sp, K, id_node_map_train)

            # refinement fit results
            nodes_mem_train_ref, fit_param_ref, ll_train_ref, num_events = ref_tup
            nodes_mem_all_ref = MBHP.assign_node_membership_for_missing_nodes(nodes_mem_train_ref, nodes_not_in_train)
            ll_all_ref, n_events_all = MBHP.model_LL_kernel_sum_external(fit_param_ref, events_dict_all, nodes_mem_all_ref, K, end_time_all)
            ll_all_event_ref = ll_all_ref / n_events_all
            ll_train_event_ref = ll_train_ref / n_events_train
            ll_test_event_ref = (ll_all_ref - ll_train_ref) / (n_events_all - n_events_train)
            print(f"\nref:\ttrain={ll_train_event_ref:.3f}\tall={ll_all_event_ref:.3f}\ttest={ll_test_event_ref:.3f}")
            # analyze_block(nodes_mem_train_ref, K, id_node_map_train)

            # save results
            results_dict = {}
            results_dict["fit_param_ref"] = fit_param_ref
            results_dict["node_mem_train_ref"] = nodes_mem_train_ref
            results_dict["node_mem_all_ref"] = nodes_mem_all_ref
            results_dict["ll_train_ref"] = ll_train_event_ref
            results_dict["ll_all_ref"] = ll_all_event_ref
            results_dict["ll_test_ref"] = ll_test_event_ref
            results_dict["max_iter"] = MAX_ITER
            results_dict["fit_param_sp"] = fit_param_sp
            results_dict["node_mem_train_sp"] = nodes_mem_train_sp
            results_dict["node_mem_all_sp"] = node_mem_all_sp
            results_dict["ll_train_sp"] = ll_train_event_sp
            results_dict["ll_all_sp"] = ll_all_event_sp
            results_dict["ll_test_sp"] = ll_test_event_sp
            results_dict["message"] = message
            # results_dict["ref_time"] = ref_time
            results_dict["n_classes"] = K
            results_dict["fit_time(s)"] = time_to_fit
            results_dict["train_end_time"] = end_time_train
            results_dict["all_end_time"] = end_time_all
            # open file in write+binary mode
            if docker:
                path = f"/data/{dataset}/6alpha_KernelSum_Ref_batch/"  # when called from docker
            else:
                path = f'/shared/Results/MultiBlockHawkesModel/{dataset}/6alpha_KernelSum_Ref_batch/'
            if dataset == "FacebookFiltered":
                if docker:
                    path = f'/result/{dataset}'
                else:
                    path = f'/shared/Results/MultiBlockHawkesModel/{dataset}'
            results_path = f"{path}/no_ref"
            file_name = f"k_{K}.p"
            pickle_file_name = f"{results_path}/{file_name}"
            with open(pickle_file_name, 'wb') as f:
                pickle.dump(results_dict, f)
#%% simulate and count motifs from refinement fit
    if SIM_COUNT_MOTIF_REF:
        SIM = True
        nRuns = 15
        all = True
        motif_delta = 45  # week
        if dataset == "Enron":
            motif_delta = 100  # week

        # fit parameters path and save results path
        if docker:
            path1 = f"/data/{dataset}/6alpha_KernelSum_Ref_batch"  # when called from docker
            path2 = f'/data/MotifCounts/{dataset}'
        else:
            path1 = f"/shared/Results/MultiBlockHawkesModel/{dataset}/6alpha_KernelSum_Ref_batch"
            path2 = f'/shared/Results/MultiBlockHawkesModel/MotifCounts/{dataset}'
        param_path = f"{path1}/2week1day2hour" # 2week1day2hour
        results_path = f"{path2}/2week1day2hour_all" # 1week1_2day1hour


        # ######## run motif counts on dataset
        # dataset_recip, dataset_trans, dataset_motif_week = MBHP.cal_recip_trans_motif(events_dict_train, n_nodes_train, motif_delta,
        #                                                          f"week_{dataset}_noise", save=True)
        # dataset_n_events_train = OneBlockFit.cal_num_events_2(events_dict_train)
        # ####### read networks recip, trans, motifs count from saved pickle
        with open(f"Datasets_motif_counts/week_{dataset}_counts.p", 'rb') as f:
            Mcounts = pickle.load(f)
        dataset_motif_week = Mcounts["dataset_motif"]
        dataset_recip = Mcounts["dataset_recip"]
        dataset_trans = Mcounts["dataset_trans"]
        dataset_n_events_train = Mcounts["dataset_n_events"]
        print(f"{dataset}: reciprocity={dataset_recip:.4f}, transitivity={dataset_trans:.4f}, #events:{dataset_n_events_train}")

        for K in range(3,11):
            print("\nsimulate at K=", K)
            file_name = f"k_{K}.p"
            with open(f"{param_path}/{file_name}", 'rb') as f:
                results_dict = pickle.load(f)
            # refinement parameters
            fit_param_ref = results_dict["fit_param_ref"]
            nodes_mem_train_ref = results_dict["node_mem_train_ref"]
            # spectral clustering parameters
            fit_param_sp = results_dict["fit_param_sp"]
            nodes_mem_train_sp = results_dict["node_mem_train_sp"]

            # simulate from saved fit parameters
            T_sim = end_time_train # only train dataset

            #  Refinement simulation
            if SIM:
                _, block_count = np.unique(nodes_mem_train_ref, return_counts=True)
                block_prob_ref = block_count / sum(block_count)
                sim_motif_avg_week = np.zeros((6, 6))
                sim_motif_all_week = np.zeros((nRuns,6,6))
                sim_n_events_avg, sim_trans_avg, sim_recip_avg = 0, 0, 0
                for run in range(nRuns):
                    # simulate using fitted parameters
                    print("simulation ", run)
                    events_dict_sim, _ = MBHP.simulate_sum_kernel_model(fit_param_ref, n_nodes_train, K, block_prob_ref, T_sim)
                    n_evens_sim = OneBlockFit.cal_num_events_2(events_dict_sim)
                    recip_sim, trans_sim, sim_motif_week = MBHP.cal_recip_trans_motif(events_dict_sim, n_nodes_train, motif_delta)
                    sim_motif_all_week[run,:,:] = sim_motif_week
                    sim_motif_avg_week += sim_motif_week
                    print(f"n_events={n_evens_sim}, recip={recip_sim:.4f}, trans={trans_sim:.4f}")
                    sim_recip_avg += recip_sim
                    sim_trans_avg += trans_sim
                    sim_n_events_avg += n_evens_sim
                # simulation runs at a certain K is done
                sim_motif_avg_week /= nRuns
                sim_motif_median_week = np.median(sim_motif_all_week, axis=0)
                sim_recip_avg /= nRuns
                sim_trans_avg /= nRuns
                sim_n_events_avg /= nRuns

                # calculate MAPE
                mape = 100/36*np.sum(np.abs(sim_motif_avg_week - dataset_motif_week)/dataset_motif_week)

                # save results
                results_dict = {}
                results_dict["K"] = K
                results_dict["nRuns"] = nRuns
                results_dict["parameters"] = fit_param_ref
                results_dict["motif_delta"] = motif_delta
                results_dict["dataset_motif_week"] = dataset_motif_week
                results_dict["dataset_recip"] = dataset_recip
                results_dict["dataset_trans"] = dataset_trans
                results_dict["dataset_n_events"] = dataset_n_events_train
                results_dict["sim_motif_avg_week"] = sim_motif_avg_week
                results_dict["sim_motif_all_week"] = sim_motif_all_week
                results_dict["sim_motif_median_week"] = sim_motif_median_week
                results_dict["sim_recip_avg"] = sim_recip_avg
                results_dict["sim_trans_avg"] = sim_trans_avg
                results_dict["sim_n_events_avg"] = sim_n_events_avg
                results_dict["mape"] = mape

                print(np.asarray(results_dict["dataset_motif_week"], dtype=int))
                print("")
                print(np.asarray(results_dict["sim_motif_avg_week"], dtype=int))
                print("")
                print(np.asarray(results_dict["sim_motif_median_week"], dtype=int))


                file_name = f"k{K}.p"
                pickle_file_name = f"{results_path}/{file_name}"
                with open(pickle_file_name, 'wb') as f:
                    pickle.dump(results_dict, f)


#%% Temporary
    if SIM_COUNT_MOTIF_TEMP:
        SIM = True
        nRuns = 15
        all = True
        motif_delta = 45  # week
        if dataset == "Enron":
            motif_delta = 100  # week

        # fit parameters path and save results path
        path1 = f"/shared/Results/MultiBlockHawkesModel/{dataset}"
        path2 = f'/shared/Results/MultiBlockHawkesModel/MotifCounts/{dataset}'

        param_path = f"{path1}/no_ref_2alpha" # 2week1day2hour
        results_path = f"{path2}/no_ref_alpha2" # 1week1_2day1hour


        with open(f"Datasets_motif_counts/week_{dataset}_counts.p", 'rb') as f:
            Mcounts = pickle.load(f)
        dataset_motif_week = Mcounts["dataset_motif"]
        dataset_recip = Mcounts["dataset_recip"]
        dataset_trans = Mcounts["dataset_trans"]
        dataset_n_events_train = Mcounts["dataset_n_events"]
        print(f"{dataset}: reciprocity={dataset_recip:.4f}, transitivity={dataset_trans:.4f}, #events:{dataset_n_events_train}")

        for K in range(1,11):
            print("\nsimulate at K=", K)
            file_name = f"k_{K}.p"
            with open(f"{param_path}/{file_name}", 'rb') as f:
                results_dict = pickle.load(f)
            # refinement parameters
            fit_param = results_dict["fit_param"]
            nodes_mem_train = results_dict["node_mem_train"]

            # simulate from saved fit parameters
            T_sim = end_time_train # only train dataset

            #  Refinement simulation
            if SIM:
                _, block_count = np.unique(nodes_mem_train, return_counts=True)
                block_prob_ref = block_count / sum(block_count)
                sim_motif_avg_week = np.zeros((6, 6))
                sim_motif_all_week = np.zeros((nRuns,6,6))
                sim_n_events_avg, sim_trans_avg, sim_recip_avg = 0, 0, 0
                for run in range(nRuns):
                    # simulate using fitted parameters
                    print("simulation ", run)
                    events_dict_sim, _ = MBHP.simulate_sum_kernel_model(fit_param, n_nodes_train, K, block_prob_ref, T_sim)
                    n_evens_sim = OneBlockFit.cal_num_events_2(events_dict_sim)
                    recip_sim, trans_sim, sim_motif_week = MBHP.cal_recip_trans_motif(events_dict_sim, n_nodes_train, motif_delta)
                    sim_motif_all_week[run,:,:] = sim_motif_week
                    sim_motif_avg_week += sim_motif_week
                    print(f"n_events={n_evens_sim}, recip={recip_sim:.4f}, trans={trans_sim:.4f}")
                    sim_recip_avg += recip_sim
                    sim_trans_avg += trans_sim
                    sim_n_events_avg += n_evens_sim
                # simulation runs at a certain K is done
                sim_motif_avg_week /= nRuns
                sim_motif_median_week = np.median(sim_motif_all_week, axis=0)
                sim_recip_avg /= nRuns
                sim_trans_avg /= nRuns
                sim_n_events_avg /= nRuns

                # calculate MAPE
                mape = 100/36*np.sum(np.abs(sim_motif_avg_week - dataset_motif_week)/dataset_motif_week)

                # save results
                results_dict = {}
                results_dict["K"] = K
                results_dict["nRuns"] = nRuns
                results_dict["parameters"] = fit_param
                results_dict["motif_delta"] = motif_delta
                results_dict["dataset_motif_week"] = dataset_motif_week
                results_dict["dataset_recip"] = dataset_recip
                results_dict["dataset_trans"] = dataset_trans
                results_dict["dataset_n_events"] = dataset_n_events_train
                results_dict["sim_motif_avg_week"] = sim_motif_avg_week
                results_dict["sim_motif_all_week"] = sim_motif_all_week
                results_dict["sim_motif_median_week"] = sim_motif_median_week
                results_dict["sim_recip_avg"] = sim_recip_avg
                results_dict["sim_trans_avg"] = sim_trans_avg
                results_dict["sim_n_events_avg"] = sim_n_events_avg
                results_dict["mape"] = mape

                print(np.asarray(results_dict["dataset_motif_week"], dtype=int))
                print("")
                print(np.asarray(results_dict["sim_motif_avg_week"], dtype=int))
                print("")
                print(np.asarray(results_dict["sim_motif_median_week"], dtype=int))


                file_name = f"k{K}.p"
                pickle_file_name = f"{results_path}/{file_name}"
                with open(pickle_file_name, 'wb') as f:
                    pickle.dump(results_dict, f)