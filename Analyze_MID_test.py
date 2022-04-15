# -*- coding: utf-8 -*-
"""
Created on Wed May 19 13:39:53 2021

@author: kevin
"""
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt


from spectral_clustering import spectral_cluster1
from utils_sum_betas_bp import cal_num_events
from refinement_alg import model_fit_refine_kernel_sum_exact
import MultiBlockFit as MBHP
from Read_results import analyze_block
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import networkx as nx



day_second_scale = 24 * 60 * 60 # 24 hour/day * 60 min/hour * 60 sec/min
week_second_scale = 7 * day_second_scale
scale_1000 = 1000

docker = False

""" model choice and dataset """
# kernel = "single"
kernel = "sum"
n_alpha = 2
std_hour = True
timestamp_scale = scale_1000

""" Spectral Clustering """
normalize_z = True
multiply_s_sqrt = True
n_init = 500


if timestamp_scale == day_second_scale:
    beta = 1/180 # half a year
    betas = np.array([1/365, 1/30, 2]) # [1 year, 1 month, half a day] --> 1 day scaling
elif timestamp_scale == week_second_scale:
    beta = 1 / 26  # half a year
    betas = np.array([1/52, 1/4, 7]) # [1 year, 1 month, 1 day] --> 1 week scaling
    # betas = np.array([1 /26, 1 / 4, 7 * 2])  # [6 monthes, 1 month, half a day] --> 1 week scaling
else: # [1:1000] scale
    # betas_recip = np.array([4*30, 30, 1]) * (1000 / 8380)  # [4 month, 1 month, 1 day]
    # betas_recip = np.array([2*30, 7, 1/12]) * (1000 / 8380)  # [2 month, 1week, 2 hour]
    # betas_recip = np.array([30, 7, 1]) * (1000 / 8380)  # [1 month, 1 week, 1 day]
    betas_recip = np.array([2 * 30, 2*7, 1 / 2]) * (1000 / 8380)  # [2 month, 2 week, 12 hours]
    betas = np.reciprocal(betas_recip)

""" fitting """
# block_range = []    # no fitting tests
block_range = list(range(1,11))

""" simulation """
sim_and_motif = False

""" fit + refinement algorithm """
# block_range_ref = list(range(1,11))
block_range_ref = []

""" simulation from refinement fit """
simulate_ref = False


#%% helper functions
def load_data_train_all(dnx_pickle_file_name, split_ratio=0.8, scale=7 * 24 * 60 * 60 ,remove_small_comp=False):
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
    print("duration = ", int(duration/(60*60*24)), " days")

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

def get_node_id_maps(node_set):
    nodes = list(node_set)
    nodes.sort()

    node_id_map = {}
    id_node_map = {}
    for i, n in enumerate(nodes):
        node_id_map[n] = i
        id_node_map[i] = n

    return node_id_map, id_node_map

def plot_kernel(alpha, betas, C, time_range):
    lambda_sum = []
    for t in time_range:
        lambda_sum.append(alpha * np.sum(betas * C * np.exp(-t * betas)))
    plt.figure()
    plt.plot(time_range, lambda_sum, color='red', label=f"betas1={betas}")
    plt.xlabel("t(s)")
    plt.ylabel("lambda(t)")
    plt.yscale('log')
    plt.title('sum of kernels C=[0.33, 0.33, 0.34] - y-log scale ')
    plt.legend()
    plt.grid(True)
    plt.show()

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
    # remove countries in small componenets
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
    # " - week scale" if timestamp_scale == week_second_scale else " - day scale"
    print("Load MID dataset - std = 1 ", "hour" if std_hour else "sec", ", [1:1000] scale")
    if std_hour:
        dnx_pickle_file_name = './storage/datasets/MID/MID_std1hour.p'
    else:
        dnx_pickle_file_name = './storage/datasets/MID/MID_std1sec.p'

    train_tup, all_tup, nodes_not_in_train = load_data_train_all(dnx_pickle_file_name, split_ratio=0.8,
                                                                 scale=timestamp_scale, remove_small_comp=True)
    events_dict_train, T_train, n_nodes_train, n_events_train, id_node_map_train = train_tup
    events_dict_all, T_all, n_nodes_all, n_events_all, id_node_map_all = all_tup
    dataset = "MID"

    agg_adj = MBHP.event_dict_to_aggregated_adjacency(n_nodes_train, events_dict_train)


#%% Fit multiblock Hawkes using sum of kernels
    if len(block_range) !=0:
        print(f"Fit {n_alpha}-alpha ",
              f"sum of kernel at betas={betas}" if kernel=="sum" else f"single beta={beta}")
    save_param_fit = True
    for n_classes in block_range:
        start_fit_time = time.time()
        # run spectral clustering on train dataset
        node_mem_train = spectral_cluster1(agg_adj, n_classes, normalize_z=normalize_z, multiply_s=multiply_s_sqrt, n_kmeans_init=n_init)
        node_mem_all = MBHP.assign_node_membership_for_missing_nodes(node_mem_train, nodes_not_in_train)
        # MBHP.plot_adj(agg_adj, node_membership, n_classes, s=f"MID K={n_classes}")
        # analyze_block(node_membership, n_classes, id_node_map)

        # fit model and calculate log-likelihood score
        if kernel == "sum":
            params_est, ll_t, n_events_t = MBHP.model_fit_kernel_sum(n_alpha, events_dict_train, node_mem_train, n_classes, T_train, betas)
        else:
            params_est, ll_t, n_events_t = MBHP.model_fit_single_beta(n_alpha, events_dict_train, node_mem_train, n_classes, T_train, betas)
        end_fit_time = time.time()
        # calculate log-likelihood
        if kernel == "sum":
            ll_all, n_events_all = MBHP.model_LL_kernel_sum_external(params_est, events_dict_all, node_mem_all, n_classes, T_all)
        else:
            ll_all, n_events_all = MBHP.model_LL_single_beta_external(params_est, events_dict_all, node_mem_all, n_classes, T_all)
        ll_all_event = ll_all / n_events_all
        ll_train_event = ll_t / n_events_t
        ll_test_event = (ll_all - ll_t) / (n_events_all - n_events_t)
        time_to_fit = end_fit_time - start_fit_time
        print(f"K={n_classes}:\ttrain={ll_train_event:.3f}\tall={ll_all_event:.3f}\ttest={ll_test_event:.3f}\ttime={time_to_fit/60:.2f}")
        # print(f"{ll_train_event:.3f}\t{ll_all_event:.3f}\t{ll_test_event:.3f}\t{time_to_fit/60:.2f}")

        # save results
        if save_param_fit:
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
            if kernel == "sum":
                results_dict["C"] = params_est[-2]
                results_dict["betas"] = betas
            else:
                results_dict["beta"] = beta
            results_dict["n_classes"] = n_classes
            results_dict["node_mem_train"] = node_mem_train
            results_dict["node_mem_all"] = node_mem_all
            results_dict["id_node_map_train"] = id_node_map_train
            results_dict["id_node_map_all"] = id_node_map_all
            results_dict["ll_train"] = ll_train_event
            results_dict["ll_all"] = ll_all_event
            results_dict["ll_test"] = ll_test_event
            results_dict["fit_time(s)"] = time_to_fit
            results_dict["train_end_time"] = T_train
            results_dict["all_end_time"] = T_all
            results_dict["timestamp_scale"] = timestamp_scale
            results_dict["SC_normalization"] = normalize_z
            results_dict["SC_multuply_s^(1/2)"] = multiply_s_sqrt
            # open file in write+binary mode
            results_path = f'/shared/Results/MultiBlockHawkesModel/MID/no_ref_{n_alpha}alpha'
            # results_path = "/data"  #when called from docker
            file_name = f"k_{n_classes}.p"
            pickle_file_name = f"{results_path}/{file_name}"
            with open(pickle_file_name, 'wb') as f:
                pickle.dump(results_dict, f)

#%% simulation and motif counts
    if sim_and_motif:
        K_sim = 14
        simulate = True
        save_motif_counts = True
        if timestamp_scale == week_second_scale:
            motif_delta = 4
        else:
            motif_delta = 7 * 4

        # # # calculate network reciprocity, transitivity, motifs
        # motifs = MBHP.get_motifs()
        # recip, trans, dataset_motif = MBHP.cal_recip_trans_motif(events_dict_all, n_nodes_all, motif_delta,
        #                                                          f"all_{dataset}_4week_scale1day", save=True)
        # n_events = n_events_all

        # # read networks recip, trans, motifs count from saved pickle
        with open(f"Datasets_motif_counts/all_MID_4week_scale1day_counts.p", 'rb') as f:
            Mcounts = pickle.load(f)
        dataset_motif = Mcounts["dataset_motif"]
        recip = Mcounts["dataset_recip"]
        trans = Mcounts["dataset_trans"]
        n_events = Mcounts["dataset_n_events"]
        print(f"{dataset}: reciprocity={recip:.4f}, transitivity={trans:.4f}, #events:{n_events}")

        # read saved paramters fit results
        if docker:
            res_path = f"/data/{dataset}/6alpha_KernelSum2_std1hour_scale1day"    # docker
        else:
            res_path = f"/shared/Results/MultiBlockHawkesModel/{dataset}/6alpha_KernelSum2_std1hour_scale1day"

        file_name = f"k_{K_sim}.p"
        with open(f"{res_path}/{file_name}", 'rb') as f:
            res = pickle.load(f)
        if n_alpha == 6:
            params_est = (res["mu"], res["alpha_n"], res["alpha_r"], res["alpha_br"], res["alpha_gr"], res["alpha_al"],
                          res["alpha_alr"], res["C"], res["betas"])
        elif n_alpha == 4:
            params_est = (res["mu"], res["alpha_n"], res["alpha_r"], res["alpha_br"], res["alpha_gr"], res["C"], res["betas"])
        node_mem_train = res["node_mem_train"]
        node_mem_all = res["node_mem_all"]
        betas = res["betas"]

        analyze_block(node_mem_all, K_sim, id_node_map_all)

        #  nodes/block probabilities
        _, block_count = np.unique(node_mem_all, return_counts=True)
        block_prob = block_count / sum(block_count)

        if simulate:
            # simulate from estimated parameters
            T_sim = T_all
            nRuns = 15
            sim_motif_avg = np.zeros((6, 6))
            sim_n_events_avg, sim_trans_avg, sim_recip_avg = 0, 0, 0
            for run in range(nRuns):
                # simulate using fitted parameters
                print("simulation ", run)
                events_dict_sim, _ = MBHP.simulate_sum_kernel_model(params_est, n_nodes_all, K_sim, block_prob, T_sim)
                n_evens_sim = cal_num_events(events_dict_sim)
                recip_sim, trans_sim, sim_motif_run = MBHP.cal_recip_trans_motif(events_dict_sim, n_nodes_all, motif_delta)
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
                results_dict["K"] = K_sim
                results_dict["betas"] = betas
                results_dict["nRuns"] = nRuns
                results_dict["parameters"] = params_est
                results_dict["dataset_motif"] = dataset_motif
                results_dict["dataset_recip"] = recip
                results_dict["dataset_trans"] = trans
                results_dict["dataset_n_events"] = n_events
                results_dict["sim_motif_avg"] = sim_motif_avg
                results_dict["sim_recip_avg"] = sim_recip_avg
                results_dict["sim_trans_avg"] = sim_trans_avg
                results_dict["sim_n_events_avg"] = sim_n_events_avg

                print(np.asarray(results_dict["dataset_motif"], dtype=int))
                print("")
                print(np.asarray(results_dict["sim_motif_avg"], dtype=int))

                # open file in write+binary mode
                if docker:
                    results_path = f"/data/MotifCounts/{dataset}"  # docker
                else:
                    results_path = f'/shared/Results/MultiBlockHawkesModel/MotifCounts/{dataset}'

                file_name = f"6alpha_KernelSum_{dataset}_on_k_{K_sim}_scale1day_delta4week.p"
                pickle_file_name = f"{results_path}/{file_name}"
                with open(pickle_file_name, 'wb') as f:
                    pickle.dump(results_dict, f)

#%% refinement - implemented for sum of kernel versions
    for K in block_range_ref:
        ref_batch = True
        MAX_ITER = 0
        print("fit refine at K=", K)
        # run one iteration of refinement algorithm

        sp_tup, ref_tup, message = model_fit_refine_kernel_sum_exact(events_dict_train, n_nodes_train, T_train, K,
                                                                     betas, n_alpha, MAX_ITER)

        # spectral clustering fit results
        nodes_mem_train_sp, fit_param_sp, ll_train_sp, n_events_train, fit_time_sp = sp_tup
        node_mem_all_sp = MBHP.assign_node_membership_for_missing_nodes(nodes_mem_train_sp, nodes_not_in_train)
        ll_all_sp, n_events_all = MBHP.model_LL_kernel_sum_external(fit_param_sp, events_dict_all, node_mem_all_sp, K, T_all)
        ll_all_event_sp = ll_all_sp / n_events_all
        ll_train_event_sp = ll_train_sp / n_events_train
        ll_test_event_sp = (ll_all_sp - ll_train_sp) / (n_events_all - n_events_train)
        print(f"spectral:\ttrain={ll_train_event_sp:.3f}\tall={ll_all_event_sp:.3f}\ttest={ll_test_event_sp:.3f}")
        # analyze_block(nodes_mem_train_sp, K, id_node_map_train)

        # refinement fit results
        nodes_mem_train_ref, fit_param_ref, ll_train_ref, num_events, fit_time_ref = ref_tup
        nodes_mem_all_ref = MBHP.assign_node_membership_for_missing_nodes(nodes_mem_train_ref, nodes_not_in_train)
        ll_all_ref, n_events_all = MBHP.model_LL_kernel_sum_external(fit_param_ref, events_dict_all, nodes_mem_all_ref, K, T_all)
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

        results_dict["fit_param_sp"] = fit_param_sp
        results_dict["node_mem_train_sp"] = nodes_mem_train_sp
        results_dict["node_mem_all_sp"] = node_mem_all_sp
        results_dict["ll_train_sp"] = ll_train_event_sp
        results_dict["ll_all_sp"] = ll_all_event_sp
        results_dict["ll_test_sp"] = ll_test_event_sp
        results_dict["message"] = message
        results_dict["n_classes"] = K
        results_dict["id_node_map_train"] = id_node_map_train
        results_dict["id_node_map_all"] = id_node_map_all
        results_dict["fit_time_sp(s)"] = fit_time_sp
        results_dict["fit_time_ref(s)"] = fit_time_ref
        results_dict["train_end_time"] = T_train
        results_dict["all_end_time"] = T_all
        results_dict["timestamp_scale"] = timestamp_scale
        results_dict["SC_normalization"] = normalize_z
        results_dict["SC_multuply_s^(1/2)"] = multiply_s_sqrt
        # open file in write+binary mode
        if docker:
            path = "/data/MID/6alpha_KernelSum_Ref_batch"
        else:
            path = '/shared/Results/MultiBlockHawkesModel/MID/6alpha_KernelSum_Ref_batch'
        results_path = f'{path}/no_ref'
        file_name = f"k_{K}.p"
        pickle_file_name = f"{results_path}/{file_name}"
        with open(pickle_file_name, 'wb') as f:
            pickle.dump(results_dict, f)

#%% simulate from refinement results
    if simulate_ref:
        simulate_ref_fit = True

        # simulate and coun motifs
        motif_delta_month = 4
        T_sim = T_train
        nRuns = 10

        # ##### run motif counts on dataset
        # dataset_recip, dataset_trans, dataset_motif_month = MBHP.cal_recip_trans_motif(events_dict_train, n_nodes_train, motif_delta_month,
        #                                                                               f"month_{dataset}", save=True)
        # dataset_n_events_train = cal_num_events(events_dict_train)

        # ##### read networks recip, trans, motifs count from saved pickle
        with open(f"Datasets_motif_counts/month_MID_counts.p", 'rb') as f:
            Mcounts = pickle.load(f)
        dataset_motif_month = Mcounts["dataset_motif"]
        recip = Mcounts["dataset_recip"]
        trans = Mcounts["dataset_trans"]
        n_events = Mcounts["dataset_n_events"]
        print(f"{dataset}: reciprocity={recip:.4f}, transitivity={trans:.4f}, #events:{n_events}")

        # paramters and results path
        if docker:
            path1 = "/data/MID/6alpha_KernelSum_Ref_batch"
            path2 = "/data/MotifCounts/MID"
        else:
            path1 = "/shared/Results/MultiBlockHawkesModel/MID/6alpha_KernelSum_Ref_batch"
            path2 = '/shared/Results/MultiBlockHawkesModel/MotifCounts/MID'
        par_path = f"{path1}/2month1week2hour" # 6month1month1_2day # 4month1month1day
        results_path = f"{path2}/2month1week2hour"

        for K in range(3, 11):
            print("\n\nsimulation at K=", K)
            file_name = f"k_{K}.p"
            with open(f"{par_path}/{file_name}", 'rb') as f:
                results_dict = pickle.load(f)
            # refinement parameters
            fit_param_ref = results_dict["fit_param_ref"]
            nodes_mem_train_ref = results_dict["node_mem_train_ref"]
            # spectral clustering parameters
            fit_param_sp = results_dict["fit_param_sp"]
            nodes_mem_train_sp = results_dict["node_mem_train_sp"]

            #  Refinement
            if simulate_ref_fit:
                _, block_count = np.unique(nodes_mem_train_ref, return_counts=True)
                block_prob_ref = block_count / sum(block_count)
                sim_motif_avg_month = np.zeros((6, 6))
                sim_motif_all_month = np.zeros((nRuns, 6, 6))
                sim_n_events_avg, sim_trans_avg, sim_recip_avg = 0, 0, 0
                for run in range(nRuns):
                    # simulate using fitted parameters
                    print("simulation ", run)
                    events_dict_sim, _ = MBHP.simulate_sum_kernel_model(fit_param_ref, n_nodes_train, K, block_prob_ref, T_sim)
                    n_evens_sim = cal_num_events(events_dict_sim)
                    recip_sim, trans_sim, sim_motif_month = MBHP.cal_recip_trans_motif(events_dict_sim, n_nodes_train, motif_delta_month)
                    sim_motif_avg_month += sim_motif_month
                    sim_motif_all_month[run, :, :] = sim_motif_month
                    print(f"n_events={n_evens_sim}, recip={recip_sim:.4f}, trans={trans_sim:.4f}")
                    sim_recip_avg += recip_sim
                    sim_trans_avg += trans_sim
                    sim_n_events_avg += n_evens_sim
                # simulation runs at a certain K is done
                sim_motif_avg_month /= nRuns
                sim_recip_avg /= nRuns
                sim_trans_avg /= nRuns
                sim_n_events_avg /= nRuns
                sim_motif_median_month = np.median(sim_motif_all_month, axis=0)

                # calculate MAPE
                mape = 100 / 36 * np.sum(np.abs(sim_motif_avg_month - dataset_motif_month) / dataset_motif_month)

                # save results
                results_dict = {}
                results_dict["K"] = K
                results_dict["betas"] = fit_param_ref[-1]
                results_dict["nRuns"] = nRuns
                results_dict["parameters"] = fit_param_ref
                results_dict["dataset_motif_month"] = dataset_motif_month
                results_dict["sim_motif_avg_month"] = sim_motif_avg_month
                results_dict["sim_motif_all_month"] = sim_motif_all_month
                results_dict["sim_motif_median_month"] = sim_motif_median_month
                results_dict["dataset_recip"] = recip
                results_dict["dataset_trans"] = trans
                results_dict["dataset_n_events"] = n_events
                results_dict["sim_recip_avg"] = sim_recip_avg
                results_dict["sim_trans_avg"] = sim_trans_avg
                results_dict["sim_n_events_avg"] = sim_n_events_avg
                results_dict["mape"] = mape

                print(np.asarray(results_dict["dataset_motif_month"], dtype=int))
                print("")
                print(np.asarray(results_dict["sim_motif_avg_month"], dtype=int))
                # print("")
                # print(np.asarray(results_dict["sim_motif_median_month"], dtype=int))

                file_name = f"k{K}.p"
                pickle_file_name = f"{results_path}/{file_name}"
                with open(pickle_file_name, 'wb') as f:
                    pickle.dump(results_dict, f)
