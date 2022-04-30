# TODO remove saved motif for dataset
# change all path <<-
# TODO remember link prediction experiments should use node_mem_all and events_dict

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import time
import pickle
from scipy import integrate

import utils_fit_sum_betas_model as mulch_utils
from utils_accuracy_tests import cal_recip_trans_motif, calculate_auc
from utils_sum_betas_bp import cal_num_events
from mulch_MID_experiment import load_data_train_all

sys.path.append("./CHIP-Network-Model")
# import generative_model_utils as utils
import model_fitting_utils as chip_utils
import bhm_parameter_estimation as bhm_utils
from dataset_utils import load_enron_train_test, load_reality_mining_test_train
from chip_generative_model import community_generative_model
from bhm_generative_model import block_generative_model



def simulate_count_motif_experiment_chip_bhm(chip, dataset_motif, param, nodes_mem, K, T_sim, motif_delta, n_sim=10,
                                             verbose=False):
    mu, alpha, beta = param
    # simulate and count motifs
    if verbose:
        print("\nSimulation and motif count experiment at K=", K)
    n_nodes = len(nodes_mem)
    _, block_count = np.unique(nodes_mem, return_counts=True)
    block_prob = block_count / sum(block_count)
    sim_motif_avg = np.zeros((6, 6))
    sim_n_events_avg, sim_trans_avg, sim_recip_avg = 0, 0, 0
    for run in range(n_sim):
        # simulate using fitted parameters
        print("simulation ", run)
        # simulate from CHIP model
        if chip:
            _, events_dict_sim = community_generative_model(n_nodes, block_prob, mu, alpha, beta, T_sim)
        # simulate from BHM
        else:
            _, events_dict_sim = block_generative_model(n_nodes, block_prob, mu, alpha, beta, T_sim)
        n_evens_sim = cal_num_events(events_dict_sim)
        recip_sim, trans_sim, sim_motif_month = cal_recip_trans_motif(events_dict_sim, n_nodes, motif_delta)
        sim_motif_avg += sim_motif_month
        print(f"n_events={n_evens_sim}, recip={recip_sim:.4f}, trans={trans_sim:.4f}")
        sim_recip_avg += recip_sim
        sim_trans_avg += trans_sim
        sim_n_events_avg += n_evens_sim
    # simulation runs at a certain K is done
    sim_motif_avg /= n_sim
    sim_recip_avg /= n_sim
    sim_trans_avg /= n_sim
    sim_n_events_avg /= n_sim

    # calculate MAPE - NOTE: just added 1 to avoid division by 0
    mape = 100 / 36 * np.sum(np.abs(sim_motif_avg - (dataset_motif + 1)) / (dataset_motif + 1))

    if verbose:
        print("Actual dataset motifs count at delta=", motif_delta)
        print(np.asarray(dataset_motif, dtype=int))
        print("Simulation average motifs count")
        print(np.asarray(sim_motif_avg, dtype=int))
        print(f"MAPE = {mape:.2f}")

    motif_dict = {}
    motif_dict["K"] = K
    motif_dict["n_simulations"] = n_sim
    motif_dict["parameters"] = param
    motif_dict["motif_delta"] = motif_delta
    motif_dict["sim_motif_avg"] = sim_motif_avg
    motif_dict["sim_recip_avg"] = sim_recip_avg
    motif_dict["sim_trans_avg"] = sim_trans_avg
    motif_dict["sim_n_events_avg"] = sim_n_events_avg
    motif_dict["mape"] = mape
    return motif_dict

def bhm_predict_probs_and_actual(chip, t0, delta, n_nodes, events_dict_all, params_tup, K, node_mem_train):
    # event_dictionaries for block pairs
    bp_events_dict_all = mulch_utils.events_dict_to_events_dict_bp(events_dict_all, node_mem_train, K)
    # number of node pairs per block pair & number of nodes in one block
    bp_M, n_nodes_b = mulch_utils.num_nodes_pairs_per_block_pair(node_mem_train, K)
    bp_mu, bp_alpha, bp_beta = params_tup
    predict = np.zeros((n_nodes, n_nodes))  # Predicted probs that link exists
    # node pairs in same block pair have equal probabilities - store to avoid re-calculations
    bp_predict = [[None]*K for _ in range(K)]  # (K,K) list
    actual = np.zeros((n_nodes, n_nodes))  # actual link
    for u in range(n_nodes):
        for v in range(n_nodes):
            if u != v:
                # blocks of node u and v
                u_b, v_b = node_mem_train[u], node_mem_train[v]
                if bp_predict[u_b][v_b] is None:
                    par = (bp_mu[u_b, v_b], bp_alpha[u_b, v_b], bp_beta[u_b, v_b])
                    # pass timestamps in block pair (u_b, v_b) where time<t0
                    timestamps_less_t0 = []
                    for (x, y) in bp_events_dict_all[u_b][v_b]:
                        xy_timestamps = np.array(bp_events_dict_all[u_b][v_b][(x, y)])
                        xy_timestamps_less_t0 = xy_timestamps[xy_timestamps < t0]
                        timestamps_less_t0.extend(xy_timestamps_less_t0.tolist())
                    if chip:    # CHIP intensity function
                        integral = integrate.quad(chip_uv_intensity, t0, t0 + delta,
                                                  args=(timestamps_less_t0, par), limit=100)[0]
                    else:   # BHM intensity function
                        integral = integrate.quad(bhm_uv_intensity, t0, t0 + delta,
                                                  args=(timestamps_less_t0, par, bp_M[u_b, v_b]), limit=100)[0]
                    predict[u, v] = 1 - np.exp(-integral)
                    # predict[u, v] = (1/bp_M[u_b, v_b]) * (1 - np.exp(-integral))
                    bp_predict[u_b][v_b] = predict[u, v]
                else:
                    predict[u, v] = bp_predict[u_b][v_b]
                # calculate y
                if (u, v) in bp_events_dict_all[u_b][v_b]:
                    uv_times = bp_events_dict_all[u_b][v_b][(u, v)]
                    if len(uv_times[np.logical_and(uv_times >= t0, uv_times <= t0 + delta)]) > 0:
                        actual[u, v] = 1
    return actual, predict

def bhm_uv_intensity(t, timestamps, params, M):
    mu, alpha, beta = params
    timestamps = np.array(timestamps)
    # intensity = mu + alpha * np.sum(np.exp(-beta * (t - timestamps)))
    intensity = (1/M) * (mu + alpha * np.sum(np.exp(-beta * (t - timestamps))))
    return intensity

def chip_uv_intensity(t, timestamps, params):
    mu, alpha, beta = params
    timestamps = np.array(timestamps)
    intensity = mu + alpha * np.sum(np.exp(-beta * (t - timestamps)))
    return intensity

if __name__ == "__main__":

    docker = False
    if docker:
        save_path = f"/data"  # when called from docker
    else:
        save_path = f'/shared/Results/MultiBlockHawkesModel'


    CHIP = True # if chip is false then experiments for BHM
    K_range = range(1, 11)

    """ model fitting """
    fit_model = True # false means read a saved fit
    save_fit = False # remember to set save path


    """ motif simulation """
    motif_experiment = False
    n_motif_simulations = 2
    save_motif = True  # specify path in code

    """ link prediction """
    link_prediction_experiment = True
    save_link = True  # specify path in code


    # # # load Dataset
    dataset = "MID"


    if dataset == "RealityMining" or dataset =="Enron" or dataset=="FacebookFiltered":
        if dataset =="Enron":
            train_tuple, test_tuple, all_tuple, nodes_not_in_train = load_enron_train_test(remove_nodes_not_in_train=False)
            motif_delta = 100
            link_pred_delta = 125 # week and quarter
            motif_delta_txt = 'week'
        elif dataset == "RealityMining":
            train_tuple, test_tuple, all_tuple, nodes_not_in_train = load_reality_mining_test_train(remove_nodes_not_in_train=False)
            link_pred_delta = 60 # should be two weeks
            motif_delta_txt = 'week'
        else:   # Filtered Facebook dataset
            facebook_path = os.path.join(os.getcwd(), "storage", "datasets", "facebook_filtered",
                                         "facebook-wall-filtered.txt")
            train_tuple, all_tuple, nodes_not_in_train = mulch_utils.read_cvs_split_train(facebook_path)
        events_dict_train, n_nodes_train, T_train = train_tuple
        events_dict_all, n_nodes_all, T_all = all_tuple
        n_events_train = cal_num_events(events_dict_train)
        n_events_all = cal_num_events(events_dict_all)
    elif dataset == "MID":
        pickle_file = os.path.join(os.getcwd(), "storage", "datasets", "MID", "MID_std1hour.p")
        train_tup, all_tup, nodes_not_in_train = load_data_train_all(pickle_file)
        events_dict_train, T_train, n_nodes_train, n_events_train, id_node_map_train = train_tup
        events_dict_all, T_all, n_nodes_all, n_events_all, id_node_map_all = all_tup
        motif_delta = 4
        motif_delta_txt = 'month'
        link_pred_delta = 7.15


    for K in K_range:
        if CHIP:
            start_fit_time = time.time()
            # fit CHIP on train dataset
            node_mem_train, bp_mu_t, bp_alpha_t, bp_beta_t, events_dict_bp_train = chip_utils.fit_community_model(
                events_dict_train, n_nodes_train, T_train, K, 0, -1, verbose=False)
            param = (bp_mu_t, bp_alpha_t, bp_beta_t)
            end_time_fit = time.time()
            # Add nodes that were not in train to the largest block
            node_mem_all = chip_utils.assign_node_membership_for_missing_nodes(node_mem_train, nodes_not_in_train)
            # Calculate log-likelihood given the entire dataset
            events_dict_bp_all = mulch_utils.event_dict_to_block_pair_events(events_dict_all, node_mem_all, K)
            ll_all = chip_utils.calc_full_log_likelihood(events_dict_bp_all, node_mem_all, bp_mu_t, bp_alpha_t, bp_beta_t,
                                                         T_all, K)
            # Calculate log-likelihood given the train dataset
            ll_train = chip_utils.calc_full_log_likelihood(events_dict_bp_train, node_mem_train, bp_mu_t, bp_alpha_t, bp_beta_t,
                                                           T_train, K)
            fit_time = end_time_fit - start_fit_time
            ll_all_event = ll_all / n_events_all
            ll_train_event = ll_train / n_events_train
            ll_test_event = (ll_all - ll_train) / (n_events_all - n_events_train)
            print(f"K={K}:\ttrain={ll_train_event:.3f}\tall={ll_all_event:.3f}\ttest={ll_test_event:.3f}")
            # print(f"{ll_train_event:.3f}\t{ll_all_event:.3f}\t{ll_test_event:.3f}\t{fit_time:.3f}")

            if save_fit:
                # save fit parameters
                fit_dict = {}
                fit_dict["param"] = param
                fit_dict["node_mem_train"] = node_mem_train
                fit_dict["node_mem_all"] = node_mem_all
                fit_dict["ll_train"] = ll_train_event
                fit_dict["ll_all"] = ll_all_event
                fit_dict["ll_test"] = ll_test_event
                full_fit_path = f'/{save_path}/CHIP/{dataset}/test'
                pickle_file_name = f"{full_fit_path}/k_{K}.p"
                with open(pickle_file_name, 'wb') as f:
                    pickle.dump(fit_dict, f)

            # simulate from fit parameters and count motifs
            if motif_experiment and dataset != "FacebookFiltered":
                # read saved dataset motif counts
                with open(f"storage/datasets_motif_counts/{motif_delta_txt}_{dataset}_counts.p", 'rb') as f:
                    dataset_motif_dict = pickle.load(f)
                dataset_motif = dataset_motif_dict["dataset_motif"]
                recip = dataset_motif_dict["dataset_recip"]
                trans = dataset_motif_dict["dataset_trans"]
                n_events_train = dataset_motif_dict["dataset_n_events"]
                print(f"{dataset}: reciprocity={recip:.4f}, transitivity={trans:.4f}, #events:{n_events_train}")

                motif_dict = simulate_count_motif_experiment_chip_bhm(CHIP, dataset_motif, param, node_mem_train, K,
                                                                      T_train, motif_delta, n_sim=n_motif_simulations
                                                                      , verbose=False)

                if save_motif:
                    motif_dict["dataset_motif"] = dataset_motif
                    motif_dict["dataset_recip"] = recip
                    motif_dict["dataset_trans"] = trans
                    motif_dict["dataset_n_events"] = n_events_train
                    full_motif_path = f"{save_path}/MotifCounts/CHIP/{dataset}/test"
                    pickle_file_name = f"{full_motif_path}/k{K}.p"
                    with open(pickle_file_name, 'wb') as f:
                        pickle.dump(motif_dict, f)

            # link prediction experiments --> use node_mem_all and events_dict_all
            if link_prediction_experiment and dataset != "FacebookFiltered":
                print("Link Prediction Experiments at delta=", link_pred_delta)
                t0s = np.loadtxt(f"storage/t0/{dataset}_t0.csv", delimiter=',', usecols=1)
                runs = len(t0s)
                auc = np.zeros(runs)
                y_runs = np.zeros((n_nodes_all, n_nodes_all, runs))
                pred_runs = np.zeros((n_nodes_all, n_nodes_all, runs))
                for i, t0 in enumerate(t0s):
                    # t0 = np.random.uniform(low=end_time_train, high=end_time_all - delta, size=None)
                    y_bhm, pred_bhm = bhm_predict_probs_and_actual(CHIP, t0, link_pred_delta, n_nodes_all,
                                                                   events_dict_all, param, K, node_mem_all)
                    y_runs[:, :, i] = y_bhm
                    pred_runs[:, :, i] = pred_bhm
                    auc[i] = calculate_auc(y_bhm, pred_bhm, show_figure=False)
                    print(f'\trun#{i}: auc={auc[i]:.4f}')
                print(f'at K={K}: log-likelihood={fit_dict["ll_test"]:.5f}, AUC-avg={np.average(auc):.5f}, '
                      f'AUC-std{auc.std():.3f}')
                # print(f'{fit_dict["ll_test"]:.5f}\t{K}\t{np.average(auc):.5f}\t{auc.std():.3f}')
                if save_link:
                    pickle_file_name = f"{save_path}/BHM/{dataset}_auc_K_{K}.p"
                    auc_dict = {"t0": t0s, "auc": auc, "avg": np.average(auc), "std": auc.std(), "y__runs": y_runs,
                                "pred_runs": pred_runs}
                    with open(pickle_file_name, 'wb') as f:
                        pickle.dump(auc_dict, f)

        # BHM experiments
        else:
            print("K = ", K)
            try:
                if fit_model:
                    # Fitting the model to the train data
                    node_mem_train, bp_mu_t, bp_alpha_t, bp_beta_t, events_dict_bp_train = bhm_utils.fit_block_model(events_dict_train,
                        n_nodes_train, T_train, K, local_search_max_iter=200, local_search_n_cores=0, verbose=True)
                    param = (bp_mu_t, bp_alpha_t, bp_beta_t)
                    # Add nodes that were not in train to the largest block
                    node_mem_all = chip_utils.assign_node_membership_for_missing_nodes(node_mem_train, nodes_not_in_train)

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

                    if save_fit:
                        fit_dict = {}
                        fit_dict["param"] = param
                        fit_dict["node_mem_train"] = node_mem_train
                        fit_dict["node_mem_all"] = node_mem_all
                        fit_dict["ll_train"] = ll_train_event
                        fit_dict["ll_all"] = ll_all_event
                        fit_dict["ll_test"] = ll_test_event
                        fit_dict["local_search"] = 200
                        full_fit_path = f'/{save_path}/BHM/{dataset}/test'
                        pickle_file_name = f"{full_fit_path}/k_{K}.p"
                        with open(pickle_file_name, 'wb') as f:
                            pickle.dump(fit_dict, f)
                else:
                    # read fit
                    full_fit_path = f'/{save_path}/BHM/{dataset}/test'
                    pickle_file_name = f"{full_fit_path}/k_{K}.p"
                    with open(pickle_file_name, 'rb') as f:
                        fit_dict = pickle.load(f)
                    param = fit_dict["param"]
                    node_mem_train = fit_dict["node_mem_train"]
                    print(f"K={K}:\ttrain={fit_dict['ll_train']:.3f}\tall={fit_dict['ll_all']:.3f}"
                          f"\ttest={fit_dict['ll_test']:.3f}")

                # simulate from fit parameters and count motifs
                if motif_experiment and dataset != "FacebookFiltered":
                    # read saved dataset motif counts
                    with open(f"storage/datasets_motif_counts/{motif_delta_txt}_{dataset}_counts.p", 'rb') as f:
                        dataset_motif_dict = pickle.load(f)
                    dataset_motif = dataset_motif_dict["dataset_motif"]
                    recip = dataset_motif_dict["dataset_recip"]
                    trans = dataset_motif_dict["dataset_trans"]
                    n_events_train = dataset_motif_dict["dataset_n_events"]
                    print(f"{dataset}: reciprocity={recip:.4f}, transitivity={trans:.4f}, #events:{n_events_train}")

                    motif_dict = simulate_count_motif_experiment_chip_bhm(CHIP, dataset_motif, param, node_mem_train, K,
                                                                          T_train, motif_delta,
                                                                          n_sim=n_motif_simulations, verbose=False)
                    if save_motif:
                        motif_dict["dataset_motif"] = dataset_motif
                        motif_dict["dataset_recip"] = recip
                        motif_dict["dataset_trans"] = trans
                        motif_dict["dataset_n_events"] = n_events_train
                        full_motif_path = f"{save_path}/MotifCounts/BHM/{dataset}/test"
                        pickle_file_name = f"{full_motif_path}/k{K}.p"
                        with open(pickle_file_name, 'wb') as f:
                            pickle.dump(motif_dict, f)

                # link prediction experiments --> use node_mem_all and events_dict_all
                if link_prediction_experiment and dataset != "FacebookFiltered":
                    print("Link Prediction Experiments at delta=", link_pred_delta)
                    t0s = np.loadtxt(f"storage/t0/{dataset}_t0.csv", delimiter=',', usecols=1)
                    runs = len(t0s)
                    auc = np.zeros(runs)
                    y_runs = np.zeros((n_nodes_all, n_nodes_all, runs))
                    pred_runs = np.zeros((n_nodes_all, n_nodes_all, runs))
                    for i, t0 in enumerate(t0s):
                        # t0 = np.random.uniform(low=end_time_train, high=end_time_all - delta, size=None)
                        y_bhm, pred_bhm = bhm_predict_probs_and_actual(CHIP, t0, link_pred_delta, n_nodes_all,
                                                                       events_dict_all, param, K, node_mem_all)
                        y_runs[:, :, i] = y_bhm
                        pred_runs[:, :, i] = pred_bhm
                        auc[i] = calculate_auc(y_bhm, pred_bhm, show_figure=False)
                        print(f'\trun#{i}: auc={auc[i]:.4f}')
                    print(f'at K={K}: log-likelihood={fit_dict["ll_test"]:.5f}, AUC-avg={np.average(auc):.5f}, '
                          f'AUC-std{auc.std():.3f}')
                    # print(f'{fit_dict["ll_test"]:.5f}\t{K}\t{np.average(auc):.5f}\t{auc.std():.3f}')
                    if save_link:
                        pickle_file_name = f"{save_path}/BHM/{dataset}_auc_K_{K}.p"
                        auc_dict = {"t0": t0s, "auc": auc, "avg": np.average(auc), "std": auc.std(), "y__runs": y_runs,
                                    "pred_runs": pred_runs}
                        with open(pickle_file_name, 'wb') as f:
                            pickle.dump(auc_dict, f)

            except Exception:
                pass









