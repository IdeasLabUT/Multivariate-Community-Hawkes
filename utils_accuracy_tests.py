# TODO delete those link prediction experiments

import networkx as nx
import numpy as np
from scipy import integrate
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sys
import copy

from utils_sum_betas_bp import cal_num_events
from utils_fit_sum_betas_model import event_dict_to_adjacency
from utils_generate_sum_betas_model import simulate_sum_kernel_model
from dynetworkx import ImpulseDiGraph

sys.path.append("./CHIP-Network-Model")
from dataset_utils import load_enron_train_test, load_reality_mining_test_train
import utils_fit_sum_betas_model as MBHP
from bisect import bisect_left


#%% motif counts functions
def get_motifs():
    motifs = [[((1, 2), (3, 2), (1, 2)), ((1, 2), (3, 2), (2, 1)), ((1, 2), (3, 2), (1, 3)), ((1, 2), (3, 2), (3, 1)),
               ((1, 2), (3, 2), (2, 3)), ((1, 2), (3, 2), (3, 2))],
              [((1, 2), (2, 3), (1, 2)), ((1, 2), (2, 3), (2, 1)), ((1, 2), (2, 3), (1, 3)), ((1, 2), (2, 3), (3, 1)),
               ((1, 2), (2, 3), (2, 3)), ((1, 2), (2, 3), (3, 2))],
              [((1, 2), (3, 1), (1, 2)), ((1, 2), (3, 1), (2, 1)), ((1, 2), (3, 1), (1, 3)), ((1, 2), (3, 1), (3, 1)),
               ((1, 2), (3, 1), (2, 3)), ((1, 2), (3, 1), (3, 2))],
              [((1, 2), (1, 3), (1, 2)), ((1, 2), (1, 3), (2, 1)), ((1, 2), (1, 3), (1, 3)), ((1, 2), (1, 3), (3, 1)),
               ((1, 2), (1, 3), (2, 3)), ((1, 2), (1, 3), (3, 2))],
              [((1, 2), (2, 1), (1, 2)), ((1, 2), (2, 1), (2, 1)), ((1, 2), (2, 1), (1, 3)), ((1, 2), (2, 1), (3, 1)),
               ((1, 2), (2, 1), (2, 3)), ((1, 2), (2, 1), (3, 2))],
              [((1, 2), (1, 2), (1, 2)), ((1, 2), (1, 2), (2, 1)), ((1, 2), (1, 2), (1, 3)), ((1, 2), (1, 2), (3, 1)),
               ((1, 2), (1, 2), (2, 3)), ((1, 2), (1, 2), (3, 2))]]
    return motifs

def cal_recip_trans_motif(events_dict, N, motif_delta, verbose=False,save_name=None):
    adj = event_dict_to_adjacency(N, events_dict)
    net = nx.DiGraph(adj)
    recip = nx.overall_reciprocity(net)
    trans = nx.transitivity(net)
    # create ImpulseDiGraph from network
    G_data = ImpulseDiGraph()
    for (u,v) in events_dict:
        events_list_uv = events_dict[u,v]
        for t in events_list_uv:
            G_data.add_edge(u,v,t)
    if verbose:
        print(f"{1:>10}{2:>10}{3:>10}{4:>10}{5:>10}{6:>10}")
    motifs = get_motifs()
    dataset_motif = np.zeros((6,6),dtype=int)
    for i in range(6):
        for j in range(6):
            dataset_motif[i,j] = G_data.calculate_temporal_motifs(motifs[i][j], motif_delta)
        if verbose:
            print(f"{dataset_motif[i,0]:>10}{dataset_motif[i,1]:>10}{dataset_motif[i,2]:>10}{dataset_motif[i,3]:>10}"
                  f"{dataset_motif[i,4]:>10}{dataset_motif[i,5]:>10}")
    n_events = cal_num_events(events_dict)
    if save_name is not None:
        results_dict = {}
        results_dict["dataset_motif"] = dataset_motif
        results_dict["dataset_recip"] = recip
        results_dict["dataset_trans"] = trans
        results_dict["dataset_n_events"] = n_events
        with open(f"{save_name}.p", 'wb') as fil:
            pickle.dump(results_dict, fil)
    return recip, trans, dataset_motif


def simulate_count_motif_experiment(dataset_motif_tuple, param, nodes_mem, K, T_sim, motif_delta, n_sim=10,
                                    verbose=False):
    dataset_recip, dataset_trans, dataset_motif, dataset_n_events = dataset_motif_tuple
    # simulate and count motifs
    if verbose:
        print("\nSimulation and motif count experiment at K=", K)
    n_nodes = len(nodes_mem)
    _, block_count = np.unique(nodes_mem, return_counts=True)
    block_prob = block_count / sum(block_count)
    sim_motif_avg = np.zeros((6, 6))
    sim_motif_all = np.zeros((n_sim, 6, 6))
    sim_n_events_avg, sim_trans_avg, sim_recip_avg = 0, 0, 0
    for run in range(n_sim):
        # simulate using fitted parameters
        if verbose:
            print("\n\tsimulation#", run)
        events_dict_sim, _ = simulate_sum_kernel_model(param, n_nodes, K, block_prob, T_sim)
        n_evens_sim = cal_num_events(events_dict_sim)
        recip_sim, trans_sim, sim_motif = cal_recip_trans_motif(events_dict_sim, n_nodes,
                                                                motif_delta, verbose)
        sim_motif_avg += sim_motif
        sim_motif_all[run, :, :] = sim_motif
        if verbose:
            print(f"\trecip={recip_sim:.4f}, trans={trans_sim:.4f}, n_events={n_evens_sim}")
        sim_recip_avg += recip_sim
        sim_trans_avg += trans_sim
        sim_n_events_avg += n_evens_sim
    # simulation runs at a certain K is done
    sim_motif_avg /= n_sim
    sim_recip_avg /= n_sim
    sim_trans_avg /= n_sim
    sim_n_events_avg /= n_sim
    sim_motif_median = np.median(sim_motif_all, axis=0)

    # calculate MAPE - NOTE: just added 1 to avoid division by 0
    mape = 100 / 36 * np.sum(np.abs(sim_motif_avg - (dataset_motif + 1)) / (dataset_motif + 1))

    # save results
    results_dict = {}
    results_dict["K"] = K
    results_dict["betas"] = param[-1]
    results_dict["n_simulation"] = n_sim
    results_dict["motif_delta"] = motif_delta
    results_dict["parameters"] = param
    results_dict["dataset_motif"] = dataset_motif
    results_dict["dataset_recip"] = dataset_recip
    results_dict["dataset_trans"] = dataset_trans
    results_dict["dataset_n_events"] = dataset_n_events
    results_dict["sim_motif_avg"] = sim_motif_avg
    results_dict["sim_motif_all"] = sim_motif_all
    results_dict["sim_motif_median"] = sim_motif_median
    results_dict["sim_recip_avg"] = sim_recip_avg
    results_dict["sim_trans_avg"] = sim_trans_avg
    results_dict["sim_n_events_avg"] = sim_n_events_avg
    results_dict["mape"] = mape

    if verbose:
        print("Actual dataset motifs count at delta=", motif_delta)
        print(np.asarray(results_dict["dataset_motif"], dtype=int))
        print("Simulation average motifs count")
        print(np.asarray(results_dict["sim_motif_avg"], dtype=int))
        print(f"MAPE = {mape:.2f}")

    return results_dict

#%% link prediction test functions
def mulch_predict_probs_and_actual(n_nodes, t0, delta, events_dict, params_tup, nodes_mem_all):
    """
    Computes the predicted probability that a link from u to v appears in
    [t, t + delta)
    """
    mu_bp, alpha_n_bp, alpha_r_bp, alpha_br_bp, alpha_gr_bp, alpha_al_bp, alpha_alr_bp, C_bp, betas = params_tup
    K = len(mu_bp)
    # hold timestamps < t0, and compute [t0 - timestamps]
    events_dict_less_t0 = {}
    for u, v in events_dict:
        t_uv = np.array(events_dict[(u, v)])
        # index of timestamps < t0
        index = bisect_left(t_uv, t0)
        events_dict_less_t0[(u, v)] = t0 - t_uv[:index]
    bp_events_dict_less_t0 = MBHP.events_dict_to_events_dict_bp(events_dict_less_t0, nodes_mem_all, K)
    prob_dict = np.zeros((n_nodes, n_nodes))  # Predicted probs that link exists
    y = np.zeros((n_nodes, n_nodes))  # actual link
    for u in range(n_nodes):
        for v in range(n_nodes):
            if u != v:
                u_b, v_b = nodes_mem_all[u], nodes_mem_all[v]
                par = (
                mu_bp[u_b, v_b], alpha_n_bp[u_b, v_b], alpha_r_bp[u_b, v_b], alpha_br_bp[u_b, v_b], alpha_gr_bp[u_b, v_b],
                alpha_al_bp[u_b, v_b], alpha_alr_bp[u_b, v_b], C_bp[u_b, v_b], betas)
                if u_b == v_b:
                    # integral = \
                    #     integrate.quad(mulch_uv_intensity_dia, t0, t0 + delta,
                    #                    args=((u, v), par, bp_events_dict[u_b][v_b], t0), limit=100)[0]
                    integral2 = mulch_uv_intensity_dia_integral(delta, (u, v), par, bp_events_dict_less_t0[u_b][v_b])
                else:
                    # integral = integrate.quad(mulch_uv_intensity_off, t0, t0 + delta,
                    #                           args=((u, v), par, bp_events_dict[u_b][v_b], bp_events_dict[v_b][u_b], t0),
                    #                           limit=100)[0]
                    integral2 = mulch_uv_intensity_off_integral(delta, (u, v), par, bp_events_dict_less_t0[u_b][v_b],
                                                                 bp_events_dict_less_t0[v_b][u_b])

                prob_dict[u, v] = 1 - np.exp(-integral2)

                if (u, v) in events_dict:
                    uv_times = np.array(events_dict[(u, v)])
                    if len(uv_times[np.logical_and(uv_times >= t0, uv_times <= t0 + delta)]) > 0:
                        y[u, v] = 1

    return y, prob_dict

# integral of a diagonal block pair intensity function
def mulch_uv_intensity_dia_integral(delta, uv, params, events_dict):
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, C, betas = params
    u, v = uv
    Q = len(betas)
    integral = mu * delta
    # assume all timestamps in events_dict are less than t0
    for (x, y) in events_dict:
        t0_minus_txy = events_dict[(x, y)]
        exp_sum_q = 0
        for q in range(Q):
            exp_sum_q += C[q] * (np.exp(-betas[q]*delta) - 1) * np.sum(np.exp(-betas[q] * t0_minus_txy))
        if (u, v) == (x, y):  # same node_pair events (alpha_n)
            integral -= alpha_n * exp_sum_q
        elif (v, u) == (x, y):  # reciprocal node_pair events (alpha_r)
            integral -= alpha_r * exp_sum_q
        # br node_pairs events (alpha_br)
        elif u == x and v != y:
            integral -= alpha_br * exp_sum_q
        # gr node_pairs events (alpha_gr)
        elif u == y and v != x:
            integral -= alpha_gr * exp_sum_q
        # alliance np (alpha_al)
        elif v == y and u != x:
            integral -= alpha_al * exp_sum_q
        # alliance reciprocal np (alpha_alr)
        elif v == x and u != y:
            integral -= alpha_alr * exp_sum_q
    return integral
# integral of an off-diagonal block pair intensity function
def mulch_uv_intensity_off_integral(delta, uv, params, events_dict, events_dict_r):
    # assume timestamps in events_dict, events_dict_r are less than t0
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, C, betas = params
    u, v = uv
    Q = len(betas)
    integral = mu * delta
    # loop through node pairs in block pair ab
    for (x, y) in events_dict:
        t0_minus_txy = events_dict[(x, y)]
        exp_sum_q = 0
        for q in range(Q):
            exp_sum_q += C[q] * (np.exp(-betas[q] * delta) - 1) * np.sum(np.exp(-betas[q] * t0_minus_txy))
        # same node_pair events (alpha_n)
        if (u, v) == (x, y):
            integral -= alpha_n * exp_sum_q
        # br node_pairs events (alpha_br)
        elif u == x and v != y:
            integral -= alpha_br * exp_sum_q
        # alliance np (alpha_al)
        elif v == y and u != x:
            integral -= alpha_alr * exp_sum_q
    # loop through node pairs in reciprocal block pair ba
    for (x, y) in events_dict_r:
        t0_minus_txy = events_dict_r[(x, y)]
        exp_sum_q = 0
        for q in range(Q):
            exp_sum_q += C[q] * (np.exp(-betas[q] * delta) - 1) * np.sum(np.exp(-betas[q] * t0_minus_txy))
        # reciprocal node_pair events (alpha_r)
        if (v, u) == (x, y):
            integral -= alpha_r * exp_sum_q
        # gr node_pairs events (alpha_gr)
        elif u == y and v != x:
            integral -= alpha_gr * exp_sum_q
        # alliance reciprocal np (alpha_alr)
        elif v == x and u != y:
            integral -= alpha_alr * exp_sum_q
    return integral


# a diagonal block pair intensity function
def mulch_uv_intensity_dia(t, uv, params, events_dict, t0):
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, C, betas = params
    u, v = uv
    Q = len(betas)
    intensity = mu
    for (x, y) in events_dict:
        xy_timestamps_less_t0 = events_dict[(x, y)][events_dict[(x, y)] < t0]
        exp_sum_q = 0
        for q in range(Q):
            exp_sum_q += C[q] * betas[q] * np.sum(np.exp(-betas[q] * (t - xy_timestamps_less_t0)))
        if (u, v) == (x, y):  # same node_pair events (alpha_n)
            intensity += alpha_n * exp_sum_q
        elif (v, u) == (x, y):  # reciprocal node_pair events (alpha_r)
            intensity += alpha_r * exp_sum_q
        # br node_pairs events (alpha_br)
        elif u == x and v != y:
            intensity += alpha_br * exp_sum_q
        # gr node_pairs events (alpha_gr)
        elif u == y and v != x:
            intensity += alpha_gr * exp_sum_q
        # alliance np (alpha_al)
        elif v == y and u != x:
            intensity += alpha_al * exp_sum_q
        # alliance reciprocal np (alpha_alr)
        elif v == x and u != y:
            intensity += alpha_alr * exp_sum_q
    return intensity
# an off-diagonal block pair intensity function
def mulch_uv_intensity_off(t, uv, params, events_dict, events_dict_r, t0):
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, C, betas = params
    u, v = uv
    Q = len(betas)
    intensity = mu
    # loop through node pairs in block pair ab
    for (x, y) in events_dict:
        xy_timestamps_less_t0 = events_dict[(x, y)][events_dict[(x, y)] < t0]
        exp_sum_q = 0
        for q in range(Q):
            exp_sum_q += C[q] * betas[q] * np.sum(np.exp(-betas[q] * (t - xy_timestamps_less_t0)))
        # same node_pair events (alpha_n)
        if (u, v) == (x, y):
            intensity += alpha_n * exp_sum_q
        # br node_pairs events (alpha_br)
        elif u == x and v != y:
            intensity += alpha_br * exp_sum_q
        # alliance np (alpha_al)
        elif v == y and u != x:
            intensity += alpha_alr * exp_sum_q
    # loop through node pairs in reciprocal block pair ba
    for (x, y) in events_dict_r:
        xy_timestamps_less_t0 = events_dict_r[(x, y)][events_dict_r[(x, y)] < t0]
        exp_sum_q = 0
        for q in range(Q):
            exp_sum_q += C[q] * betas[q] * np.sum(np.exp(-betas[q] * (t0 - xy_timestamps_less_t0)))
        # reciprocal node_pair events (alpha_r)
        if (v, u) == (x, y):
            intensity += alpha_r * exp_sum_q
        # gr node_pairs events (alpha_gr)
        elif u == y and v != x:
            intensity += alpha_gr * exp_sum_q
        # alliance reciprocal np (alpha_alr)
        elif v == x and u != y:
            intensity += alpha_alr * exp_sum_q
    return intensity


def calculate_auc(y, preds, show_figure=False):
    fpr, tpr, thresholds = metrics.roc_curve(y.flatten(), preds.flatten(), pos_label=1)
    roc_auc = metrics.roc_auc_score(y.flatten(), preds.flatten())
    if show_figure == True:
        plt.figure(1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        # plt.legend(loc="lower right")
        plt.show()
    return roc_auc


def list_to_events_dict(l):
    events_dict = {}
    for r in range(len(l)):
        if (l[r][0], l[r][1]) not in events_dict:
            events_dict[(l[r][0], l[r][1])] = []

        events_dict[(l[r][0], l[r][1])].append(l[r][2])
    return events_dict
#%% motif count

if __name__ == "__main__":
    DOCKER = False
    if DOCKER:
        save_path = "/result/AUC"
    else:
        save_path = "/shared/Results/MultiBlockHawkesModel/AUC"

    dataset = "RealityMining" # RealityMining or Enron or MID


    if dataset == "RealityMining":
        delta = 60
        train_tuple, test_tuple, all_tuple, nodes_not_in_train = load_reality_mining_test_train(
            remove_nodes_not_in_train=True)
        events_dict_train, n_nodes_train, end_time_train = train_tuple
        events_dict_all, n_nodes_all, end_time_all = all_tuple
        K = 2
        print(f"{dataset} Dataset at K = {K}")
    elif dataset == "Enron":
        delta = 125
        train_tuple, test_tuple, all_tuple, nodes_not_in_train = load_enron_train_test(remove_nodes_not_in_train=True)
        events_dict_train, n_nodes_train, end_time_train = train_tuple
        events_dict_all, n_nodes_all, end_time_all = all_tuple
        K = 1
        print(f"{dataset} Dataset at K = {K}")
    elif dataset == "MID":
        with open(f'./storage/datasets/MID/MID_train_all_test.p', 'rb') as file:
            train_tup, all_tup, test_set = pickle.load(file)
        # read version with nodes not in train removed and timestamped scaled [0:1000]
        train_list, end_time_train, n_nodes_train, n_events_train, id_node_map_train = train_tup
        all_list, end_time_all, n_nodes_all, n_events_all, id_node_map_all = all_tup
        events_dict_all = list_to_events_dict(all_list)
        delta = 7.15
        K = 5
        print(f"{dataset} Dataset at K = {K}")


    file_name = f"{dataset}_k_{K}.p"
    # file_name = "Enron_k_2_no_ref.p"
    pickle_file_name = f"{save_path}/{file_name}"
    with open(pickle_file_name, 'rb') as f:
        results_dict = pickle.load(f)
    # refinement fit parameters
    fit_params_tup = results_dict["fit_param_ref"]
    nodes_mem_all = results_dict["node_mem_train_ref"]  # should I only use train node membership??
    t0s = np.loadtxt(f"{save_path}/t0/{dataset}_t0.csv", delimiter=',', usecols=1)
    runs = len(t0s)
    auc = np.zeros(runs)
    y_runs = np.zeros((n_nodes_all, n_nodes_all, runs))
    pred_runs = np.zeros((n_nodes_all, n_nodes_all, runs))
    for i, t0 in enumerate(t0s):
        # t0 = np.random.uniform(low=end_time_train, high=end_time_all - delta, size=None)
        y_mulch, pred_mulch = mulch_predict_probs_and_actual(n_nodes_all, t0, delta, events_dict_all, fit_params_tup,
                                                             nodes_mem_all)
        y_runs[:, :, i] = y_mulch
        pred_runs[:, :, i] = pred_mulch
        auc[i] = calculate_auc(y_mulch, pred_mulch, show_figure=False)
        print(f"at i={i} -> auc={auc[i]}")

    print(f"{results_dict['ll_test_ref']:.5f}\t{K}\t{np.average(auc):.5f}\t{auc.std():.3f}")
    pickle_file_name = f"{save_path}/{dataset}_auc_K_{K}_analytical.p"
    auc_dict = {"t0": t0s, "auc": auc, "avg": np.average(auc), "std": auc.std(), "y__runs": y_runs,
                "pred_runs": pred_runs, "ll_test": results_dict['ll_test_ref']}
    with open(pickle_file_name, 'wb') as f:
        pickle.dump(auc_dict, f)

