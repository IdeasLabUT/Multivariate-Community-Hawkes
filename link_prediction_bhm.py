import numpy as np
from scipy import integrate
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sys
import copy
sys.path.append("./CHIP-Network-Model")
from dataset_utils import load_enron_train_test, load_reality_mining_test_train
import MultiBlockFit as MBHP

def mulch_predict_probs_and_actual(n_nodes, t0, delta, bp_events_dict, params_tup, nodes_mem_all):
    """
    Computes the predicted probability that a link from u to v appears in
    [t, t + delta)
    """
    mu_bp, alpha_n_bp, alpha_r_bp, alpha_br_bp, alpha_gr_bp, alpha_al_bp, alpha_alr_bp, C_bp, betas = params_tup
    prob_dict = np.zeros((n_nodes, n_nodes))  # Predicted probs that link exists
    y = np.zeros((n_nodes, n_nodes))  # actual link
    for u in range(n_nodes):
        for v in range(n_nodes):
            u_b, v_b = nodes_mem_all[u], nodes_mem_all[v]
            par = (mu_bp[u_b, v_b], alpha_n_bp[u_b, v_b], alpha_r_bp[u_b, v_b], alpha_br_bp[u_b, v_b], alpha_gr_bp[u_b, v_b],
                   alpha_al_bp[u_b, v_b], alpha_alr_bp[u_b, v_b], C_bp[u_b, v_b], betas)
            if u_b == v_b:
                integral = \
                integrate.quad(mulch_uv_intensity_dia, t0, t0 + delta, args=((u, v), par, bp_events_dict[u_b][v_b], t0), limit=100)[0]
            else:
                integral = integrate.quad(mulch_uv_intensity_off, t0, t0 + delta,
                                          args=((u, v), par, bp_events_dict[u_b][v_b], bp_events_dict[v_b][u_b], t0), limit=100)[0]
            prob_dict[u, v] = 1 - np.exp(-integral)

            if (u, v) in bp_events_dict[u_b][v_b]:
                uv_times = bp_events_dict[u_b][v_b][(u, v)]
                if len(uv_times[np.logical_and(uv_times >= t0, uv_times <= t0 + delta)]) > 0:
                    y[u, v] = 1

    return y, prob_dict

def bhm_predict_probs_and_actual(t0, delta, n_nodes, events_dict_all, params_tup, K, node_mem_train):
    # event_dictionaries for block pairs
    bp_events_dict_all = MBHP.events_dict_to_blocks(events_dict_all, node_mem_train, K)
    # number of node pairs per block pair & number of nodes in one block
    bp_M, n_nodes_b = MBHP.num_nodes_pairs_per_block_pair(node_mem_train, K)
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

def mulch_uv_intensity_dia(t, uv, params, events_dict, t0):
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, C, betas = params
    u, v = uv
    Q = len(betas)
    intensity = mu
    for (x, y) in events_dict:
        xy_timestamps = events_dict[(x, y)][events_dict[(x, y)] < t0]
        exp_sum_q = 0
        for q in range(Q):
            exp_sum_q += C[q] * betas[q] * np.sum(np.exp(-betas[q] * (t - xy_timestamps)))
        if (u, v) == (x, y): # same node_pair events (alpha_n)
            intensity += alpha_n * exp_sum_q
        elif (v, u) == (x, y): # reciprocal node_pair events (alpha_r)
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
def mulch_uv_intensity_off(t, uv, params, events_dict, events_dict_r, t0):
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, C, betas = params
    u, v = uv
    Q = len(betas)
    intensity = mu
    # loop through node pairs in block pair ab
    for (x, y) in events_dict:
        xy_timestamps = events_dict[(x, y)][events_dict[(x, y)] < t0]
        exp_sum_q = 0
        for q in range(Q):
            exp_sum_q += C[q] * betas[q] * np.sum(np.exp(-betas[q] * (t - xy_timestamps)))
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
        xy_timestamps = events_dict_r[(x, y)][events_dict_r[(x, y)] < t0]
        exp_sum_q = 0
        for q in range(Q):
            exp_sum_q += C[q] * betas[q] * np.sum(np.exp(-betas[q] * (t0 - xy_timestamps)))
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

def calculate_auc(y, preds, show_figure = False):
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
        #plt.legend(loc="lower right")
        plt.show()
    return roc_auc

def list_to_events_dict(l):
    events_dict = {}
    for r in range(len(l)):
        if (l[r][0], l[r][1]) not in events_dict:
            events_dict[(l[r][0], l[r][1])] = []

        events_dict[(l[r][0], l[r][1])].append(l[r][2])
    return events_dict

if __name__ == "__main__":
    save_path = "/shared/Results/MultiBlockHawkesModel/AUC"
    dataset = "RealityMining"

    if dataset == "RealityMining":
        delta = 60
        print(f"{dataset} Dataset - BHM")
        train_tuple, test_tuple, all_tuple, nodes_not_in_train = load_reality_mining_test_train(remove_nodes_not_in_train=True)
        events_dict_train, n_nodes_train, end_time_train = train_tuple
        events_dict_all, n_nodes_all, end_time_all = all_tuple
        K = 50
    elif dataset == "Enron-2":
        delta = 125
        print(f"{dataset} Dataset - BHM")
        train_tuple, test_tuple, all_tuple, nodes_not_in_train = load_enron_train_test(remove_nodes_not_in_train=True)
        events_dict_train, n_nodes_train, end_time_train = train_tuple
        events_dict_all, n_nodes_all, end_time_all = all_tuple
        K = 16
    elif dataset == "MID":
        with open(f'./storage/datasets/MID/MID_train_all_test.p', 'rb') as file:
            train_tup, all_tup, test_set = pickle.load(file)
        # read version with nodes not in train removed and timestamped scaled [0:1000]
        train_list, end_time_train, n_nodes_train, n_events_train, id_node_map_train = train_tup
        all_list, end_time_all, n_nodes_all, n_events_all, id_node_map_all = all_tup
        events_dict_all = list_to_events_dict(all_list)
        results_path = "/shared/Results/MultiBlockHawkesModel/BHM_MID"
        delta = 7.15
        K = 95
    elif dataset == "Enron-15":
        delta = 60
        K= 14
        print(f"{dataset} Dataset - BHM")
        with open('storage/datasets/enron2/enron-events.pckl', 'rb') as f:
            n_nodes_all, T_all, enron_all = pickle.load(f)
        events_dict_all = {}
        n_events_all = len(enron_all)
        for u, v, t in enron_all:
            if (u, v) not in events_dict_all:
                events_dict_all[(u, v)] = []
            events_dict_all[(u, v)].append(t)
    elif dataset == "fb-forum":
        delta = 80
        K = 57
        print(f"{dataset} Dataset - BHM")
        data_path = "/nethome/hsolima/MultivariateBlockHawkesProject/MultivariateBlockHawkes/storage/datasets"
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
        n_nodes_all = len(nodes_train_set)
        # create events_dict_train
        events_dict_all = {}
        for i in range(len(train_data)):
            sender_id, receiver_id = node_id_map[train_data[i, 0]], node_id_map[train_data[i, 1]]
            if (sender_id, receiver_id) not in events_dict_all:
                events_dict_all[(sender_id, receiver_id)] = []
            events_dict_all[(sender_id, receiver_id)].append(train_data[i, 2])

        # remove nodes in test not in train
        for i in range(len(test)):
            if test[i, 0] in node_id_map and test[i, 1] in node_id_map:
                sender_id, receiver_id = node_id_map[test[i, 0]], node_id_map[test[i, 1]]
                if (sender_id, receiver_id) not in events_dict_all:
                    events_dict_all[(sender_id, receiver_id)] = []
                events_dict_all[(sender_id, receiver_id)].append(test[i, 2])


    # results_path = save_path
    # file_name = f"{dataset}_k_{K}.p"
    # pickle_file_name = f"{results_path}/{file_name}"
    # with open(pickle_file_name, 'rb') as f:
    #     results_dict = pickle.load(f)
    # node_mem_train = results_dict["node_mem_train"]  # use train node membership (removed nodes not in train)
    # param_tuple = results_dict["param"]
    # runs = 100
    # t0s = np.loadtxt(f"{save_path}/t0/{dataset}_t0.csv", delimiter=',', usecols=1)
    # auc = np.zeros(runs)
    # y_runs = np.zeros((n_nodes_all, n_nodes_all, runs))
    # pred_runs = np.zeros((n_nodes_all, n_nodes_all, runs))
    # for i, t0 in enumerate(t0s):
    #     # t0 = np.random.uniform(low=end_time_train, high=end_time_all - delta, size=None)
    #     y_bhm, pred_bhm = bhm_predict_probs_and_actual(t0, delta, n_nodes_all, events_dict_all, param_tuple, K, node_mem_train)
    #     y_runs[:, :, i] = y_bhm
    #     pred_runs[:, :, i] = pred_bhm
    #     auc[i] = calculate_auc(y_bhm, pred_bhm, show_figure=False)
    # print(f'{results_dict["ll_test"]:.5f}\t{K}\t{np.average(auc):.5f}\t{auc.std():.3f}')
    # pickle_file_name = f"{save_path}/{dataset}_auc_K_{K}.p"
    # auc_dict = {"t0": t0s, "auc": auc, "avg": np.average(auc), "std": auc.std(), "y__runs": y_runs, "pred_runs": pred_runs}
    # with open(pickle_file_name, 'wb') as f:
    #     pickle.dump(auc_dict, f)

    results_path = save_path
    file_name = f"{dataset}_k_{K}.p"
    pickle_file_name = f"{results_path}/{file_name}"
    with open(pickle_file_name, 'rb') as f:
        results_dict = pickle.load(f)
    # refinement fit parameters
    fit_params_tup = results_dict["fit_param_ref"]
    nodes_mem_all = results_dict["node_mem_all_ref"] # should I only use train node membership??
    block_pairs_all = MBHP.events_dict_to_blocks(events_dict_all, nodes_mem_all, K)
    runs = 100
    t0s = np.loadtxt(f"{save_path}/t0/{dataset}_t0.csv", delimiter=',', usecols=1)
    auc = np.zeros(runs)
    y_runs = np.zeros((n_nodes_all, n_nodes_all, runs))
    pred_runs = np.zeros((n_nodes_all, n_nodes_all, runs))
    for i, t0 in enumerate(t0s):
        # t0 = np.random.uniform(low=end_time_train, high=end_time_all - delta, size=None)
        y_bhm, pred_bhm = bhm_predict_probs_and_actual(t0, delta, n_nodes_all, events_dict_all, param_tuple, K,
                                                       node_mem_train)
        y_runs[:, :, i] = y_bhm
        pred_runs[:, :, i] = pred_bhm
        auc[i] = calculate_auc(y_bhm, pred_bhm, show_figure=False)
    print(f'{results_dict["ll_test"]:.5f}\t{K}\t{np.average(auc):.5f}\t{auc.std():.3f}')
    pickle_file_name = f"{save_path}/{dataset}_auc_K_{K}.p"
    auc_dict = {"t0": t0s, "auc": auc, "avg": np.average(auc), "std": auc.std(), "y__runs": y_runs,
                "pred_runs": pred_runs}
    with open(pickle_file_name, 'wb') as f:
        pickle.dump(auc_dict, f)


    # delta = 60
    # K=8
    #
    # path = f'/shared/Results/MultiBlockHawkesModel/{dataset}/6alpha_KernelSum_Ref_batch'
    # fit_path = f"{path}/2week1day2hour"
    # file_name = f"k_{K}.p"
    # with open(f"{fit_path}/{file_name}", 'rb') as f:
    #     results_dict = pickle.load(f)
    # # refinement fit parameters
    # fit_params_tup = results_dict["fit_param_ref"]
    # nodes_mem_all = results_dict["node_mem_all_ref"]
    # block_pairs_all = MBHP.events_dict_to_blocks(events_dict_all, nodes_mem_all, K)
    # runs = 3
    # auc = np.zeros(runs)
    # for i in range(runs):
    #     t0 = np.random.uniform(low=end_time_train, high=end_time_all-delta, size=None)
    #     y_mul, preds_mul = mulch_predict_probs_and_actual(n_nodes_all, t0, delta, block_pairs_all, fit_params_tup, nodes_mem_all)
    #     auc[i] = calculate_auc(y_mul,preds_mul)
        
    '''
    y = actual_y(n_nodes_train, t0, delta, P_all)
    scores = predict_probs(n_nodes_train, t0, delta, P_sim, mu_est, alpha, beta, C)
    fpr, tpr, thresholds = metrics.roc_curve(y.flatten(), scores.flatten(), pos_label=1)
    roc_auc = metrics.roc_auc_score(y.flatten(), scores.flatten())
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()'''
 
