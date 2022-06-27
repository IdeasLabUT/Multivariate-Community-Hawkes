import numpy as np
from tick.hawkes import SimuHawkesExpKernels, HawkesExpKern, HawkesADM4
import sys
sys.path.append("./CHIP-Network-Model")
from dataset_utils import load_enron_train_test, load_reality_mining_test_train


def log_likelihood_detailed(params_array, events_list, end_time, M):
    """ mu_array : (M,) array : baseline intensity of each process
        alpoha_array: (M,M) narray: adjacency*beta (actual jumbs) """
    mu_array, alpha_array, beta = params_array
    # set all mu=0 to mu=1e-10/end_time
   # mu_array[mu_array==0] = 1e-10/end_time

    # first term
    first = - np.sum(mu_array)*end_time
    # second term
    second = 0
    for m in range(M):
        for v in range(M):
            if len(events_list[v]) == 0:
                continue
            second -= alpha_array[m, v] / beta * np.sum(1 - np.exp(-beta * (end_time - events_list[v])))
    # print("second detailed = ", second)
    # third term
    third = 0
    for m in range(M):
        for k in range(len(events_list[m])):
            tmk = events_list[m][k]
            inter_sum = 0
            for v in range(M):
                v_less = events_list[v][events_list[v] < tmk]
                Rmvk = np.sum(np.exp(-beta * (tmk - v_less)))
                inter_sum += alpha_array[m, v] * Rmvk
            third += np.log(mu_array[m] + inter_sum)
    # print("third detailed = ", third)
    return first+second+third


def add_empty_node_pairs(events_dict, n_nodes):
    events_list_full = []
    for u in range(n_nodes):
        for v in range(n_nodes):
            if u == v: continue
            if (u, v) in events_dict:
                events_list_full.append(np.array(events_dict[(u, v)]))
            else:
                events_list_full.append(np.array([]))
    return events_list_full


def cal_num_events(events_list):
    num_events = 0
    for events_array in events_list:
        num_events += len(events_array)
    return num_events


def print_log_likelihood_event(est_params_adm4, events_list_train_full, duration_train, M_train,
                               events_list_all_full, duration_all, M_all):
    ll_adm4_train = log_likelihood_detailed(est_params_adm4, events_list_train_full, duration_train, M_train)
    ll_adm4_all = log_likelihood_detailed(est_params_adm4, events_list_all_full, duration_all, M_all)
    num_events_train = cal_num_events(events_list_train_full)
    num_events_all   = cal_num_events(events_list_all_full)
    num_events_test  = num_events_all - num_events_train
    print("\nlog-likelihood per event:")
    print(f"\ttrain = {(ll_adm4_train/num_events_train):4f}")
    print(f"\tall = {(ll_adm4_all/num_events_all):4f}")
    print(f"\ttest = {((ll_adm4_all-ll_adm4_train)/num_events_test):4f}")
    return (ll_adm4_train/num_events_train, ll_adm4_all/num_events_all, (ll_adm4_all-ll_adm4_train)/num_events_test)

if __name__ == "__main__":
    print("Reality Mining Dataset - ADM4")
    train_tuple, test_tuple, all_tuple, nodes_not_in_train = load_reality_mining_test_train(remove_nodes_not_in_train=True)
    # dataset tuple
    events_dict_train, n_nodes_train, duration_train = train_tuple
    events_dict_all, n_nodes_all, duration_all = all_tuple
    M_train = n_nodes_train * (n_nodes_train -1)
    M_all = n_nodes_all * (n_nodes_all - 1)
    # full list of events corresponding to all combinations of node_pairs
    # node_pairs with no events have empty list []
    events_list_train_full = add_empty_node_pairs(events_dict_train, n_nodes_train)
    events_list_all_full = add_empty_node_pairs(events_dict_all, n_nodes_all)


    # define model
    adm4 = HawkesADM4(decay=0.02, lasso_nuclear_ratio=0.5, verbose=True)
    # fit model
    adm4.fit(events_list_train_full, end_times=duration_train)
    # estimated ADM4 parameters
    est_mu_adm4 = adm4.baseline
    est_mu_adm4[est_mu_adm4==0] = 1e-10
    est_alpha_adm4 = adm4.adjacency * adm4.decay
    est_params_adm4 = (est_mu_adm4, est_alpha_adm4, adm4.decay)
    # model perfermance
    print(f"fitting time =  {adm4.time_elapsed:.2f} s")
    # log-likelihood per event
    ll = print_log_likelihood_event(est_params_adm4, events_list_train_full, duration_train, M_train,
                                   events_list_all_full, duration_all, M_all)

    results_dict = {}
    results_dict["param"] = est_params_adm4
    results_dict["ll"] = ll
    results_dict["fit_time"] = adm4.time_elapsed
