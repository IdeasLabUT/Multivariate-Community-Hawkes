import numpy as np
from tick.hawkes import SimuHawkesExpKernels, HawkesExpKern, HawkesADM4
import sys
sys.path.append("./CHIP-Network-Model")
from dataset_utils import load_enron_train_test, load_reality_mining_test_train
import utils_sum_betas_bp
import pickle


# # read Reality Mining dataset
# dataset = "Reality Mining"
# print("Reality Mining Dataset - multiblock HawkesProcess")
# train_tuple, test_tuple, all_tuple, nodes_not_in_train = load_reality_mining_test_train(remove_nodes_not_in_train=False)

# # read Enron dataset
# dataset = "Enron"
# print("Enron Dataset - multiblock HawkesProcess")
# train_tuple, test_tuple, all_tuple, nodes_not_in_train = load_enron_train_test(remove_nodes_not_in_train=False)

def print_log_likelihood_event(est_params_adm4, events_list_train_full, duration_train, M_train,
                               events_list_all_full, duration_all, M_all):
    ll_adm4_train = OneBlockFit.log_likelihood_detailed(est_params_adm4, events_list_train_full, duration_train, M_train)
    ll_adm4_all = OneBlockFit.log_likelihood_detailed(est_params_adm4, events_list_all_full, duration_all, M_all)
    num_events_train = OneBlockFit.cal_num_events(events_list_train_full)
    num_events_all   = OneBlockFit.cal_num_events(events_list_all_full)
    num_events_test  = num_events_all - num_events_train
    print("\nlog-likelihood per event:")
    print(f"\ttrain = {(ll_adm4_train/num_events_train):4f}")
    print(f"\tall = {(ll_adm4_all/num_events_all):4f}")
    print(f"\ttest = {((ll_adm4_all-ll_adm4_train)/num_events_test):4f}")
    return (ll_adm4_train/num_events_train, ll_adm4_all/num_events_all, (ll_adm4_all-ll_adm4_train)/num_events_test)


#
# # define model
# adm4 = HawkesADM4(decay=0.035, lasso_nuclear_ratio=0.5)
# # fit model
# adm4.fit(events_list_train_full, end_times=duration_train)
#
# print(f"ADM4 fitting time =  {adm4.time_elapsed:.2f} s")
#
# est_mu_adm4 = adm4.baseline
# est_alpha_adm4 = adm4.adjacency * adm4.decay
# est_params_adm4 = (est_mu_adm4, est_alpha_adm4, adm4.decay)
#
# # log-likelihood per event
# print_log_likelihood_event(est_params_adm4, events_list_train_full, duration_train, M_train,
#                                events_list_all_full, duration_all, M_all)



# read dataset

# print("ADM4 on Enron")
# train_tuple, test_tuple, all_tuple, enron_nodes_not_in_train = load_enron_train_test(remove_nodes_not_in_train=True)
print("Reality Mining Dataset - ADM4")
train_tuple, test_tuple, all_tuple, nodes_not_in_train = load_reality_mining_test_train(remove_nodes_not_in_train=True)
# dataset tuple
events_dict_train, n_nodes_train, duration_train = train_tuple
events_dict_all, n_nodes_all, duration_all = all_tuple
M_train = n_nodes_train * (n_nodes_train -1)
M_all = n_nodes_all * (n_nodes_all - 1)
# full list of events corresponding to all combinations of node_pairs
# node_pairs with no events have empty list []
events_list_train_full = OneBlockFit.add_empty_node_pairs(events_dict_train, n_nodes_train)
events_list_all_full = OneBlockFit.add_empty_node_pairs(events_dict_all, n_nodes_all)

# for (i, np) in zip(range(len(events_list_train_full)), events_list_train_full):
#     if len(np) != 0:
#         print(i)
#         print(np)
#         break

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

# results_path = '/shared/Results/MultiBlockHawkesModel/Adm4'
results_path = "/data/Adm4"  #when called from docker
file_name = f"adm4_fit_reality.p"
pickle_file_name = f"{results_path}/{file_name}"
with open(pickle_file_name, 'wb') as f:
    pickle.dump(results_dict, f)