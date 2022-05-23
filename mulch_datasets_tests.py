"""Real datasets MULCH Experiments (Section 5.2)

This script runs MULCH on 4 datasets [Reality Mining, Enron, Facebook, Filtered Facebook].
Then, evaluate the model's predictive and generative accuracy.

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

For each dataset, we can set the following variables:
 - betas: np.array of MULCH decays
 - motif_delta: (float) for motif count experiment
 - link_pred_delta: (float) delta for link prediction experiment

@author: Hadeel Soliman
"""

# TODO add one beta fit function -- OR remove from MID
# TODO change t0 array to random
# TODO link prediction is only for n_alpha=6
# TODO remove saved motif from dataset
# TODO should I remove the filtered facebook dataset
# TODO motif and link experiments are only for reality and enron

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import sys

import utils_fit_model as mulch_fit
import utils_accuracy_tests as accuracy_test
from utils_fit_refine_mulch import fit_refinement_mulch
from utils_fit_bp import cal_num_events
from utils_fit_one_beta_model import model_fit_cal_log_likelihood_one_beta

sys.path.append("./CHIP-Network-Model")
from dataset_utils import load_enron_train_test, load_reality_mining_test_train, get_node_map, load_facebook_wall




if __name__ == "__main__":
    dataset = "RealityMining"   # "RealityMining" OR "Enron" OR "Facebook" OR "FacebookFiltered"
    
    docker = False
    if docker:
        save_path = f"/data"  # when called from docker
    else:
        save_path = f'/shared/Results/MultiBlockHawkesModel'

    PRINT_DETAILS = True   # print intermediate details of fitting and other test experiments

    """ Model Fitting"""
    fit_model = True  # either fit mulch or read saved fit
    K_range = [2]  # number of blocks (K) range ex: range(1,11)
    n_alpha = 6  # number of excitations types choose between 2, 4, or 6
    save_fit = False  # save fitted model - specify path
    REF_ITER = 4  # maximum refinement interation - set to 0 for no refinement

    """ Simulation from fit parameters and count motifs experiment"""
    motif_experiment = True
    n_motif_simulations = 2 # number of simulations to count motifs on
    save_motif = False # save simulation motif counts - specify save path in code

    """ link prediction experiment"""
    link_prediction_experiment = True
    save_link = False  # save link prediction results specify path in code


    np.set_printoptions(suppress=True)  # always print floating point numbers using fixed point notation

    # read specified dataset. For each, betas, motif delta, and link_pred_delta are specified
    if dataset == "RealityMining":
        train_tup, test_tuple, all_tup, nodes_not_in_train = load_reality_mining_test_train(remove_nodes_not_in_train=False)
        events_dict_train, n_nodes_train, end_time_train = train_tup
        events_dict_all, n_nodes_all, end_time_all = all_tup
        # betas_recip = np.array([7, 1/2, 1 / 24]) * (1000 / 150)  # [1week, 1/2day, 1hour]
        # betas_recip = np.array([7*2, 1, 1/12]) * (1000 / 150)  # [2week, 1day, 2hour]
        betas_recip = np.array([7, 1, 1 / 24]) * (1000 / 150)  # [1week, 2day, 1hour]
        betas = np.reciprocal(betas_recip)
        motif_delta = 45  # week
        link_pred_delta = 60 # should be two weeks
    elif dataset == "Enron":
        train_tup, test_tuple, all_tup, nodes_not_in_train = load_enron_train_test(remove_nodes_not_in_train=False)
        events_dict_train, n_nodes_train, end_time_train = train_tup
        events_dict_all, n_nodes_all, end_time_all = all_tup
        # betas_recip = np.array([7, 1 / 2, 1 / 24]) * (1000 / 60)  # [1week, 1/2day, 1hour]
        # betas_recip = np.array([7*2, 1, 1/12]) * (1000 / 60)  # [2week, 1day, 2hour]
        betas_recip = np.array([7, 2, 1 / 4]) * (1000 / 60)  # [1week, 2days, 6 hour]
        betas = np.reciprocal(betas_recip)
        motif_delta = 100  # week
        link_pred_delta = 125 # week and quarter
    elif dataset == "Facebook":
        train_tup, test_tuple, all_tup, nodes_not_in_train = load_facebook_wall(timestamp_max=1000,
                                                                                    largest_connected_component_only=True,
                                                                                    train_percentage=0.8)
        events_dict_train, n_nodes_train, end_time_train = train_tup
        events_dict_all, n_nodes_all, end_time_all = all_tup
        betas = np.array([0.02, 0.2, 20])  # [2 month , 1 week , 2 hours]
    else:
        facebook_path = os.path.join(os.getcwd(), "storage", "datasets", "facebook_filtered",
                                     "facebook-wall-filtered.txt")
        train_tup, all_tup, nodes_not_in_train = mulch_fit.read_csv_split_train(facebook_path, delimiter=' ')
        events_dict_train, n_nodes_train, end_time_train, n_events_train, id_node_map_train = train_tup
        events_dict_all, n_nodes_all, end_time_all, n_events_all, id_node_map_all = all_tup
        days = (1196972372 - 1168985687) / 60 / 60 / 24  # dataset lasted for (324 days)
        betas_recip = np.array([2 * 7, 2, 1 / 4]) * (1000 / days)  # [2week, 2days, 6 hour]
        betas = np.reciprocal(betas_recip)



#%% fit MULCH with refinement
    if len(K_range) > 0:
        print(f"Fit {dataset} using {n_alpha}-alpha MULCH at betas={betas}, max #refinement iterations={REF_ITER}")
    for K in K_range:
        if fit_model:
            print("\nFit MULCH at K=", K)
            sp_tup, ref_tup, ref_message = fit_refinement_mulch(events_dict_train, n_nodes_train, end_time_train, K,
                                                                betas, n_alpha, max_iter=REF_ITER, verbose=PRINT_DETAILS)

            # Fit results using spectral clustering for node membership
            nodes_mem_train_sp, fit_param_sp, ll_train_sp, n_events_train, fit_time_sp = sp_tup
            # full dataset nodes membership
            node_mem_all_sp = mulch_fit.assign_node_membership_for_missing_nodes(nodes_mem_train_sp, nodes_not_in_train)
            ll_all_sp, n_events_all = mulch_fit.log_likelihood_mulch(fit_param_sp, events_dict_all, node_mem_all_sp, K,
                                                                     end_time_all)
            # train, full, test log-likelihoods per event
            ll_train_event_sp = ll_train_sp / n_events_train
            ll_all_event_sp = ll_all_sp / n_events_all
            ll_test_event_sp = (ll_all_sp - ll_train_sp) / (n_events_all - n_events_train)

            # Fit results after nodes membership refinement iterations
            nodes_mem_train_ref, fit_param_ref, ll_train_ref, num_events, fit_time_ref = ref_tup
            # full dataset nodes membership
            nodes_mem_all_ref = mulch_fit.assign_node_membership_for_missing_nodes(nodes_mem_train_ref,
                                                                                   nodes_not_in_train)
            ll_all_ref, n_events_all = mulch_fit.log_likelihood_mulch(fit_param_ref, events_dict_all, nodes_mem_all_ref,
                                                                      K, end_time_all)
            # train, full, test log-likelihoods per event
            ll_all_event_ref = ll_all_ref / n_events_all
            ll_train_event_ref = ll_train_ref / n_events_train
            ll_test_event_ref = (ll_all_ref - ll_train_ref) / (n_events_all - n_events_train)

            print(f"->Spectral log-likelihood:\ttrain={ll_train_event_sp:.3f}\tall={ll_all_event_sp:.3f}"
                  f"\ttest={ll_test_event_sp:.3f}")
            print(f"->Refinement log-likelihood:  \ttrain={ll_train_event_ref:.3f}\tall={ll_all_event_ref:.3f}"
                f"\ttest={ll_test_event_ref:.3f}")


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
                fit_dict["train_end_time"] = end_time_train
                fit_dict["all_end_time"] = end_time_all
                full_fit_path = f'/{save_path}/{dataset}/test'
                pickle_file_name = f"{full_fit_path}/k_{K}.p"
                with open(pickle_file_name, 'wb') as f:
                    pickle.dump(fit_dict, f)

        # read saved fit
        else:
            full_fit_path = f"/{save_path}/{dataset}/6alpha_KernelSum_Ref_batch/2month1week2hour"
            with open(f"{full_fit_path}/k_{K}.p", 'rb') as f:
                fit_dict = pickle.load(f)
            # refinement parameters
            fit_param_ref = fit_dict["fit_param_ref"]
            nodes_mem_train_ref = fit_dict["node_mem_train_ref"]
            nodes_mem_all_ref = fit_dict["node_mem_all_ref"]
            ll_test_event_ref = fit_dict["ll_test_ref"]
            print(f"->spectral log-likelihood:\ttrain={fit_dict['ll_train_sp']:.3f}\tall={fit_dict['ll_all_sp']:.3f}"
                  f"\ttest={fit_dict['ll_test_sp']:.3f}")
            print(
                f"->refine log-likelihood:  \ttrain={fit_dict['ll_train_ref']:.3f}\tall={fit_dict['ll_all_ref']:.3f}"
                f"\ttest={fit_dict['ll_test_ref']:.3f}")


        # Simulation and motif experiments
        if motif_experiment and (dataset == "RealityMining" or dataset == "Enron"):
            print(f"\n\nMotifs Count Experiment at delta={motif_delta} (#simulations={n_motif_simulations})")
            # # ---> Either run motif counts on dataset
            # # compute dataset's reciprocity, transitivity, and (6, 6) temporal motifs counts matrix
            # recip, trans, dataset_motif, n_events_train = \
            #     accuracy_test.cal_recip_trans_motif(events_dict_train, n_nodes_train, motif_delta, verbose=PRINT_DETAILS)
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
            with open(f"storage/datasets_motif_counts/week_{dataset}_counts.p", 'rb') as f:
                dataset_motif_dict = pickle.load(f)
            dataset_motif = dataset_motif_dict["dataset_motif"]
            recip = dataset_motif_dict["dataset_recip"]
            trans = dataset_motif_dict["dataset_trans"]
            n_events_train = dataset_motif_dict["dataset_n_events"]
            print(f"->actual {dataset}: reciprocity={recip:.4f}, transitivity={trans:.4f}, #events:{n_events_train}")
            dataset_motif_tup = (recip, trans, dataset_motif, n_events_train)

            # run simulation and count motifs
            motif_test_dict = accuracy_test.simulate_count_motif_experiment(dataset_motif_tup, fit_param_ref, nodes_mem_train_ref,
                                                              K, end_time_train, motif_delta, n_sim=n_motif_simulations,
                                                              verbose=PRINT_DETAILS)
            print("\n->actual dataset motifs count at delta=", motif_delta)
            print(np.asarray(motif_test_dict["dataset_motif"], dtype=int))
            print("->average motifs count over ", n_motif_simulations, " simulations")
            print(np.asarray(motif_test_dict["sim_motif_avg"], dtype=int))
            print(f'->MAPE = {motif_test_dict["mape"]:.2f}')

            if save_motif:
                full_motif_path = f"{save_path}/MotifCounts/{dataset}/test"  # 2month1week2hour
                pickle_file_name = f"{full_motif_path}/k{K}.p"
                with open(pickle_file_name, 'wb') as f:
                    pickle.dump(motif_test_dict, f)

        # Link prediction experiment -- NOTE: used nodes in full dataset
        if link_prediction_experiment and (dataset == "RealityMining" or dataset == "Enron"):
            print("\n\nLink Prediction Experiments at delta=", link_pred_delta)

            t0s = np.loadtxt(f"storage/t0/{dataset}_t0.csv", delimiter=',', usecols=1)
            runs = len(t0s)
            auc = np.zeros(runs)
            y_runs = np.zeros((n_nodes_all, n_nodes_all, runs))
            pred_runs = np.zeros((n_nodes_all, n_nodes_all, runs))
            for i, t0 in enumerate(t0s):
                # t0 = np.random.uniform(low=end_time_train, high=end_time_all - delta, size=None)
                y_mulch, pred_mulch = accuracy_test.mulch_predict_probs_and_actual(n_nodes_all, t0, link_pred_delta,
                                                                     events_dict_all, fit_param_ref, nodes_mem_all_ref)
                y_runs[:, :, i] = y_mulch
                pred_runs[:, :, i] = pred_mulch
                auc[i] = accuracy_test.calculate_auc(y_mulch, pred_mulch, show_figure=False)
                if PRINT_DETAILS:
                    print(f"at i={i} -> auc={auc[i]}")
            print(f"->average AUC={np.average(auc):.5f}, std={auc.std():.3f}")

            if save_link:
                full_link_path = f"{save_path}/AUC/{dataset}"
                pickle_file_name = f"{full_link_path}/auc_k{K}.p"
                auc_dict = {"t0": t0s, "auc": auc, "avg": np.average(auc), "std": auc.std(), "y__runs": y_runs,
                            "pred_runs": pred_runs, "ll_test": ll_test_event_ref}
                with open(pickle_file_name, 'wb') as f:
                    pickle.dump(auc_dict, f)
