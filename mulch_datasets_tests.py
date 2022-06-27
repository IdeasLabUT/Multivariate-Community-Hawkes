"""Real datasets MULCH Experiments (Section 5.2)

This script runs MULCH on 3 datasets [Reality Mining, Enron, Facebook].
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
 - REF_ITER: # maximum refinement interation - set to 0 for no refinement
 - motif_experiment: (bool) if True, run motif count experiment
 - n_motif_simulations: # number of simulations for motif count experiment (default=10)
 - link_prediction_experiment: (bool) if True, run link prediction experiment

For each dataset, we can set the following variables:
 - betas: np.array of MULCH decays
 - motif_delta: (float) for motif count experiment
 - link_pred_delta: (float) delta for link prediction experiment

@author: Hadeel Soliman
"""

import numpy as np
import pickle
import os
import sys
import utils_fit_model as fit_model
import utils_accuracy_tests as accuracy_tests
from utils_fit_refine_mulch import fit_refinement_mulch

sys.path.append("./CHIP-Network-Model")
from dataset_utils import load_enron_train_test, load_reality_mining_test_train, get_node_map, \
    load_facebook_wall

if __name__ == "__main__":
    dataset = "RealityMining"  # "RealityMining" OR "Enron" OR "Facebook"

    PRINT_DETAILS = True  # print intermediate details of fitting and other test experiments

    """ Model Fitting"""
    K_range = range(1, 11)  # number of blocks (K) range ex: range(1,11)
    n_alpha = 6  # number of excitations types choose between 2, 4, or 6
    REF_ITER = 7  # maximum refinement interation - set to 0 for no refinement
    save_fit = False  # save fitted model - specify path

    """ Simulation from fit parameters and count motifs experiment"""
    motif_experiment = True
    n_motif_simulations = 10  # number of simulations to count motifs on
    save_motif = False  # save simulation motif counts - specify save path in code

    """ link prediction experiment"""
    link_prediction_experiment = True
    save_link = False  # save link prediction results specify path in code

    np.set_printoptions(
        suppress=True)  # always print floating point numbers using fixed point notation

    # read specified dataset. For each, betas, motif delta, and link_pred_delta are specified
    if dataset == "RealityMining":
        train_tup, test_tuple, all_tup, nodes_not_in_train = load_reality_mining_test_train(
            remove_nodes_not_in_train=False)
        events_dict_train, n_nodes_train, T_train = train_tup
        events_dict_all, n_nodes_all, T_all = all_tup
        # betas_recip = np.array([7*2, 1, 1/12]) * (1000 / 150)  # [2week, 1day, 2hour]
        betas_recip = np.array([7, 1, 1 / 24]) * (1000 / 150)  # [1week, 2day, 1hour] <- paper
        betas = np.reciprocal(betas_recip)
        motif_delta = 45  # week
        link_pred_delta = 60  # should be two weeks
    elif dataset == "Enron":
        train_tup, test_tuple, all_tup, nodes_not_in_train = load_enron_train_test(
            remove_nodes_not_in_train=False)
        events_dict_train, n_nodes_train, T_train = train_tup
        events_dict_all, n_nodes_all, T_all = all_tup
        # betas_recip = np.array([7*2, 1, 1/12]) * (1000 / 60)  # [2week, 1day, 2hour]
        betas_recip = np.array([7, 2, 1 / 4]) * (1000 / 60)  # [1week, 2days, 6 hour] <- paper
        betas = np.reciprocal(betas_recip)
        motif_delta = 100  # week
        link_pred_delta = 125  # week and quarter
    else:
        train_tup, test_tuple, all_tup, nodes_not_in_train = load_facebook_wall(timestamp_max=1000,
                                                                                largest_connected_component_only=True,
                                                                                train_percentage=0.8)
        events_dict_train, n_nodes_train, T_train = train_tup
        events_dict_all, n_nodes_all, T_all = all_tup
        betas = np.array([0.02, 0.2, 20])  # almost [2 month , 1 week , 2 hours]
        # note: dataset_duration = 1591 days  &  test_period = 90 days
        motif_delta = 14 * (1000 / 1591)  # 2 weeks
        link_pred_delta = 14 * (1000 / 1591)  # 2 week
        # another suggested beta
        # betas = np.reciprocal(np.array([30, 2, 3/24]) * (1000/1591)) # [1month, 2days, 3hours]

    # %% fit MULCH with refinement
    if len(K_range) > 0:
        print(f"Fit {dataset} using {n_alpha}-alpha MULCH at betas={betas}, max #ref={REF_ITER}")
    for K in K_range:
        print("\nFit MULCH at K=", K)
        sp_tup, ref_tup, ref_message = fit_refinement_mulch(events_dict_train, n_nodes_train,
                                                            T_train, K,
                                                            betas, n_alpha, max_ref_iter=REF_ITER,
                                                            verbose=PRINT_DETAILS)

        # Fit results using spectral clustering for node membership
        nodes_mem_train_sp, fit_param_sp, ll_train_sp, n_events_train, fit_time_sp = sp_tup
        # full dataset nodes membership
        node_mem_all_sp = fit_model.assign_node_membership_for_missing_nodes(nodes_mem_train_sp,
                                                                             nodes_not_in_train)
        ll_all_sp, n_events_all = fit_model.log_likelihood_mulch(fit_param_sp, events_dict_all,
                                                                 node_mem_all_sp, K,
                                                                 T_all)
        # train, full, test log-likelihoods per event
        ll_train_event_sp = ll_train_sp / n_events_train
        ll_all_event_sp = ll_all_sp / n_events_all
        ll_test_event_sp = (ll_all_sp - ll_train_sp) / (n_events_all - n_events_train)

        # Fit results after nodes membership refinement iterations
        nodes_mem_train_ref, fit_param_ref, ll_train_ref, num_events, fit_time_ref = ref_tup
        # full dataset nodes membership
        nodes_mem_all_ref = fit_model.assign_node_membership_for_missing_nodes(nodes_mem_train_ref,
                                                                               nodes_not_in_train)
        ll_all_ref, n_events_all = fit_model.log_likelihood_mulch(fit_param_ref, events_dict_all,
                                                                  nodes_mem_all_ref,
                                                                  K, T_all)
        # train, full, test log-likelihoods per event
        ll_all_event_ref = ll_all_ref / n_events_all
        ll_train_event_ref = ll_train_ref / n_events_train
        ll_test_event_ref = (ll_all_ref - ll_train_ref) / (n_events_all - n_events_train)

        print(f"->Spectral log-likelihood:\ttrain={ll_train_event_sp:.3f}"
              f"\tall={ll_all_event_sp:.3f}\ttest={ll_test_event_sp:.3f}")
        print(f"->Refinement log-likelihood:  \ttrain={ll_train_event_ref:.3f}"
              f"\tall={ll_all_event_ref:.3f}\ttest={ll_test_event_ref:.3f}")

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
            pickle_file_name = f"{dataset}_mulch_fit_k_{K}.p"
            with open(pickle_file_name, 'wb') as f:
                pickle.dump(fit_dict, f)

        # %% Simulation and motif experiments
        if motif_experiment and (dataset == "RealityMining" or dataset == "Enron"):
            print(
                f"\n\nMotifs Count Experiment at delta={motif_delta} (#simulations={n_motif_simulations})")
            # # ---> Either run motif counts on dataset
            # recip, trans, dataset_motif, n_events_train = \
            #     accuracy_tests.cal_recip_trans_motif(events_dict_train, n_nodes_train, motif_delta,
            #                                          verbose=PRINT_DETAILS)

            # ---> OR read networks recip, trans, motifs count from saved pickle
            dataset_motif_path = os.path.join(os.getcwd(), "storage", "datasets_motif_counts",
                                              f"week_{dataset}_counts.p")
            with open(dataset_motif_path, 'rb') as f:
                dataset_motif_dict = pickle.load(f)
            dataset_motif = dataset_motif_dict["dataset_motif"]
            recip = dataset_motif_dict["dataset_recip"]
            trans = dataset_motif_dict["dataset_trans"]
            n_events_train = dataset_motif_dict["dataset_n_events"]

            print(
                f"->actual {dataset}: recip={recip:.4f}, trans={trans:.4f}, #events:{n_events_train}")
            dataset_motif_tup = (recip, trans, dataset_motif, n_events_train)

            # run simulation and count motifs
            motif_test_dict = accuracy_tests.simulate_count_motif_experiment(dataset_motif_tup,
                                                                             fit_param_ref,
                                                                             nodes_mem_train_ref,
                                                                             K, T_train,
                                                                             motif_delta,
                                                                             n_sim=n_motif_simulations,
                                                                             verbose=PRINT_DETAILS)
            print("\n->actual dataset motifs count at delta=", motif_delta)
            print(np.asarray(motif_test_dict["dataset_motif"], dtype=int))
            print("->average motifs count over ", n_motif_simulations, " simulations")
            print(np.asarray(motif_test_dict["sim_motif_avg"], dtype=int))
            print(f'->at K={K}: MAPE = {motif_test_dict["mape"]:.2f}')

            if save_motif:
                pickle_file_name = f"{dataset}_motif_counts_k_{K}.p"
                with open(pickle_file_name, 'wb') as f:
                    pickle.dump(motif_test_dict, f)

        # %% Link prediction experiment
        if link_prediction_experiment and (dataset == "RealityMining" or dataset == "Enron"):
            print("\n\nLink Prediction Experiments at delta=", link_pred_delta)

            # read saved t0s to replicate experiments results
            # t0s_path = os.path.join(os.getcwd(), "storage", "t0", f"{dataset}_t0.csv")
            # t0s = np.loadtxt(t0s_path, delimiter=',', usecols=1)

            t0s = np.random.uniform(low=T_train, high=T_all - link_pred_delta, size=100)
            runs = len(t0s)
            auc = np.zeros(runs)
            y_runs = np.zeros((n_nodes_all, n_nodes_all, runs))
            pred_runs = np.zeros((n_nodes_all, n_nodes_all, runs))
            for i, t0 in enumerate(t0s):
                y_mulch, pred_mulch = accuracy_tests.mulch_predict_probs_and_actual(n_nodes_all, t0,
                                                                                    link_pred_delta,
                                                                                    events_dict_all,
                                                                                    fit_param_ref,
                                                                                    nodes_mem_all_ref)
                y_runs[:, :, i] = y_mulch
                pred_runs[:, :, i] = pred_mulch
                auc[i] = accuracy_tests.calculate_auc(y_mulch, pred_mulch, show_figure=False)
                if PRINT_DETAILS:
                    print(f"at i={i} -> auc={auc[i]}")
            print(f"->at K={K}: average AUC={np.average(auc):.5f}, std={auc.std():.3f}")

            if save_link:
                auc_dict = {"t0": t0s, "auc": auc, "avg": np.average(auc), "std": auc.std(),
                            "y__runs": y_runs,
                            "pred_runs": pred_runs, "ll_test": ll_test_event_ref}
                pickle_file_name = f"{dataset}_link_pred_auc_k_{K}.p"
                with open(pickle_file_name, 'wb') as f:
                    pickle.dump(auc_dict, f)
