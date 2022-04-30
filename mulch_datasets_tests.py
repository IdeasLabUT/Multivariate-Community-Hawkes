# TODO add one beta fit function -- OR remove from MID
# TODO change t0 array to random
# TODO link prediction is only for n_alpha=6
# TODO remove saved motif from dataset

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from utils_fit_sum_betas_model import read_cvs_split_train
import sys
sys.path.append("./CHIP-Network-Model")
from dataset_utils import load_enron_train_test, load_reality_mining_test_train, get_node_map, load_facebook_wall
import utils_accuracy_tests as accuracy_test
from utils_fit_one_beta_model import model_fit_cal_log_likelihood_one_beta
from refinement_alg import model_fit_cal_log_likelihood_sum_betas
from utils_sum_betas_bp import cal_num_events



if __name__ == "__main__":
    dataset = "RealityMining"   # "RealityMining" OR "Enron" OR "Facebook" OR "FacebookFiltered"
    docker = False
    if docker:
        save_path = f"/data"  # when called from docker
    else:
        save_path = f'/shared/Results/MultiBlockHawkesModel'

    """ model fitting """
    fit_model = True
    save_fit = False # set save path
    REF_ITER = 3
    n_alpha = 6
    K_range = [4]   #list(range(1, 11))

    """ motif simulation """
    motif_experiment = False
    n_motif_simulations = 2
    save_motif = True  # specify path in code

    """ link prediction """
    link_prediction_experiment = True
    save_link = True  # specify path in code




    np.set_printoptions(suppress=True)  # always print floating point numbers using fixed point notation

    if dataset == "RealityMining":
        train_tup, test_tuple, all_tup, nodes_not_in_train = load_reality_mining_test_train(remove_nodes_not_in_train=False)
        # betas_recip = np.array([7, 1/2, 1 / 24]) * (1000 / 150)  # [1week, 1/2day, 1hour]
        # betas_recip = np.array([7*2, 1, 1/12]) * (1000 / 150)  # [2week, 1day, 2hour]
        betas_recip = np.array([7, 1, 1 / 24]) * (1000 / 150)  # [1week, 2day, 1hour]
        betas = np.reciprocal(betas_recip)
        motif_delta = 45  # week
        link_pred_delta = 60 # should be two weeks
    elif dataset == "Enron":
        train_tup, test_tuple, all_tup, nodes_not_in_train = load_enron_train_test(remove_nodes_not_in_train=False)
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
        betas = np.array([0.02, 0.2, 20])  # [2 month , 1 week , 2 hours]
    else:
        facebook_path = os.path.join(os.getcwd(), "storage", "datasets", "facebook_filtered",
                                     "facebook-wall-filtered.txt")
        train_tup, all_tup, nodes_not_in_train = read_cvs_split_train(facebook_path)
        days = (1196972372 - 1168985687) / 60 / 60 / 24  # dataset lasted for (324 days)
        betas_recip = np.array([2 * 7, 2, 1 / 4]) * (1000 / days)  # [2week, 2days, 6 hour]
        betas = np.reciprocal(betas_recip)

    # dataset tuple to events_list_full, num_nodes, duration
    events_dict_train, n_nodes_train, end_time_train = train_tup
    events_dict_all, n_nodes_all, end_time_all = all_tup


#%% fit MULCH with refinement
    if len(K_range) > 0:
        print(f"Fit {dataset} using {n_alpha}-alpha (sum of kernel) at betas={betas}")
    for K in K_range:
        if fit_model:
            print("\nfit MULCH + refinement at K=", K)
            fit_dict = model_fit_cal_log_likelihood_sum_betas(train_tup, all_tup, nodes_not_in_train, n_alpha, K,
                                                              betas, REF_ITER, verbose=True)
            # NOTE: if max_iter=0 no refinement tuple is returned
            print(f"spectral log-likelihood:\ttrain={fit_dict['ll_train_sp']:.3f}\tall={fit_dict['ll_all_sp']:.3f}"
                  f"\ttest={fit_dict['ll_test_sp']:.3f}")
            print(
                f"refine log-likelihood:  \ttrain={fit_dict['ll_train_ref']:.3f}\tall={fit_dict['ll_all_ref']:.3f}"
                f"\ttest={fit_dict['ll_test_ref']:.3f}")


            if save_fit:
                # full_fit_path = f"{fit_path}/6alpha_KernelSum_Ref_batch/2month1week2hour"
                full_fit_path = f'/{save_path}/{dataset}/test'
                pickle_file_name = f"{full_fit_path}/k_{K}.p"
                with open(pickle_file_name, 'wb') as f:
                    pickle.dump(fit_dict, f)

        # read saved fit
        else:
            full_fit_path = f"/{save_path}/{dataset}/6alpha_KernelSum_Ref_batch/2month1week2hour"
            with open(f"{full_fit_path}/k_{K}.p", 'rb') as f:
                fit_dict = pickle.load(f)
            print(f"spectral log-likelihood:\ttrain={fit_dict['ll_train_sp']:.3f}\tall={fit_dict['ll_all_sp']:.3f}"
                  f"\ttest={fit_dict['ll_test_sp']:.3f}")
            print(
                f"refine log-likelihood:  \ttrain={fit_dict['ll_train_ref']:.3f}\tall={fit_dict['ll_all_ref']:.3f}"
                f"\ttest={fit_dict['ll_test_ref']:.3f}")


        # Simulation and motif experiments
        if motif_experiment and (dataset == "RealityMining" or dataset == "Enron"):
            # refinement parameters
            fit_param_ref = fit_dict["fit_param_ref"]
            nodes_mem_train_ref = fit_dict["node_mem_train_ref"]

            # # ---> Either run motif counts on dataset
            # dataset_recip, dataset_trans, dataset_motif = accuracy_test.cal_recip_trans_motif(events_dict_train,
            #                                                                                   n_nodes_train,motif_delta,
            #                                                                                   verbose=False)
            # dataset_n_events_train = cal_num_events(events_dict_train)
            # dataset_motif_tup = (dataset_recip, dataset_trans, dataset_motif, dataset_n_events_train)

            # ---> OR read networks recip, trans, motifs count from saved pickle
            with open(f"storage/datasets_motif_counts/week_{dataset}_counts.p", 'rb') as f:
                dataset_motif_dict = pickle.load(f)
            dataset_motif = dataset_motif_dict["dataset_motif"]
            recip = dataset_motif_dict["dataset_recip"]
            trans = dataset_motif_dict["dataset_trans"]
            n_events_train = dataset_motif_dict["dataset_n_events"]
            print(f"{dataset}: reciprocity={recip:.4f}, transitivity={trans:.4f}, #events:{n_events_train}")
            dataset_motif_tup = (recip, trans, dataset_motif, n_events_train)

            # run simulation and count motifs
            motif_test_dict = accuracy_test.simulate_count_motif_experiment(dataset_motif_tup, fit_param_ref, nodes_mem_train_ref,
                                                              K, end_time_train, motif_delta, n_sim=n_motif_simulations,
                                                              verbose=True)
            if save_motif:
                full_motif_path = f"{save_path}/MotifCounts/{dataset}/test"  # 2month1week2hour
                pickle_file_name = f"{full_motif_path}/k{K}.p"
                with open(pickle_file_name, 'wb') as f:
                    pickle.dump(motif_test_dict, f)

        # Link prediction experiment -- NOTE: didn't remove nodes from train
        if link_prediction_experiment and (dataset == "RealityMining" or dataset == "Enron") and n_alpha==6:
            print("Link Prediction Experiments at delta=", link_pred_delta)
            fit_params_tup = fit_dict["fit_param_ref"]
            nodes_mem_all = fit_dict["node_mem_all_ref"]  # <--- using full node membership
            t0s = np.loadtxt(f"storage/t0/{dataset}_t0.csv", delimiter=',', usecols=1)
            runs = len(t0s)
            auc = np.zeros(runs)
            y_runs = np.zeros((n_nodes_all, n_nodes_all, runs))
            pred_runs = np.zeros((n_nodes_all, n_nodes_all, runs))
            for i, t0 in enumerate(t0s):
                # t0 = np.random.uniform(low=end_time_train, high=end_time_all - delta, size=None)
                y_mulch, pred_mulch = accuracy_test.mulch_predict_probs_and_actual(n_nodes_all, t0, link_pred_delta,
                                                                     events_dict_all, fit_params_tup, nodes_mem_all)
                y_runs[:, :, i] = y_mulch
                pred_runs[:, :, i] = pred_mulch
                auc[i] = accuracy_test.calculate_auc(y_mulch, pred_mulch, show_figure=False)
                print(f"at i={i} -> auc={auc[i]}")

            print(f"{fit_dict['ll_test_ref']:.5f}\t{K}\t{np.average(auc):.5f}\t{auc.std():.3f}")
            if save_link:
                full_link_path = f"{save_path}/AUC/{dataset}"
                pickle_file_name = f"{full_link_path}/auc_k{K}.p"
                auc_dict = {"t0": t0s, "auc": auc, "avg": np.average(auc), "std": auc.std(), "y__runs": y_runs,
                            "pred_runs": pred_runs, "ll_test": fit_dict['ll_test_ref']}
                with open(pickle_file_name, 'wb') as f:
                    pickle.dump(auc_dict, f)
