import pickle
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

#%% helper functions
def plot_paramters(results_dict, n_alpha):
    for key, i in zip(results_dict, range(n_alpha + 1)):
        plt.figure()
        plt.imshow(results_dict[key])
        plt.colorbar()
        plt.title(key)
        plt.show()

def plot_paramters2(param_tup, n_alpha):
    param_name = ["mu", "alpha_self", "alpha_recip", "alpha_turn_cont", "alpha_generalized_recip",
                  "alpha_allied_cont", "alpha_allied_recip"]
    for param, i in zip(param_tup, range(n_alpha + 1)):
        fig, ax = plt.subplots(figsize=(5, 4))
        plot = ax.pcolor(param, cmap='gist_yarg')
        ax.set_xticks(np.arange(0.5, 4))
        ax.set_xticklabels(np.arange(1,5))
        ax.set_yticks(np.arange(0.5, 4))
        ax.set_yticklabels(np.arange(1, 5))
        ax.invert_yaxis()
        fig.colorbar(plot, ax=ax)
        fig.tight_layout()
        # ax.set_title(param_name[i])
        fig.savefig(f"/shared/Results/MultiBlockHawkesModel/figures/MID_casestudy/{param_name[i]}.pdf")
        plt.show()


def print_model_param_kernel_sum(params_est):
    print("mu")
    print(params_est[0])
    print("\nalpha_n")
    print(params_est[1])
    print("\nalpha_r")
    print(params_est[2])
    classes = np.shape(params_est[0])[0]
    if len(params_est) == 5:
        print("\nC")
        for i in range(classes):
            for j in range(classes):
                print(params_est[3][i, j, :], end='\t')
            print(" ")
    else:
        print("\nalpha_br")
        print(params_est[3])
        print("\nalpha_gr")
        print(params_est[4])
        if len(params_est) == 7:
            print("\nC")
            for i in range(classes):
                for j in range(classes):
                    print(params_est[5][i, j, :], end='\t')
                print(" ")
        elif len(params_est) == 9:
            print("\nalpha_al")
            print(params_est[5])
            print("\nalpha_alr")
            print(params_est[6])
            print("\nC")
            for i in range(classes):
                for j in range(classes):
                    print(params_est[7][i, j, :], end='\t')
                print(" ")

def print_parameters(results_dict, n_alpha):
    for key, i in zip(results_dict, range(n_alpha + 2)):
        print(key)
        print(results_dict[key])
        print('')

def analyze_block(node_mem, K, id_node_map):
    print(np.histogram(node_mem, bins=K))
    for i in range(K):
        print(f"Class {i}")
        nodes_in_class_i = np.where(node_mem == i)[0]
        for id in nodes_in_class_i:
            print(id_node_map[id], end=' ')
        print()
#%% tests

if __name__ == "__main__":
    MBHM_loop = False
    CHIP_loop = False
    save = True

    # dataset = "Enron"
    dataset = "RealityMining"
    # dataset = "MID"
    motif_delta = "week"
    # motif_delta = "month"
    betas = "2month1week2hour" # 2week1day2hour_all
    results_path = f"/shared/Results/MultiBlockHawkesModel/MotifCounts/{dataset}"
    fig_path = f"/shared/Results/MultiBlockHawkesModel/figures/{dataset}"

    # with open(f"Datasets_motif_counts/{motif_delta}_{dataset}_counts.p", 'rb') as f:
    #     Mcounts = pickle.load(f)
    # dataset_counts = Mcounts["dataset_motif"]
    # print(f"actual at delta = 1 {motif_delta}")
    # print(np.asarray(dataset_counts, dtype=int))
    #
    # if save:
    #     # read best of 3 models
    #     pickle_file_name = f"{results_path}/old/1week2days1hour/k7.p"  # 1week2days1_4day  2week2days1_4day
    #     with open(pickle_file_name, 'rb') as f:
    #         results_dict = pickle.load(f)
    #         sim_counts_mbhm = results_dict[f"sim_motif_avg_{motif_delta}"]
    #         mape_res_mbhm = results_dict['mape']
    #         print(f"\nmotif count at K=7")
    #         print(np.asarray(results_dict[f"sim_motif_avg_{motif_delta}"], dtype=int))
    #         print(mape_res_mbhm)
    #     with open(f"/shared/Results/MultiBlockHawkesModel/MotifCounts/CHIP/{dataset}/k4.p", 'rb') as f:
    #         results_dict_chip = pickle.load(f)
    #         sim_counts_chip = results_dict_chip[f"sim_motif_avg_{motif_delta}"]
    #         print(f"\nCHIP motif count at K=4")
    #         print(np.asarray(results_dict_chip[f"sim_motif_avg_{motif_delta}"], dtype=int))
    #         mape_res_chip = results_dict_chip['mape']
    #         print(mape_res_chip)
    #     with open(f"/shared/Results/MultiBlockHawkesModel/MotifCounts/BHM/{dataset}/k11.p", 'rb') as f:
    #         results_dict_bhm = pickle.load(f)
    #         sim_counts_bhm = results_dict_bhm[f"sim_motif_avg_{motif_delta}"]
    #         print(f"\nbhm motif count at K=11")
    #         print(np.asarray(results_dict_bhm[f"sim_motif_avg_{motif_delta}"], dtype=int))
    #         mape_res_bhm = results_dict_bhm['mape']
    #         print(mape_res_bhm)
    #
    #     dataset_max = np.max(dataset_counts)
    #     max_mbhm = np.max(sim_counts_mbhm)
    #     max_bhm = np.max(sim_counts_bhm)
    #     # max_chip = np.max(sim_counts_chip)
    #     vmax = np.max([dataset_max, max_mbhm, max_bhm])
    #
    #     fig, ax = plt.subplots(figsize=(5, 4))
    #     c = ax.pcolor(dataset_counts, cmap='Blues', vmin=0, vmax=vmax)
    #     ax.invert_yaxis()
    #     ax.set_xticks(np.arange(6) + 0.5)
    #     ax.set_yticks(np.arange(6) + 0.5)
    #     ax.set_xticklabels(np.arange(1, 7))
    #     ax.set_yticklabels(np.arange(1, 7))
    #     # ax.set_title(f'{dataset} Actual at delta={motif_delta}')
    #     fig.colorbar(c, ax=ax)
    #     fig.tight_layout()
    #     fig.savefig(f"{fig_path}/Actual{dataset}Motifs.pdf")
    #     plt.show()
    #
    #     fig, ax = plt.subplots(figsize=(5, 4))
    #     c = ax.pcolor(sim_counts_mbhm, cmap='Blues', vmin=0, vmax=vmax)
    #     ax.invert_yaxis()
    #     ax.set_xticks(np.arange(6) + 0.5)
    #     ax.set_yticks(np.arange(6) + 0.5)
    #     ax.set_xticklabels(np.arange(1,7))
    #     ax.set_yticklabels(np.arange(1,7))
    #     # ax.set_title(f'sim counts MAPE={mape_res_mbhm:.1f}')
    #     fig.colorbar(c, ax=ax)
    #     fig.tight_layout()
    #     fig.savefig(f"{fig_path}/mbhm_new.pdf")
    #     plt.show()
    #
    #     fig, ax = plt.subplots(figsize=(5, 4))
    #     c = ax.pcolor(sim_counts_chip, cmap='Blues', vmin=0, vmax=vmax)
    #     ax.invert_yaxis()
    #     ax.set_xticks(np.arange(6) + 0.5)
    #     ax.set_yticks(np.arange(6) + 0.5)
    #     ax.set_xticklabels(np.arange(1, 7))
    #     ax.set_yticklabels(np.arange(1, 7))
    #     # ax.set_title(f'sim counts MAPE={mape_res_chip:.1f}')
    #     fig.colorbar(c, ax=ax)
    #     fig.tight_layout()
    #     fig.savefig(f"{fig_path}/chip_new.pdf")
    #     plt.show()
    #
    #     fig, ax = plt.subplots(figsize=(5, 4))
    #     c = ax.pcolor(sim_counts_bhm, cmap='Blues', vmin=0, vmax=vmax)
    #     ax.invert_yaxis()
    #     ax.set_xticks(np.arange(6) + 0.5)
    #     ax.set_yticks(np.arange(6) + 0.5)
    #     ax.set_xticklabels(np.arange(1, 7))
    #     ax.set_yticklabels(np.arange(1, 7))
    #     # ax.set_title(f'sim counts at K=11, MAPE={mape_res_bhm:.1f}')
    #     fig.colorbar(c, ax=ax)
    #     fig.tight_layout()
    #     fig.savefig(f"{fig_path}/bhm_new.pdf")
    #     plt.show()
    #
    # if MBHM_loop:
    #     mape_res = {}
    #     mape_res_med = {}
    #     motif_counts = {}
    #     motif_counts_med = {}
    #     for K in range(3, 11):
    #         pickle_file_name = f"{results_path}/{betas}/k{K}.p" #  1week2days1_4day
    #         with open(pickle_file_name, 'rb') as f:
    #             results_dict = pickle.load(f)
    #
    #             print(f"\nmotif count at K = {K}")
    #             sim_counts = results_dict[f"sim_motif_avg_{motif_delta}"]
    #             print(np.asarray(results_dict[f"sim_motif_avg_{motif_delta}"], dtype=int))
    #             mape_res[K] = 100 / 36 * np.sum(np.abs(sim_counts - (dataset_counts+1)) / (dataset_counts+1))
    #             print("average mape = ", mape_res[K])
    #             motif_counts[K] = sim_counts
    #
    #             sim_counts_med = results_dict[f"sim_motif_median_{motif_delta}"]
    #             print(np.asarray(results_dict[f"sim_motif_median_{motif_delta}"], dtype=int))
    #             mape_res_med[K] = 100 / 36 * np.sum(np.abs(sim_counts_med - (dataset_counts + 1)) / (dataset_counts + 1))
    #             print("Median mape = ", mape_res_med[K])
    #             motif_counts_med[K] = sim_counts_med
    #
    #     plt.plot(list(mape_res.keys()), list(mape_res.values()), 'g*-', markersize=12, label="MAPE score - Average")
    #     # plt.xlabel("K")
    #     # plt.ylim((0, 500))
    #     # plt.ylabel("MAPE score")
    #     # plt.show()
    #
    #     plt.plot(list(mape_res_med.keys()), list(mape_res_med.values()), 'r*-', markersize=12, label="MAPE score - Median")
    #     plt.xlabel("K")
    #     plt.ylim((0, 300))
    #     # plt.ylabel("MAPE score - Median")
    #     plt.legend()
    #     plt.show()
    #
    #     PLOT = True
    #     median = True
    #     best1 = 5
    #     best2 = 7
    #     if PLOT:
    #         # path = "/shared/Results/MultiBlockHawkesModel/figures/figures2"
    #         if not median:
    #             sim_counts_1 = motif_counts[best1]
    #             sim_counts_2 = motif_counts[best2]
    #         else:
    #             sim_counts_1 = motif_counts_med[best1]
    #             sim_counts_2 = motif_counts_med[best2]
    #         dataset_max = np.max(dataset_counts)
    #         sim_max1 = np.max(sim_counts_1)
    #         sim_max2 = np.max(sim_counts_2)
    #         vmax = np.max([dataset_max, sim_max1, sim_max2])
    #
    #         fig, ax = plt.subplots(figsize=(5,4))
    #         c = ax.pcolor(dataset_counts, cmap='Blues', vmin=0, vmax=vmax)
    #         ax.invert_yaxis()
    #         ax.set_title(f'{dataset} Actual motifs count at delta={motif_delta}')
    #         fig.colorbar(c, ax=ax)
    #         fig.tight_layout()
    #         # fig.savefig(f"{path}/Actual{dataset}Motifs.pdf")
    #         plt.show()
    #
    #         fig, ax = plt.subplots(figsize=(5,4))
    #         c = ax.pcolor(sim_counts_1, cmap='Blues', vmin=0, vmax=vmax)
    #         ax.invert_yaxis()
    #         if not median:
    #             ax.set_title(f'sim counts at K={best1}, MAPE={mape_res[best1]:.1f}')
    #         else:
    #             ax.set_title(f'Median - sim counts at K={best1}, MAPE={mape_res_med[best1]:.1f}')
    #         fig.colorbar(c, ax=ax)
    #         fig.tight_layout()
    #         # fig.savefig(f"{path}/KernelSumFull{dataset}Motifs.pdf")
    #         plt.show()
    #
    #         fig, ax = plt.subplots(figsize=(5, 4))
    #         c = ax.pcolor(sim_counts_2, cmap='Blues', vmin=0, vmax=vmax)
    #         ax.invert_yaxis()
    #         if not median:
    #             ax.set_title(f'sim counts at K={best2}, MAPE={mape_res[best2]:.1f}')
    #         else:
    #             ax.set_title(f'Median - sim counts at K={best2}, MAPE={mape_res_med[best2]:.1f}')
    #         fig.colorbar(c, ax=ax)
    #         fig.tight_layout()
    #         # fig.savefig(f"{path}/KernelSumFull{dataset}Motifs.pdf")
    #         plt.show()
    #
    # if CHIP_loop:
    #     mape_res_chip = {}
    #     motif_counts_chip = {}
    #     rangeK = [4,5,6,7,14,20,45]
    #     for K in range(1,15):
    #         pickle_file_name = f"/shared/Results/MultiBlockHawkesModel/MotifCounts/CHIP/{dataset}/k{K}.p"
    #         with open(pickle_file_name, 'rb') as f:
    #             results_dict_chip = pickle.load(f)
    #             sim_counts_chip = results_dict_chip[f"sim_motif_avg_{motif_delta}"]
    #             print(f"\n\nmotif count at K = {K}")
    #             print(np.asarray(results_dict_chip[f"sim_motif_avg_{motif_delta}"], dtype=int))
    #             mape_res_chip[K] = results_dict_chip['mape']
    #             print(f"MAPE={mape_res_chip[K]:.1f}, recip={results_dict_chip['sim_recip_avg']:.3f},"
    #                   f", trans={results_dict_chip['sim_trans_avg']:.3f}, , n_events={results_dict_chip['sim_n_events_avg']:.0f}")
    #             motif_counts_chip[K] = sim_counts_chip
    #     plt.plot(list(mape_res_chip.keys()), list(mape_res_chip.values()), 'g*', markersize=12)
    #     plt.xlabel("K")
    #     plt.ylabel("MAPE score")
    #     plt.show()
    #
    #     PLOT = True
    #     best1_chip = 9
    #     best2_chip = 9
    #     if PLOT:
    #         # path = "/shared/Results/MultiBlockHawkesModel/figures/figures2"
    #         sim_counts_1_chip = motif_counts_chip[best1_chip]
    #         sim_counts_2_chip = motif_counts_chip[best2_chip]
    #         dataset_max = np.max(dataset_counts)
    #         sim_max1_chip = np.max(sim_counts_1_chip)
    #         sim_max2_chip = np.max(sim_counts_2_chip)
    #         vmax = np.max([dataset_max, sim_max1_chip, sim_max2_chip])
    #
    #         fig, ax = plt.subplots(figsize=(5,4))
    #         c = ax.pcolor(dataset_counts, cmap='Blues', vmin=0, vmax=vmax)
    #         ax.invert_yaxis()
    #         ax.set_title(f'{dataset} Actual count at delta={motif_delta}')
    #         fig.colorbar(c, ax=ax)
    #         fig.tight_layout()
    #         # fig.savefig(f"{path}/Actual{dataset}Motifs.pdf")
    #         plt.show()
    #
    #         fig, ax = plt.subplots(figsize=(5,4))
    #         c = ax.pcolor(sim_counts_1_chip, cmap='Blues', vmin=0, vmax=vmax)
    #         ax.invert_yaxis()
    #         ax.set_title(f'sim counts at K={best1_chip}, MAPE={mape_res_chip[best1_chip]:.1f}')
    #         fig.colorbar(c, ax=ax)
    #         fig.tight_layout()
    #         # fig.savefig(f"{path}/KernelSumFull{dataset}Motifs.pdf")
    #         plt.show()
    #
    #         fig, ax = plt.subplots(figsize=(5, 4))
    #         c = ax.pcolor(sim_counts_2_chip, cmap='Blues', vmin=0, vmax=vmax)
    #         ax.invert_yaxis()
    #         ax.set_title(f'sim counts at K={best2_chip}, MAPE={mape_res_chip[best2_chip]:.1f}')
    #         fig.colorbar(c, ax=ax)
    #         fig.tight_layout()
    #         # fig.savefig(f"{path}/KernelSumFull{dataset}Motifs.pdf")
    #         plt.show()


    # # read bhm fit results
    # # /shared/Results/MultiBlockHawkesModel/BHM_MID" list(range(1,15))+ [20, 30, 40] +list(range(45, 110, 10))
    # file_path = "/shared/Results/MultiBlockHawkesModel/MotifCounts/BHM/Enron/param_fit"
    # for K in [6, 11, 16]:
    #     file_name = f"k_{K}.p"
    #     pickle_file_name = f"{file_path}/{file_name}"
    #     with open(pickle_file_name, 'rb') as f:
    #         results_dict = pickle.load(f)
    #     print(f"K={K}:\tll={results_dict['ll_test']:.3f}")


    # results_path = f"/shared/Results/MultiBlockHawkesModel/BHM_MID"
    # # #  /shared/Results/MultiBlockHawkesModel/MotifCounts/BHM/RealityMining/param_fit
    # # # [1,6,11,16,21,26,31,36,41,46]
    # # results_path = f"/shared/Results/MultiBlockHawkesModel/Facebook/6alpha_KernelSum"
    # K_range = list(range(1,15)) + [20,30,40,45]
    # for k in K_range:
    #     pickle_file_name = f"{results_path}/k_{k}.p"
    #     with open(pickle_file_name, 'rb') as f:
    #         results_dict_s = pickle.load(f)
    #     # print(f"{results_dict_s['ll_train']:.3f}\t{results_dict_s['ll_all']:.3f}\t{results_dict_s['ll_test']:.3f}")
    #     print(f"K={k}:\ttrain={results_dict_s['ll_train']:.3f}\tall={results_dict_s['ll_all']:.3f}\ttest={results_dict_s['ll_test']:.3f}")
    # plot_paramters2(results_dict_s["fit_param"], n_alpha=6)



    # # read refinement results
    # np.set_printoptions(precision=3)
    # results_path = "/shared/Results/MultiBlockHawkesModel/Facebook/6alpha_KernelSum"
    # results_path = f"/shared/Results/MultiBlockHawkesModel/MID/6alpha_KernelSum_Ref_batch/2month2week1_2day"
    # results_path = "/shared/Results/MultiBlockHawkesModel/RealityMining/6alpha_KernelSum_Ref_batch/old/1week2days1hour"
    # results_path = "/shared/Results/MultiBlockHawkesModel/Enron/6alpha_KernelSum_Ref_batch/old/1week2days1_4day"
    # results_path = "/shared/Results/MultiBlockHawkesModel/FacebookFiltered/no_ref"
    # for k in range(1,11):
    #     pickle_file_name = f"{results_path}/k_{k}.p"
    #     with open(pickle_file_name, 'rb') as f:
    #         results_dict_s = pickle.load(f)
    #     # print(f"{results_dict_s['ll_train_sp']:.3f}\t{results_dict_s['ll_all_sp']:.3f}"
    #     #       f"\t{results_dict_s['ll_test_sp']:.3f}")
    #     print(f"{results_dict_s['ll_train_ref']:.3f}\t{results_dict_s['ll_all_ref']:.3f}\t{results_dict_s['ll_test_ref']:.3f}"
    #           f"\t{results_dict_s['fit_time(s)']/60:.3f}")
        # print(f"{results_dict_s['ll_train']:.3f}\t{results_dict_s['ll_all']:.3f}"
        #       f"\t{results_dict_s['ll_test']:.3f}\t{results_dict_s['fit_time(s)'] / 60/60:.3f}")
        # print("\n")


    # read link prediction AUC results
    auc_path = "/shared/Results/MultiBlockHawkesModel/AUC"
    file_name = f"{auc_path}/RealityMining_auc_K_2.p"
    with open(file_name, 'rb') as f:
        auc_dict = pickle.load(f)
    print(f"auc={auc_dict['avg']}, std={auc_dict['std']}, ll_test{auc_dict['ll_test']}")


    # # read no ref dataset results - ONLY for ICML submission
    # no_ref_path = "/shared/Results/MultiBlockHawkesModel/Enron/no_ref_2alpha"
    # for k in range(1,11):
    #     file_name = f"{no_ref_path}/k_{k}.p"
    #     with open(file_name, 'rb') as f:
    #         no_dict = pickle.load(f)
    #     print(f"{no_dict['ll_train']:.3f}\t{no_dict['ll_all']:.3f}\t{no_dict['ll_test']:.3f}"
    #           f"\t{no_dict['fit_time(s)']/60:.3f}")



    # # find mape score
    # path = "/shared/Results/MultiBlockHawkesModel/MotifCounts/RealityMining/no_ref_alpha2"
    # for k in range(1, 11):
    #     file_name = f"{path}/k{k}.p"
    #     with open(file_name, 'rb') as f:
    #         dict1 = pickle.load(f)
    #     print(k, dict1['mape'])


    # with open(f"Datasets_motif_counts/{motif_delta}_{dataset}_counts.p", 'rb') as f:
    #     Mcounts = pickle.load(f)
    # dataset_counts = Mcounts["dataset_motif"]
    # print(f"actual at delta = 1 {motif_delta}")
    # print(np.asarray(dataset_counts, dtype=int))
    #
    # if save:
    #     # read best of 3 models
    #     pickle_file_name = f"{results_path}/old/1week2days1hour/k7.p"  # 1week2days1_4day  2week2days1_4day
    #     with open(pickle_file_name, 'rb') as f:
    #         results_dict = pickle.load(f)
    #         sim_counts_mbhm = results_dict[f"sim_motif_avg_{motif_delta}"]
    #         mape_res_mbhm = results_dict['mape']
    #         print(f"\nmotif count at K=7")
    #         print(np.asarray(results_dict[f"sim_motif_avg_{motif_delta}"], dtype=int))
    #         print(mape_res_mbhm)
    #     with open(f"/shared/Results/MultiBlockHawkesModel/MotifCounts/RealityMining/no_ref_alpha2/k8.p", 'rb') as f:
    #         results_dict1 = pickle.load(f)
    #         sim_counts1 = results_dict1[f"sim_motif_avg_{motif_delta}"]
    #         print(f"\nCHIP motif count at K=4")
    #         print(np.asarray(results_dict1[f"sim_motif_avg_{motif_delta}"], dtype=int))
    #         mape_res1 = results_dict1['mape']
    #         print(mape_res1)
    #
    #     dataset_max = np.max(dataset_counts)
    #     max_mbhm = np.max(sim_counts_mbhm)
    #     max1 = np.max(sim_counts1)
    #     vmax = np.max([dataset_max, max_mbhm, max1])
    #
    #     fig, ax = plt.subplots(figsize=(5, 4))
    #     c = ax.pcolor(dataset_counts, cmap='Blues', vmin=0, vmax=vmax)
    #     ax.invert_yaxis()
    #     ax.set_xticks(np.arange(6) + 0.5)
    #     ax.set_yticks(np.arange(6) + 0.5)
    #     ax.set_xticklabels(np.arange(1, 7))
    #     ax.set_yticklabels(np.arange(1, 7))
    #     # ax.set_title(f'{dataset} Actual at delta={motif_delta}')
    #     fig.colorbar(c, ax=ax)
    #     fig.tight_layout()
    #     fig.savefig(f"{fig_path}/{dataset}Motifs_rebuttal.pdf")
    #     plt.show()
    #
    #     fig, ax = plt.subplots(figsize=(5, 4))
    #     c = ax.pcolor(sim_counts_mbhm, cmap='Blues', vmin=0, vmax=vmax)
    #     ax.invert_yaxis()
    #     ax.set_xticks(np.arange(6) + 0.5)
    #     ax.set_yticks(np.arange(6) + 0.5)
    #     ax.set_xticklabels(np.arange(1,7))
    #     ax.set_yticklabels(np.arange(1,7))
    #     # ax.set_title(f'MULCH (6alpha) MAPE={mape_res_mbhm:.1f}')
    #     fig.colorbar(c, ax=ax)
    #     fig.tight_layout()
    #     fig.savefig(f"{fig_path}/MULCH_6alpha_rebuttal.pdf")
    #     plt.show()
    #
    #     fig, ax = plt.subplots(figsize=(5, 4))
    #     c = ax.pcolor(sim_counts1, cmap='Blues', vmin=0, vmax=vmax)
    #     ax.invert_yaxis()
    #     ax.set_xticks(np.arange(6) + 0.5)
    #     ax.set_yticks(np.arange(6) + 0.5)
    #     ax.set_xticklabels(np.arange(1, 7))
    #     ax.set_yticklabels(np.arange(1, 7))
    #     # ax.set_title(f'MULCH (2alpha) MAPE={mape_res1:.1f}')
    #     fig.colorbar(c, ax=ax)
    #     fig.tight_layout()
    #     fig.savefig(f"{fig_path}/MULCH_2alpha_rebuttal.pdf")
    #     plt.show()






