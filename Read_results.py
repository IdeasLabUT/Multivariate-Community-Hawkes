# TODO Delete
import pickle
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
from utils_fit_model import print_mulch_param


def plot_motif_matix(motif, vmax, mape, title, save_pdf):
    fig, ax = plt.subplots(figsize=(5, 4))
    c = ax.pcolor(motif, cmap='Blues', vmin=0, vmax=vmax)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(6) + 0.5)
    ax.set_yticks(np.arange(6) + 0.5)
    ax.set_xticklabels(np.arange(1, 7))
    ax.set_yticklabels(np.arange(1, 7))
    fig.colorbar(c, ax=ax)
    fig.tight_layout()
    if save_pdf:
        fig.savefig(f"{fig_path}/{title}.pdf")
    else:
        ax.set_title(f'{title}, MAPE={mape:.1f}')
    plt.show()

#%% tests

if __name__ == "__main__":
    MBHM_loop = False
    CHIP_loop = False

    save_pdf = False


    # dataset = "RealityMining"
    # motif_delta = "week"    # 'week'or 'month'
    # results_path = f"/shared/Results/MultiBlockHawkesModel/MotifCounts/{dataset}"
    # fig_path = f"/shared/Results/MultiBlockHawkesModel/figures/{dataset}"
    #
    # with open(f"storage/datasets_motif_counts/{motif_delta}_{dataset}_counts.p", 'rb') as f:
    #     Mcounts = pickle.load(f)
    # motifs = Mcounts["dataset_motif"]
    # print(f"actual at delta = 1 {motif_delta}")
    # print(np.asarray(motifs, dtype=int))
    #
    # # read best of 3 models
    # pickle_file_name = f"{results_path}/old/1week2days1hour/k7.p"
    # with open(pickle_file_name, 'rb') as f:
    #     results_dict = pickle.load(f)
    #     motifs1 = results_dict[f"sim_motif_avg_{motif_delta}"]
    #     mape1 = results_dict['mape']
    #     print(f"\nmotif count  6alpha_K7")
    #     print(np.asarray(results_dict[f"sim_motif_avg_{motif_delta}"], dtype=int))
    #     print(mape1)
    #
    # with open(f"/shared/Results/MultiBlockHawkesModel/MotifCounts/{dataset}/4alpha/ref/k9.p", 'rb') as f:
    #     results_dict_chip = pickle.load(f)
    #     motifs2 = results_dict_chip[f"sim_motif_avg"]
    #     print(f"\nmotif count  4alpha_K9")
    #     print(np.asarray(motifs2, dtype=int))
    #     mape2 = results_dict_chip['mape']
    #     print(mape2)
    #
    # with open(f"/shared/Results/MultiBlockHawkesModel/MotifCounts/{dataset}/2alpha/ref/k3.p", 'rb') as f:
    #     results_dict_bhm = pickle.load(f)
    #     motifs3 = results_dict_bhm[f"sim_motif_avg"]
    #     print(f"\nmotif count  2alpha_K3")
    #     print(np.asarray(motifs3, dtype=int))
    #     mape3= results_dict_bhm['mape']
    #     print(mape3)
    #
    # dataset_max = np.max(motifs)
    # vmax = np.max([dataset_max, np.max(motifs1), np.max(motifs2), np.max(motifs3)])
    #
    # plot_motif_matix(motifs, vmax, 0, 'actual', save_pdf)
    # plot_motif_matix(motifs1, vmax, mape1, '6alpha_K7', save_pdf)
    # plot_motif_matix(motifs2, vmax, mape2, '4alpha_K9', save_pdf)
    # plot_motif_matix(motifs3, vmax, mape3, '2alpha_K3', save_pdf)



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
    # plot_mulch_param(results_dict_s["fit_param"], n_alpha=6)



    # # # read refinement results
    # np.set_printoptions(precision=3)
    # # results_path = "/shared/Results/MultiBlockHawkesModel/Facebook/6alpha_KernelSum"
    # # results_path = f"/shared/Results/MultiBlockHawkesModel/MID/6alpha_KernelSum_Ref_batch/2month2week1_2day"
    # results_path = "/shared/Results/MultiBlockHawkesModel/RealityMining/2alpha/ref"
    # # results_path = "/shared/Results/MultiBlockHawkesModel/Enron/6alpha_KernelSum_Ref_batch/old/1week2days1_4day"
    # # results_path = "/shared/Results/MultiBlockHawkesModel/FacebookFiltered/no_ref"
    # for k in range(1,11):
    #     pickle_file_name = f"{results_path}/k_{k}.p"
    #     with open(pickle_file_name, 'rb') as f:
    #         results_dict_s = pickle.load(f)
    #     # print(f"{results_dict_s['ll_train_sp']:.3f}\t{results_dict_s['ll_all_sp']:.3f}"
    #     #       f"\t{results_dict_s['ll_test_sp']:.3f}")
    #     print(f"{results_dict_s['ll_train_ref']:.3f}\t{results_dict_s['ll_all_ref']:.3f}\t{results_dict_s['ll_test_ref']:.3f}"
    #           f"\t{results_dict_s['fit_time_ref(s)']/60:.3f}")
    #     # print(f"{results_dict_s['ll_train_ref']:.3f}\t{results_dict_s['ll_all_ref']:.3f}"
    #     #       f"\t{results_dict_s['ll_test_ref']:.3f}\t{results_dict_s['fit_time(s)'] / 60:.3f}")



    # # # # print mulch parameters
    # pickle_file_name = f"{results_path}/k_3.p"
    # with open(pickle_file_name, 'rb') as f:
    #     fit_dict = pickle.load(f)
    # print_mulch_param(fit_dict['fit_param_ref'])


    # # read link prediction AUC results
    # auc_path = "/shared/Results/MultiBlockHawkesModel/AUC/MID/6alpha"
    # for K in range(1, 11):
    #     file_name = f"{auc_path}/auc_k{K}.p"
    #     with open(file_name, 'rb') as f:
    #         auc_dict = pickle.load(f)
    #     print(f"{auc_dict['avg']:.5f}\t{auc_dict['std']:.5f}")
    #     # print(f"auc={auc_dict['avg']}, std={auc_dict['std']}, ll_test{auc_dict['ll_test']}")

    # read link prediction AUC results
    auc_path = "/shared/Results/MultiBlockHawkesModel/BHM/AUC/Enron"
    for K in [1,2,6,11,16] + [12,13,14,15] + list(range(17,40, 2)):
        file_name = f"{auc_path}/auc_K_{K}.p"
        with open(file_name, 'rb') as f:
            auc_dict = pickle.load(f)
        print(f"{K}\t{auc_dict['avg']:.5f}\t{auc_dict['std']:.5f}")

    # # find mape score
    # path = "/shared/Results/MultiBlockHawkesModel/MotifCounts/RealityMining/6alpha/2week1day2hour"
    # for k in range(3, 11):
    #     file_name = f"{path}/k{k}.p"
    #     with open(file_name, 'rb') as f:
    #         dict1 = pickle.load(f)
    #     print(f"{dict1['mape']:.3f}")


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






