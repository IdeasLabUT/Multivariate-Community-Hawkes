# TODO Delete file
import numpy as np
import pickle
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from sklearn import metrics

sys.path.append("./CHIP-Network-Model")
from dataset_utils import load_and_combine_nodes_for_test_train
from utils_fit_bp import cal_num_events

# # ROC curves
# results_path = "/shared/Results/MultiBlockHawkesModel/LSH_tests/BHM"
# datasets = ["RealityMining", "MID", "Enron-2", "Enron-15", "fb-forum"]
# Ks = [50, 95, 16, 14, 57]
# auc_dict = {}
# for dataset, K in zip(datasets, Ks):
#     with open(f'{results_path}/{dataset}_auc_K_{K}.p', 'rb') as file:
#         result_dict = pickle.load(file)
#     auc_dict[dataset] = result_dict["auc"]
#     print(dataset, result_dict["avg"])
#     y_runs = result_dict["y__runs"]
#     pred_runs = result_dict["pred_runs"]
#     fig, ax = plt.subplots(figsize=(5, 4))
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     for r in range(100):
#         fpr, tpr, thresholds = metrics.roc_curve(y_runs[:,:,r].flatten(), pred_runs[:,:,r].flatten(), pos_label=1)
#         plt.plot(fpr, tpr, color='darkorange', lw=2)
#         plt.xlabel('False Positive Rate', fontsize=12)
#         plt.ylabel('True Positive Rate', fontsize=12)
#         # plt.title('Receiver operating characteristic')
#         # plt.legend(loc="lower right")
#         plt.tight_layout()
#     plt.show()
#     fig.savefig(f"{results_path}/{dataset}_BHM_ROC.pdf")


def plot_motif_matix(motif, vmax, mape, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    c = ax.pcolor(motif, cmap='Blues', vmin=0, vmax=vmax)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(6) + 0.5)
    ax.set_yticks(np.arange(6) + 0.5)
    ax.set_xticklabels(np.arange(1, 7))
    ax.set_yticklabels(np.arange(1, 7))
    fig.colorbar(c, ax=ax)
    ax.set_title(f'{title}, MAPE={mape:.1f}')
    plt.show()

path = "/shared/Results/MultiBlockHawkesModel/CHIP/motif_counts/RealityMining/sim30"
for k in range(1,11):
    file_name = f"{path}/k{k}.p"
    with open(file_name, 'rb') as f:
        dict1 = pickle.load(f)
    dataset_motif = dict1["dataset_motif"]
    motif_all = dict1['sim_motif_all']
    motif_median = np.median(motif_all, axis=0)
    # mape_all = np.zeros(len(motif_all))
    # mape_all = dict1['mape_all']
    # motif_median = dict1['sim_motif_median']
    # for run in range(len(motif_all)):
        # mape_all[run] = 100 / 36 * np.sum(np.abs(motif_all[run] - (dataset_motif + 1)) / (dataset_motif + 1))
        # print(f'run={run}, mape={mape_all[run]:.1f}')
    mape_median = 100 / 36 * np.sum(np.abs(motif_median - (dataset_motif + 1)) / (dataset_motif + 1))
    print(f'{dict1["mape"]:.1f}\t{mape_median:.1f}')
    # print(f'K={k}, mape={dict1["mape"]:.1f} avg(mape)={np.average(mape_all):.1f}, SE(mape)={np.std(mape_all)/np.sqrt(dict1["n_simulation"]):.1f}\n')

    # max_ = max(dict1['dataset_motif'].max(), motif_median.max())
    # plot_motif_matix(dict1['dataset_motif'], max_, 0, 'actual')
    # plot_motif_matix(motif_median, max_, mape_median, f'K={k}')






