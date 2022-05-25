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






