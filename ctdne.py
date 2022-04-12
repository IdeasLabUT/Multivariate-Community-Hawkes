import numpy as np
import pandas as pd
import pickle
import random
from stellargraph import StellarGraph, StellarDiGraph
from stellargraph.data import TemporalRandomWalk
from gensim.models import Word2Vec, KeyedVectors
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append("./CHIP-Network-Model")
from dataset_utils import load_enron_train_test, load_reality_mining_test_train, load_and_combine_nodes_for_test_train


# data_path = "/nethome/hsolima/MultivariateBlockHawkesProject/MultivariateBlockHawkes/storage/datasets"
# save_path = "/shared/Results/MultiBlockHawkesModel/CTDNE"

data_path = "/data" # path from docker
save_path = "/result/CTDNE" # path frondocker

CREATE_WALK_EMBEDDINGS, TRAIN_CLF = True, True # False, False #
SAVE = False

#%% helper functions
def positive_and_negative_links(g, edges):
    pos = list(edges[["source", "target"]].itertuples(index=False))
    pos1 = list(edges[["source", "target"]].to_records(index=False))
    neg = sample_negative_examples(g, pos)
    return pos1, neg


def sample_negative_examples(g, positive_examples):
    positive_set = set(positive_examples)

    def valid_neg_edge(src, tgt):
        return (
            # no self-loops
            src != tgt
            and
            # neither direction of the edge should be a positive one
            (src, tgt) not in positive_set
            and (tgt, src) not in positive_set
        )

    possible_neg_edges = [
        (src, tgt) for src in g.nodes() for tgt in g.nodes() if valid_neg_edge(src, tgt)
    ]
    return random.sample(possible_neg_edges, k=len(positive_examples))


# binary operator for link prediction
def operator_l2(u, v):
    return (u - v) ** 2
def operator_l1(u, v):
    return np.absolute(u - v)
def hadamard(u, v):
    return np.multiply(u, v) # element-wise multiplication
def average(u, v):
    return (u+v)/2
binary_operator = operator_l2


def link_examples_to_features(link_examples, nodes_embedding):
    return [binary_operator(nodes_embedding(src), nodes_embedding(dst)) for src, dst in link_examples]


def link_prediction_classifier(max_iter=5000):
    lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=max_iter)
    return Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])


def evaluate_roc_auc(clf, link_features, link_labels):
    predicted = clf.predict_proba(link_features)
    # check which class corresponds to positive links
    positive_column = list(clf.classes_).index(1)
    return roc_auc_score(link_labels, predicted[:, positive_column])

# test
def pred_clf_and_actual(n_nodes, t0, delta, events_dict, clf, embedding):
    pred = np.zeros((n_nodes, n_nodes))  # Predicted probs that link exists
    y = np.zeros((n_nodes, n_nodes))  # actual link
    # print("classifer classes: ", clf.classes_)
    for u in range(n_nodes):
        for v in range(n_nodes):
            # classifier predictions
            uv_feature = binary_operator(embedding(u), embedding(v))
            # print("\n", uv_feature)
            uv_feature = np.reshape(uv_feature, (1, -1))  # array of ( 1 sample, embeddings size)
            pred_uv = clf.predict_proba(uv_feature)  # array of ( 1 sample, 2 classes)
            pred[u, v] = pred_uv[0, 1]  # prediction of a link
            # print("clf predictions ", clf.predict(uv_feature), "probability_1 ", pred[u, v])

            # actual link
            if (u, v) in events_dict:
                uv_times = np.array(events_dict[(u, v)])
                if len(uv_times[np.logical_and(uv_times >= t0, uv_times <= t0 + delta)]) > 0:
                    y[u, v] = 1
    return pred, y

def labelled_links(positive_examples, negative_examples):
    return (positive_examples + negative_examples, np.repeat([1, 0], [len(positive_examples), len(negative_examples)]))

def dataframe_to_events_dict(df):
    array = df.to_numpy()
    events_dict = {}
    for r in range(len(df)):
        if (array[r,0], array[r,1]) not in events_dict:
            events_dict[(array[r,0], array[r,1])] = []

        events_dict[(array[r,0], array[r,1])].append(array[r, 2])
    return events_dict


#%% read dataset
for dataset in ["RealityMining", "Enron2", "MID", "fb-forum", "EnronSmall"]:
    # "fb-forum" , "RealityMining" , "Enron2" , "MID", "EnronSmall"
    if dataset == "RealityMining" or dataset =="EnronSmall":
        if dataset == "RealityMining":
            train_path = "reality-mining/train_reality.csv"
            test_path = "reality-mining/test_reality.csv"
        else:
            train_path = "enron/train_enron.csv"
            test_path = "enron/test_enron.csv"
        # read source, target, timestamp of train dataset
        train_data = pd.read_csv(f"{data_path}/{train_path}", sep=",", header=None, names=["source", "target", "time"]
                                 , usecols=[0, 1, 2])
        test_np = np.loadtxt(f"{data_path}/{test_path}", delimiter=",")
        start_t = train_data.loc[0, "time"]
        scale = 1000 / (test_np[-1, 2] - start_t)
        # change nodes id to match CHIP dataset-reading functions
        _, train_node_id_map, _, _ = load_and_combine_nodes_for_test_train(f"{data_path}/{train_path}", f"{data_path}/{test_path}",
                                                                           remove_nodes_not_in_train=True)
        for i in range(len(train_data)):
            train_data.loc[i, "source"] = train_node_id_map[train_data.loc[i, "source"]]
            train_data.loc[i, "target"] = train_node_id_map[train_data.loc[i, "target"]]
            train_data.loc[i, "time"] = (train_data.loc[i, "time"] - start_t) *scale
        test_data = []
        for i in range(len(test_np)):
            if test_np[i, 0] in train_node_id_map and test_np[i, 1] in train_node_id_map:
                sender = train_node_id_map[test_np[i, 0]]
                receiver = train_node_id_map[test_np[i, 1]]
                time = (test_np[i, 2] - start_t) * scale
                test_data.append([sender, receiver, time])
        print(len(test_data))
        test_data = pd.DataFrame(test_data, columns=["source", "target", "time"])
        all_data = pd.concat([train_data, test_data], ignore_index=True)
    elif dataset == "Enron2":
        with open('storage/datasets/enron2/enron-events.pckl', 'rb') as f:
            n_nodes_all, end_time_all, enron_all = pickle.load(f)
        all_data = pd.DataFrame(enron_all, columns=["source", "target", "time"])
        end_time_train = 316
        train_data = all_data[all_data['time'] <= 316]
        test_data = all_data[all_data['time'] > 316]
    elif dataset == "MID":
        with open(f'./storage/datasets/MID/MID_train_all_test.p', 'rb') as file:
            train_tup, all_tup, test_set = pickle.load(file)
        # read version with nodes not in train removed and timestamped scaled [0:1000]
        train_set, end_time_train, n_nodes_train, n_events_train, id_node_map_train = train_tup
        all_set, end_time_all, n_nodes_all, n_events_all, id_node_map_all = all_tup
        all_data = pd.DataFrame(all_set, columns=["source", "target", "time"])
        train_data = pd.DataFrame(train_set, columns=["source", "target", "time"])
        test_data = pd.DataFrame(test_set, columns=["source", "target", "time"])
    elif dataset == "fb-forum":
        train_path = "fb-forum/fb_forum_train.csv"
        test_path = "fb-forum/fb_forum_test.csv"
        # read source, target, timestamp of train dataset - ignore heades
        train_data = pd.read_csv(f"{data_path}/{train_path}", sep=",", header=0, names=["source", "target", "time"], usecols=[0, 1, 2])
        test = pd.read_csv(f"{data_path}/{test_path}", sep=",", header=0, names=["source", "target", "time"], usecols=[0, 1, 2])
        # remove nodes in test not in train
        nodes_train_set = set(np.r_[train_data["source"].to_numpy(), train_data["target"].to_numpy()])
        train_node_id_map = {}
        for i, n in enumerate(nodes_train_set):
            train_node_id_map[n] = i
        test_data = []
        for i in range(len(test)):
            if test.loc[i, "source"] in train_node_id_map and test.loc[i, "target"] in train_node_id_map:
                u = test.loc[i, "source"]
                v = test.loc[i, "target"]
                test_data.append([u, v, test.loc[i, "time"]])
        test_data = pd.DataFrame(test_data, columns=["source", "target", "time"])
        n_nodes_train = n_nodes_all = len(nodes_train_set)
        end_time_train = train_data.loc[len(train_data) - 1, "time"]
        all_data = pd.concat([train_data, test_data], ignore_index=True)
        end_time_all = all_data.loc[len(all_data) - 1, "time"]

#%% code

    # create Stellar Graph
    print(f"\n\nTEST on {dataset}")
    train_per = 0.9
    train_per_length = int(len(train_data)*train_per)
    train_data_walk = train_data[:train_per_length]
    print(f"create graph and walks on {train_per} train")
    g_walk = StellarGraph(edges=train_data_walk, edge_weight_column="time")
    print(g_walk.info())

    g = StellarGraph(edges=train_data, edge_weight_column="time")

    # temporal walk hyperparameters - as recommended in CTDNE paper
    num_walks_per_node = 10 # R
    walk_length = 80 # L
    context_window_size = 10 # omega
    embedding_size = 128

    if CREATE_WALK_EMBEDDINGS:
        # construct temporal random walk
        print("Create temporal random walks")
        num_cw = len(g_walk.nodes()) * num_walks_per_node * (walk_length - context_window_size + 1)  # beta
        temporal_rw = TemporalRandomWalk(g_walk)
        temporal_walks = temporal_rw.run(num_cw=num_cw, cw_size=context_window_size, max_walk_length=walk_length,
                                         initial_edge_bias=None, walk_bias="exponential")  # "exponential"
        print("\t#walks: {}".format(len(temporal_walks)))
        # print("first walk ", temporal_walks[0])

        # Fit temporal model and create nodes embeddings - min_count:discard words that appear less than min_count
        print("Fit word2vec model temporal walks")
        temporal_model = Word2Vec(temporal_walks, vector_size=embedding_size, window=context_window_size, min_count=0, sg=1,
                                  workers=2)  # nodes_embeddings = temporal_model.wv

        # if SAVE:
        #     with open(f'{save_path}/temporal_walk_{dataset}.p', 'wb') as file:
        #         pickle.dump(temporal_walks, file) # list of lists of node ids of each walk
        #     temporal_model.save(f"{save_path}/word2vec_{dataset}.model")
        #     # nodes_embeddings.save(f"{save_path}/word2vec_{dataset}.wordvectors")
    else:
        # read saved random walk
        print("\nRead saved temporal random walks")
        with open(f'{save_path}/temporal_walk_{dataset}.p', 'rb') as f:
            temporal_walks = pickle.load(f)
        print("\nRead saved word2vec model and nodes embeddings")
        temporal_model = Word2Vec.load(f"{save_path}/word2vec_{dataset}.model")
        # nodes_embeddings = KeyedVectors.load(f"{save_path}/word2vec_{dataset}.wordvectors", mmap='r')


    # read embeddings of node (u) learned by temporal graph
    def temporal_embedding(u):
        unseen_node_embedding = np.zeros(embedding_size)  # handle unseen nodes (don't have embeddings)
        try:
            return temporal_model.wv[u] # (wv:word_vector) load nodes embeddings
        except KeyError:
            return unseen_node_embedding # all zeros embeddings


    if TRAIN_CLF:
        print("moved to test")
        # if SAVE:
        #     pos_neg_dict = {"pos": pos, "neg": neg}
        #     with open(f'{save_path}/pos_neg_{dataset}.p', 'wb') as file:
        #         pickle.dump(pos_neg_dict, file)
        #     with open(f'{save_path}/temporal_clf_{dataset}.p', 'wb') as file:
        #         pickle.dump(temporal_clf, file)
    else:
        # read saved negative samples
        print("\nRead saved Postive & negative samples")
        with open(f'{save_path}/pos_neg_{dataset}.p', 'rb') as f:
            pos_neg_dict = pickle.load(f)
        pos = pos_neg_dict["pos"]
        neg = pos_neg_dict["neg"]
        print("\nRead saved link prediction classifier")
        with open(f'{save_path}/temporal_clf_{dataset}.p', 'rb') as f:
            temporal_clf = pickle.load(f)


    # read Reality Mining dataset
    print(f"\n\nTest prediction on {dataset} dataset")
    if dataset == "RealityMining":
        train_tuple, test_tuple, all_tuple, nodes_not_in_train = load_reality_mining_test_train(remove_nodes_not_in_train=True)
        delta = 60
        events_dict_train, n_nodes_train, end_time_train = train_tuple
        events_dict_all, n_nodes_all, end_time_all = all_tuple
    elif dataset == "EnronSmall":
        train_tuple, test_tuple, all_tuple, nodes_not_in_train = load_enron_train_test(remove_nodes_not_in_train=True)
        delta = 125
        events_dict_train, n_nodes_train, end_time_train = train_tuple
        events_dict_all, n_nodes_all, end_time_all = all_tuple
    elif dataset == "Enron2":
        delta=14
        events_dict_all = dataframe_to_events_dict(all_data)
    elif dataset == "MID":
        # delta = 1.67 # 2 weeks
        delta = 7.15 # 2 month
        print(delta)
        events_dict_all = dataframe_to_events_dict(all_data)
    elif dataset == "fb-forum":
        delta=80
        events_dict_all = dataframe_to_events_dict(all_data)


    print(f"N_all={n_nodes_all}, T_train={end_time_train:.2f}, delta={delta}")

    # sample t0
    runs = 100
    auc = np.zeros(runs)
    t0s = np.zeros(runs)
    for i in range(runs):
        t0 = np.random.uniform(low=end_time_train, high=end_time_all - delta, size=None)

        # # create graph up to t0
        # train_data_t0 = all_data[all_data['time'] <= t0]
        # print(f"create graph and walks up to t0")
        # g_walk = StellarGraph(edges=train_data_t0, edge_weight_column="time")
        # print(g_walk.info())

        # Create negative samples for training classifier
        # all_data[all_data['time'] <= t0]
        train_data_clf = pd.concat([train_data[train_per_length:], test_data[test_data['time'] <= t0]], ignore_index=True)
        print(f"\ttrain clf [{train_per}train : t0] length=", len(train_data_clf))
        pos, neg = positive_and_negative_links(g, train_data_clf)
         # Create classifier train examples - link:list of (u,v) and label:list of (0 or 1)
        link_examples, link_labels = labelled_links(pos, neg)
        # define link prediction classifier - standard_scaler + logistic_regression model
        temporal_clf = link_prediction_classifier()
        temporal_link_features = link_examples_to_features(link_examples, temporal_embedding)
        print("Fit link prediction classifier")
        temporal_clf.fit(temporal_link_features, link_labels)

        # # # generate random walks up to t0 update embeddings
        # t0_data = all_data[all_data['time'] <= t0]
        # t0_g = StellarGraph(edges=t0_data, edge_weight_column="time")
        # t0_temporal_rw = TemporalRandomWalk(t0_g)
        # num_walks_per_node1 = 3
        # t0_num_cw = len(g.nodes()) * num_walks_per_node1 * (walk_length - context_window_size + 1)  # beta
        # t0_temporal_walks = t0_temporal_rw.run(num_cw=t0_num_cw, cw_size=context_window_size, max_walk_length=walk_length,
        #                                     initial_edge_bias=None, walk_bias="exponential")  # "exponential"
        # print("\t\t#walks: {}".format(len(t0_temporal_walks)))
        # # update embeddins
        # temporal_model.build_vocab(t0_temporal_walks, update=True)
        # # & total_examples=temporal_model.corpus_count total_examples=len(t0_temporal_walks)
        # print("update embeddings up to t0")
        # temporal_model.train(t0_temporal_walks, total_examples=temporal_model.corpus_count, epochs=temporal_model.epochs)

        clf_pred, actual_y = pred_clf_and_actual(n_nodes_all, t0, delta, events_dict_all, temporal_clf, temporal_embedding)
        auc[i] = roc_auc_score(actual_y.flatten(), clf_pred.flatten())
        t0s[i] = t0
        print(f"\trun({i}): t0={t0:.2f} --> auc={auc[i]:.5f}")
    print("average= ", np.average(auc))
    result_dic = {"t0s": t0s, "auc": auc, "auc_avg": np.average(auc)}
    with open(f'{save_path}/{dataset}_senario2_undirected_exp.p', 'wb') as file:
        pickle.dump(result_dic, file)  # list of lists of node ids of each walk

