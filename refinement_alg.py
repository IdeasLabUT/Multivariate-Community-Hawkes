
import numpy as np
import time
import copy
import pickle
from sklearn.metrics import adjusted_rand_score

from spectral_clustering import spectral_cluster1

import MultiBlockFit as MBHP
import utils_sum_betas_bp as sum_betas_bp
import utils_one_beta_bp as one_beta_bp
import utils_generate_bp as generate_bp


def get_simulation_params(K):
    # simulation parameters
    if K == 3:
        mu_sim = np.array([[0.0001, 0.0001, 0.0001], [0.0003, 0.0003, 0.0003], [0.0003, 0.0001, 0.0003]])
        alpha_n_sim = np.array([[0.03, 0.03, 0.02], [0.0, 0.1, 0.01], [0.0, 0.03, 0.1]])
        alpha_r_sim = np.array([[0.01, 0.05, 0.07], [0.01, 0.01, 0.01], [0.001, 0.0, 0.35]])
        alpha_br_sim = np.array([[0.009, 0.001, 0.0001], [0.0, 0.07, 0.0006], [0.0001, 0.01, 0.05]])
        alpha_gr_sim = np.array([[0.001, 0.0, 0.0001], [0.0, 0.008, 0.0001], [0.0, 0.0002, 0.0]])
        alpha_al_sim = np.array([[0.001, 0.0001, 0.0], [0.0, 0.02, 0.0], [0.0001, 0.005, 0.01]])
        alpha_alr_sim = np.array([[0.001, 0.0001, 0.0001], [0.0, 0.001, 0.0006], [0.0001, 0.0, 0.003]])
        C_sim = np.array([[[0.33, 0.34, 0.33], [0.33, 0.34, 0.33], [0.33, 0.34, 0.33]]] * K)
        betas = np.array([0.01, 0.1, 20])
    elif K == 4:
        theta_dia = [0.0002, 0.3, 0.3, 0.004, 0.001, 0.003, 0.001]
        theta_off = [0.0002, 0.02, 0.01, 0.0002, 0.0001, 0.0002, 0.00005]
        # assortative mixing
        mu_sim = np.ones((K, K)) * theta_off[0]
        mu_sim[np.diag_indices_from(mu_sim)] = theta_dia[0]

        alpha_n_sim = np.ones((K, K)) * theta_off[1]
        alpha_n_sim[np.diag_indices_from(mu_sim)] = theta_dia[1]

        alpha_r_sim = np.ones((K, K)) * theta_off[2]
        alpha_r_sim[np.diag_indices_from(mu_sim)] = theta_dia[2]

        alpha_br_sim = np.ones((K, K)) * theta_off[3]
        alpha_br_sim[np.diag_indices_from(mu_sim)] = theta_dia[3]

        alpha_gr_sim = np.ones((K, K)) * theta_off[4]
        alpha_gr_sim[np.diag_indices_from(mu_sim)] = theta_dia[4]

        alpha_al_sim = np.ones((K, K)) * theta_off[5]
        alpha_al_sim[np.diag_indices_from(mu_sim)] = theta_dia[5]

        alpha_alr_sim = np.ones((K, K)) * theta_off[6]
        alpha_alr_sim[np.diag_indices_from(mu_sim)] = theta_dia[6]
        C_sim = np.array([[[0.33, 0.33, 0.34]] * K for _ in range(K)])
        betas_recip = np.array([7 * 2, 1, 1 / 12])  # [2week, 1day, 2hour]
        betas = np.reciprocal(betas_recip)
    param = (mu_sim, alpha_n_sim, alpha_r_sim, alpha_br_sim, alpha_gr_sim, alpha_al_sim, alpha_alr_sim, C_sim, betas)
    return param

#%% refinement OLD functions -- DELETE

def cal_diff_T_sum_q_dict(events_dict, T, betas):
    Q = len(betas)
    diff_T_sum_q = {}
    for node_p in events_dict:
        T_diff_sum_q_np = np.zeros(Q, )
        T_diff = T - events_dict[node_p]
        for q in range(Q):
            T_diff_sum_q_np[q] = np.sum(1 - np.exp(-betas[q] * T_diff))
        diff_T_sum_q[node_p] = T_diff_sum_q_np
    return diff_T_sum_q


def LL_node_pair(i, j, nodes_a, nodes_b, param_ab, events_dict, diff_T_sum_q_dict, T):
    # assume node pair (i, j) belong to block pair (a, b)
    # assume list of nodes does not include i, j
    mu, alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr, C, betas = param_ab
    n_alpha = 6
    ij_has_events = (i, j) in events_dict  # no third term if node pair (i, j) has no events
    Q = len(betas)

    # first term
    first = - mu * T

    # second and third (possibly) term
    second = 0
    if ij_has_events:
        ij_events = np.array(events_dict[(i, j)])
        n_ij_events = len(ij_events)
        ij_intertimes = (ij_events[1:] - ij_events[:-1])
        # (#ij_events-1, Q) array
        e_intertimes_Q = np.zeros((len(ij_intertimes), Q))
        for q in range(Q):
            e_intertimes_Q[:, q] = np.exp(-betas[q] * ij_intertimes)
        # n_alpha*Q columns (alpha_n, alpha_r, alpha_br, alpha_gr, alpha_al, alpha_alr)*Q
        Ri = np.zeros((n_ij_events, n_alpha * Q))

    # alpha_n
    diff = diff_T_sum_q_dict.get((i, j))
    if diff is not None:
        second -= alpha_n * (C @ diff)
        ij_has_events = True
        for e in range(1, n_ij_events):
            for q in range(Q):
                Ri[e, 0 + q * n_alpha] = e_intertimes_Q[e - 1, q] * (1 + Ri[e - 1, 0 + q * n_alpha])
    # alpha_r
    diff = diff_T_sum_q_dict.get((j, i))
    if diff is not None:
        second -= alpha_r * (C @ diff)
        if ij_has_events:
            Ri_temp = sum_betas_bp.get_Ri_temp_Q(ij_events, e_intertimes_Q, events_dict[(j, i)], betas)  # (#ij_events, Q) array
            Ri[:, 1::n_alpha] = Ri_temp

    # alpha_br and alpha_gr
    for u in nodes_b:
        # alpha_br
        diff = diff_T_sum_q_dict.get((i, u))
        if diff is not None:
            second -= alpha_br * (C @ diff)
            if ij_has_events:
                Ri_temp = sum_betas_bp.get_Ri_temp_Q(ij_events, e_intertimes_Q, events_dict[(i, u)], betas)  # (#ij_events, Q) array
                Ri[:, 2::n_alpha] = Ri_temp
        # alpha_gr
        diff = diff_T_sum_q_dict.get((u, i))
        if diff is not None:
            second -= alpha_gr * (C @ diff)
            if ij_has_events:
                Ri_temp = sum_betas_bp.get_Ri_temp_Q(ij_events, e_intertimes_Q, events_dict[(u, i)], betas)  # (#ij_events, Q) array
                Ri[:, 3::n_alpha] = Ri_temp

    # alpha_al and alpha_alr
    for u in nodes_a:
        # alpha_al
        diff = diff_T_sum_q_dict.get((u, i))
        if diff is not None:
            second -= alpha_al * (C @ diff)
            if ij_has_events:
                Ri_temp = sum_betas_bp.get_Ri_temp_Q(ij_events, e_intertimes_Q, events_dict[(u, i)], betas)  # (#ij_events, Q) array
                Ri[:, 4::n_alpha] = Ri_temp
        # alpha_alr
        diff = diff_T_sum_q_dict.get((i, u))
        if diff is not None:
            second -= alpha_alr * (C @ diff)
            if ij_has_events:
                Ri_temp = sum_betas_bp.get_Ri_temp_Q(ij_events, e_intertimes_Q, events_dict[(i, u)], betas)  # (#ij_events, Q) array
                Ri[:, 5::n_alpha] = Ri_temp

    # no third term if node pair (i, j) has no events
    if not ij_has_events:
        return first + second

    # third term
    col_sum = np.zeros(n_ij_events)
    for q in range(Q):
        col_sum[:] += C[q] * betas[q] * (
                    alpha_n * Ri[:, 0 + q * n_alpha] + alpha_r * Ri[:, 1 + q * n_alpha] + alpha_br * Ri[:, 2 + q * n_alpha] +
                    alpha_gr * Ri[:, 3 + q * n_alpha] + alpha_al * Ri[:, 4 + q * n_alpha] + alpha_alr * Ri[:, 5 + q * n_alpha])
    col_sum += mu
    third = np.sum(np.log(col_sum))
    return first + second + third

def exact_LL_node_i(i, a, param_tup, events_dict, diff_T_sum_q_dict, T, nodes_in_block):
    # let node(i) belong to block(a) & node(j) belong to block(b)
    mu_bp, alpha_n_bp, alpha_r_bp, alpha_br_bp, alpha_gr_bp, alpha_al_bp, alpha_alr_bp, C_bp, betas = param_tup
    # only calculate log-likelihood of node_pairs in block_pair(*,a) & (a,*)
    exact_ll_i = 0
    K = len(nodes_in_block)
    # loop through block_pairs(a,*) and (*, a) except (a, a)
    new_nodes_in_block_a = nodes_in_block[a].union({i})
    for b in range(K):
        if b!=a:
            param_ab = (mu_bp[a, b], alpha_n_bp[a, b], alpha_r_bp[a, b], alpha_br_bp[a, b], alpha_gr_bp[a, b],
                       alpha_al_bp[a, b], alpha_alr_bp[a, b], C_bp[a, b], betas)
            param_ba = (mu_bp[b, a], alpha_n_bp[b, a], alpha_r_bp[b, a], alpha_br_bp[b, a], alpha_gr_bp[b, a],
                        alpha_al_bp[b, a], alpha_alr_bp[b, a], C_bp[b, a], betas)
            new_nodes_in_block_b = nodes_in_block[b] - {i}
            # loop through nodes_pars(u, v) in block_pair(a, b)
            for u in new_nodes_in_block_a:
                for v in new_nodes_in_block_b:
                    # remove node_i & node_j from set of nodes
                    nodes_in_u = new_nodes_in_block_a - {u, v}
                    nodes_in_v = new_nodes_in_block_b - {u, v}
                    # log-likeihood of np(u, v)
                    exact_ll_i += LL_node_pair(u, v, nodes_in_u, nodes_in_v, param_ab, events_dict, diff_T_sum_q_dict, T)
                    # log-likeihood of np(u, v)
                    exact_ll_i += LL_node_pair(v, u, nodes_in_v, nodes_in_u, param_ba, events_dict, diff_T_sum_q_dict, T)
    # loop through node_pairs in block_pair(a, a)
    param_aa = (mu_bp[a, a], alpha_n_bp[a, a], alpha_r_bp[a, a], alpha_br_bp[a, a], alpha_gr_bp[a, a], alpha_al_bp[a, a],
                alpha_alr_bp[a, a],C_bp[a, a], betas)
    for u in new_nodes_in_block_a:
        for v in new_nodes_in_block_a:
            if u !=v :
                # remove node_i & node_j from set of nodes
                nodes_in_uv = new_nodes_in_block_a - {u, v}
                # log-likeihood of np(u, v)
                exact_ll_i += LL_node_pair(u, v, nodes_in_uv, nodes_in_uv, param_aa, events_dict, diff_T_sum_q_dict, T)
    return exact_ll_i

def relative_LL_node_i(i, a, param_tup, events_dict, diff_T_sum_q_dict, T, nodes_in_block):
    # let node(i) belong to block(a) & node(j) belong to block(b)
    mu_bp, alpha_n_bp, alpha_r_bp, alpha_br_bp, alpha_gr_bp, alpha_al_bp, alpha_alr_bp, C_bp, betas = param_tup
    # only calculate log-likelihood of node pair that has node i
    relative_ll_i = 0
    K = len(nodes_in_block)
    # loop through nodes in each block
    for b in range(K):
        param_ab = (mu_bp[a, b], alpha_n_bp[a, b], alpha_r_bp[a, b], alpha_br_bp[a, b], alpha_gr_bp[a, b],
                   alpha_al_bp[a, b], alpha_alr_bp[a, b], C_bp[a, b], betas)
        param_ba = (mu_bp[b, a], alpha_n_bp[b, a], alpha_r_bp[b, a], alpha_br_bp[b, a], alpha_gr_bp[b, a],
                    alpha_al_bp[b, a], alpha_alr_bp[b, a], C_bp[b, a], betas)
        for j in nodes_in_block[b]:
            if i != j:
                # remove node_i & node_j from set of nodes
                nodes_in_a = nodes_in_block[a] - {i, j}
                nodes_in_b = nodes_in_block[b] - {i, j}
                # log-likeihood of np(i, j)
                relative_ll_i += LL_node_pair(i, j, nodes_in_a, nodes_in_b, param_ab, events_dict, diff_T_sum_q_dict, T)
                # log-likeihood of np(j, i)
                relative_ll_i += LL_node_pair(j, i, nodes_in_b, nodes_in_a, param_ba, events_dict, diff_T_sum_q_dict, T)
    return relative_ll_i

def cal_nodes_in_class(nodes_mem, K):
    # rerur list of set nodes in each class (list of sets)
    nodes_in_class =[]
    for k in range(K):
        nodes_in_class.append(set(np.where(nodes_mem == k)[0]))
    return nodes_in_class

def nodes_mem_refinement_batch_relative(nodes_mem, events_dict, param, T, K):
    nodes_in_block = cal_nodes_in_class(nodes_mem, K)
    nodes_mem_ref = np.copy(nodes_mem)
    diff_T_sum_q_dict = cal_diff_T_sum_q_dict(events_dict, T, param[-1])
    for node_i, orig_block in enumerate(nodes_mem):
        # only try to move node if it's not in a block by itself
        if len(nodes_in_block[orig_block]) > 1:
            # holds log-likelihood scores when assigning node_i to different blocks
            node_i_LL_score = np.zeros(K)
            # loop through all block and change membership of node_i
            for to_block in range(K):
                node_i_LL_score[to_block] = relative_LL_node_i(node_i, to_block, param, events_dict,
                                                               diff_T_sum_q_dict, T, nodes_in_block)
            nodes_mem_ref[node_i] = np.argmax(node_i_LL_score)
    return nodes_mem_ref

def nodes_mem_refinement_sequen_relative(nodes_mem, events_dict, param, T, K):
    nodes_in_block = cal_nodes_in_class(nodes_mem, K)
    nodes_mem_ref = np.copy(nodes_mem)
    diff_T_sum_q_dict = cal_diff_T_sum_q_dict(events_dict, T, param[-1])
    # shuffle order of nodes
    indices = np.arange(len(nodes_mem))
    np.random.shuffle(indices)
    nodes_mem = nodes_mem[indices]
    # At each iteration --> Update events_dict_bp, N_c, LL_bp to hold best result
    for node_i, orig_block in zip(indices, nodes_mem):
        # only try to move node if it's not in a block by itself
        if len(nodes_in_block[orig_block]) > 1:
            # loop through all block and change membership of node_i
            ll_best = float('-inf')
            block_i_best = -1
            for to_block in range(K):
                # as result of moving node_i from one block to another
                ll_temp = relative_LL_node_i(node_i, to_block, param, events_dict,
                                                               diff_T_sum_q_dict, T, nodes_in_block)
                if ll_temp > ll_best:
                    ll_best = ll_temp
                    block_i_best = to_block
            # update list of nodes in block
            nodes_in_block[orig_block] = nodes_in_block[orig_block] - {node_i}
            nodes_in_block[block_i_best].add(node_i)
            nodes_mem_ref[node_i] = block_i_best
    return nodes_mem_ref

def model_fit_refine_kernel_sum_relative(max_iter, n_alpha, events_dict, N, K, end_time, betas, batch=True, nodes_mem_true = None, verbose=False):
    # 1) run spectral clustering
    print("run spectral clustering - version: s^(1/2) multiplication and row normalization")
    agg_adj = MBHP.event_dict_to_aggregated_adjacency(N, events_dict)
    nodes_mem_sp = spectral_cluster1(agg_adj, K, n_kmeans_init=500, normalize_z=True, multiply_s=True)
    if nodes_mem_true is not None:
        print(f"adusted_RI between true and sp = {adjusted_rand_score(nodes_mem_true, nodes_mem_sp):.3f}")
    if verbose:
        # MBHP.plot_adj(agg_adj, nodes_mem_sp, K, f"Spectral membership K={K}")
        classes, n_node_per_class = np.unique(nodes_mem_sp, return_counts=True)
        print("nodes/class# : ", np.sort(n_node_per_class/np.sum(n_node_per_class)))

    # 2) fit model and get parameters, LL
    fit_param, ll_sp, num_events = MBHP.model_fit_kernel_sum(n_alpha, events_dict, nodes_mem_sp, K, end_time, betas)
    if verbose:
        print("ll fit = ", ll_sp)
        print("\n")

    # 3) refinement  - stop if converged or #blocks decreased
    nodes_mem_ref00 = nodes_mem_sp
    fit_param00 = fit_param
    ll_ref, ref_time = ll_sp, 0
    message = f" ({max_iter}) max iteration reached"
    for ref_iter in range(max_iter):
        print("Refinement Iterantion #", ref_iter + 1)
        time_s = time.time()
        if batch:
            nodes_mem_ref11 = nodes_mem_refinement_batch_relative(nodes_mem_ref00, events_dict, fit_param00, end_time, K)
        else:
            nodes_mem_ref11 = nodes_mem_refinement_sequen_relative(nodes_mem_ref00, events_dict, fit_param00, end_time, K)
        time_e = time.time()
        ref_time = time_e - time_s
        print("\t(Rel) Batch itration time = ", time_e - time_s)

        # break if no change in node_membership after refinemnet iteration
        if adjusted_rand_score(nodes_mem_ref00, nodes_mem_ref11) == 1:
            print("\n--> Break: node membership converged")
            message = f"Break: node membership converged after iteration# {ref_iter+1}"
            break
        # break if number of classes decreased - nodes moving into the biggest block
        classes_ref, n_node_per_class_ref = np.unique(nodes_mem_ref11, return_counts=True)
        if len(classes_ref) < K:
            print("\n--> Break: number of classes decreased")
            print("nodes/class percentage : ", np.sort(n_node_per_class_ref / np.sum(n_node_per_class_ref)))
            message = f"Break: number of classes decreased after iteration# {ref_iter+1}"
            break
        # print("nodes/class percentage : ", np.sort(n_node_per_class_ref / np.sum(n_node_per_class_ref)))

        if nodes_mem_true is not None:
            # print(f"\nadjusted RI={adjusted_rand_score(nodes_mem_true, nodes_mem_ref1):.3f}")
            print(f"\n(rel) adjusted RI={adjusted_rand_score(nodes_mem_true, nodes_mem_ref11):.3f}")

        # fit model on new refined node membership
        fit_param11, ll_ref, num_events= MBHP. model_fit_kernel_sum(n_alpha, events_dict, nodes_mem_ref11, K, end_time,betas)
        print("(rel )ll fit ref = ", ll_ref)
        print("\n")
        nodes_mem_ref00 = nodes_mem_ref11
        fit_param00 = fit_param11

    sp_tuple = (nodes_mem_sp, fit_param, ll_sp, num_events)
    ref_tuple = (nodes_mem_ref00, fit_param00, ll_ref, num_events)
    return sp_tuple, ref_tuple, message, ref_time

def nodes_mem_refinement_sequen(nodes_mem, events_dict_bp, param, T, N_c, LL_bp):
    K = len(N_c)
    nodes_mem_ref = np.copy(nodes_mem)
    # shuffle order of nodes
    indices = np.arange(len(nodes_mem))
    np.random.shuffle(indices)
    nodes_mem = nodes_mem[indices]
    # At each iteration --> Update events_dict_bp, N_c, LL_bp to hold best result
    for node_i, from_block in zip(indices, nodes_mem):
        # only try to move node if it's not in a block by itself
        if N_c[from_block] > 1:
            # loop through all block and change membership of node_i
            from_block_new = from_block
            for to_block in range(K):
                if to_block != from_block:
                    # return new events_dictionary/block_pair, #nodes/block, log-likelihood/block_pair
                    # as result of moving node_i from one block to another
                    events_dict_bp_temp, N_c_temp, LL_bp_temp = cal_new_LL_kernel_sum_move_node(
                        param, T, node_i, from_block_new, to_block, events_dict_bp, N_c, LL_bp, batch=False)
                    if np.sum(LL_bp_temp) > np.sum(LL_bp):
                        from_block_new = to_block
                        events_dict_bp = events_dict_bp_temp
                        N_c = N_c_temp
                        LL_bp = LL_bp_temp
                        nodes_mem_ref[node_i] = to_block
    return nodes_mem_ref
#%% Exact

def cal_new_event_dict_bp(idx, from_block, to_block, events_dict_bp, K):
    events_dict_bp1 = copy.deepcopy(events_dict_bp)
    # move nodepair from (a,*) to (b,*)
    for k in range(K):
        for (i, j) in events_dict_bp[from_block][k]:
            if i == idx:
                timestamps = events_dict_bp1[from_block][k].pop((i, j))
                events_dict_bp1[to_block][k][i, j] = timestamps
    # move nodepair from (*,a) to (*,b)
    for k in range(K):
        for (i, j) in events_dict_bp[k][from_block]:
            if j == idx:
                timestamps = events_dict_bp1[k][from_block].pop((i, j))
                events_dict_bp1[k][to_block][i, j] = timestamps
    return events_dict_bp1

def cal_new_LL_kernel_sum_move_node(param_tup, T, idx, from_block, to_block, events_dict_bp, N_c, LL_bp, batch = True):
    K = len(N_c)
    if len(param_tup) == 9:
        mu_bp, alpha_n_bp, alpha_r_bp, alpha_br_bp, alpha_gr_bp, alpha_al_bp, alpha_alr_bp, C_bp, betas = param_tup
    elif len(param_tup) == 7:
        mu_bp, alpha_n_bp, alpha_r_bp, alpha_br_bp, alpha_gr_bp, C_bp, betas = param_tup
    else:
        mu_bp, alpha_n_bp, alpha_r_bp, C_bp, betas = param_tup

    # calculate events_dict_bp after moving node_i
    events_dict_bp1 = cal_new_event_dict_bp(idx, from_block, to_block, events_dict_bp, K)

    # calculate new n_nodes_per_class
    N_c1 = np.copy(N_c)
    N_c1[from_block, 0] -=1
    N_c1[to_block, 0] += 1
    # calculate new M_bp
    M_bp1 = np.matmul(N_c1, N_c1.T)- np.diagflat(N_c1)

    # calculate new log-likelihood for affected block pairs
    LL_bp1 = np.copy(LL_bp)
    for a in range(K):
        for b in range(K):
            if a==from_block or a==to_block or b==from_block or b==to_block:
                if len(param_tup) == 9:
                    par = (mu_bp[a, b], alpha_n_bp[a, b], alpha_r_bp[a, b], alpha_br_bp[a, b], alpha_gr_bp[a, b], alpha_al_bp[a, b],
                           alpha_alr_bp[a, b], C_bp[a,b], betas)
                    if a == b:
                        LL_bp1[a, b] = sum_betas_bp.LL_6_alpha_kernel_sum_dia(par, events_dict_bp1[a][b], T, N_c1[b, 0], M_bp1[a, b])
                    else:
                        LL_bp1[a, b] = sum_betas_bp.LL_6_alpha_kernel_sum_off(par, (events_dict_bp1[a][b]), (events_dict_bp1[b][a]), T,
                                                                             N_c1[b, 0], M_bp1[a, b])
                elif len(param_tup) == 7:
                    par = (mu_bp[a, b], alpha_n_bp[a, b], alpha_r_bp[a, b], alpha_br_bp[a, b], alpha_gr_bp[a, b], C_bp[a, b], betas)
                    if a == b:
                        LL_bp1[a, b] = sum_betas_bp.LL_4_alpha_kernel_sum_dia(par, events_dict_bp1[a][b], T, N_c1[b, 0], M_bp1[a, b])
                    else:
                        LL_bp1[a, b] = sum_betas_bp.LL_4_alpha_kernel_sum_off(par, (events_dict_bp1[a][b]), (events_dict_bp1[b][a]), T,
                                                                             N_c1[b, 0], M_bp1[a, b])
                elif len(param_tup) == 5:
                    par = (mu_bp[a, b], alpha_n_bp[a, b], alpha_r_bp[a, b], C_bp[a, b], betas)
                    if a == b:
                        LL_bp1[a, b] = sum_betas_bp.LL_2_alpha_kernel_sum_dia(par, events_dict_bp1[a][b], T, N_c1[b, 0], M_bp1[a, b])
                    else:
                        LL_bp1[a, b] = sum_betas_bp.LL_2_alpha_kernel_sum_off(par, (events_dict_bp1[a][b]), (events_dict_bp1[b][a]), T,
                                                                             N_c1[b, 0], M_bp1[a, b])
    if batch:
        return(np.sum(LL_bp1))
    else:
        return events_dict_bp1, N_c1, LL_bp1

def nodes_mem_refinement_batch(nodes_mem, events_dict_bp, param, T, N_c, LL_bp):
    K = len(N_c)
    current_ll = np.sum(LL_bp)
    nodes_mem_ref = np.copy(nodes_mem)
    for node_i, from_block in enumerate(nodes_mem):
        # only try to move node if it's not in a block by itself
        if N_c[from_block] > 1:
            # holds log-likelihood scores when assigning node_i to different blocks
            node_i_LL_score = np.zeros(K)
            node_i_LL_score[from_block] = current_ll
            # loop through all block and change membership of node_i
            for to_block in range(K):
                if to_block != from_block:
                    node_i_LL_score[to_block] = cal_new_LL_kernel_sum_move_node(param, T, node_i, from_block, to_block,
                                                                                events_dict_bp, N_c, LL_bp)
            nodes_mem_ref[node_i] = np.argmax(node_i_LL_score)
    return nodes_mem_ref

# Model fit and refine functions
def model_fit_refine_kernel_sum_exact(events_dict, N, end_time, K, betas, n_alpha=6, max_iter=0, nodes_mem_true=None,
                                      verbose=False):
    # nodes_mem_true: only for simulation data

    # 1) run spectral clustering
    print("\nRun spectral clustering and fit mulch")
    start_fit_time = time.time()
    agg_adj = MBHP.event_dict_to_aggregated_adjacency(N, events_dict)
    nodes_mem0 = spectral_cluster1(agg_adj, K, n_kmeans_init=500, normalize_z=True, multiply_s=True)
    if nodes_mem_true is not None:
        print(f"\tadusted RI between true and sp = {adjusted_rand_score(nodes_mem_true, nodes_mem0):.3f}")

    # if verbose:
    #     MBHP.plot_adj(agg_adj, nodes_mem0, K, f"Spectral membership K={K}")
    #     classes, n_node_per_class = np.unique(nodes_mem0, return_counts=True)
    #     print("\tnodes/class# : ", np.sort(n_node_per_class/np.sum(n_node_per_class)))

    # 2) fit model and get parameters, LL
    fit_param0, LL_bp0, num_events, events_dict_bp0, N_c0 = MBHP.model_fit_kernel_sum(n_alpha, events_dict, nodes_mem0,
                                                                                  K, end_time, betas, ref=True)
    spectral_fit_time = time.time() - start_fit_time
    print("\tlog-likelihood = ", np.sum(LL_bp0))
    # MBHP.print_model_param_kernel_sum(fit_param0)

    sp_tuple = (nodes_mem0, fit_param0, np.sum(LL_bp0), num_events)
    # no refinement needed if K=1
    if K == 1:
        ref_tuple = (nodes_mem0, fit_param0, np.sum(LL_bp0), num_events, spectral_fit_time)
        return sp_tuple, ref_tuple, "No refinement needed at K=1"
    if max_iter == 0:
        return sp_tuple, "No refinement"

    # 3) refinement - stop if converged or #blocks decreased
    message = f"Max #iterations ({max_iter}) reached"
    for ref_iter in range(max_iter):
        print("\nRefinement iteration #", ref_iter + 1)
        nodes_mem1 = nodes_mem_refinement_batch(nodes_mem0, events_dict_bp0, fit_param0, end_time, N_c0, LL_bp0)

        # break if no change in node_membership after refinement iteration
        if adjusted_rand_score(nodes_mem0, nodes_mem1) == 1:
            message = f"Break: node membership converged at iter# {ref_iter+1}"
            print(f"\t--> {message}")
            break

        # break if number of classes decreased - nodes moving into the biggest block
        classes_ref, n_node_per_class_ref = np.unique(nodes_mem1, return_counts=True)
        if len(classes_ref) < K:
            # print("nodes/class percentage : ", np.sort(n_node_per_class_ref / np.sum(n_node_per_class_ref)))
            message = f"Break: number of classes decreased at iter# {ref_iter+1}"
            print(f"\t--> {message}")
            break

        # MBHP.plot_adj(agg_adj, nodes_mem1, K, f"Refinement membership K={K}, iteration={ref_iter}")
        if nodes_mem_true is not None:
            print(f"\tadjusted RI={adjusted_rand_score(nodes_mem_true, nodes_mem1):.3f}")

        # fit model on refined node membership
        fit_param1, LL_bp1, num_events, events_dict_bp1, N_c1 = MBHP.model_fit_kernel_sum(n_alpha, events_dict, nodes_mem1,
                                                                                     K, end_time, betas, ref=True)
        # break if train log-likelihood decreased
        if np.sum(LL_bp1) < np.sum(LL_bp0):
            message = f"Break: train ll decreased from {np.sum(LL_bp0):.1f} to {np.sum(LL_bp1):.1f} at iter# {ref_iter + 1}"
            print(f"\t--> {message}")
            break

        print("\tlog-likelihood = ", np.sum(LL_bp1))

        # MBHP.print_model_param_kernel_sum(fit_param1)
        # set new argument for next loop
        nodes_mem0 = nodes_mem1
        events_dict_bp0 = events_dict_bp1
        fit_param0 = fit_param1
        N_c0 = N_c1
        LL_bp0 = LL_bp1
    refinement_fit_time = time.time() - start_fit_time
    ref_tuple = (nodes_mem0, fit_param0, np.sum(LL_bp0), num_events, refinement_fit_time)
    return sp_tuple, ref_tuple, message


#%% Tests

def refinement_experiement():
    K = 4
    N_range = np.array([70])  # np.arange(40, 101, 15) np.array([70])
    T_range = np.arange(600, 1401, 200)  # np.arange(600, 1401, 200) np.array([1000])
    RI_sp = np.zeros((len(N_range), len(T_range)))  # hold RI scores while varying n_nodes & duration
    RI_ref = np.zeros((len(N_range), len(T_range)))  # hold RI scores while varying n_nodes & duration

    n_alpha = 6
    p = [1 / K] * K  # balanced node membership
    MAX_ITER = 10
    runs = 10

    # 1) simulate from 6-alpha sum of kernels model
    sim_param = get_simulation_params(K)
    betas = sim_param[-1]
    for T_idx, T in enumerate(T_range):
        for N_idx, N in enumerate(N_range):
            print(f"\n\nK={K}, N={N}")
            ri_sp_avg = 0
            ri_ref_avg = 0
            for it in range(runs):
                events_dict, nodes_mem_true = MBHP.simulate_sum_kernel_model(sim_param, N, K, p, T)
                n_events_all = sum_betas_bp.cal_num_events(events_dict)
                print("n_events simulated = ", n_events_all)
                agg_adj = MBHP.event_dict_to_aggregated_adjacency(N, events_dict)
                # MBHP.plot_adj(agg_adj, nodes_mem_true, K, "True membership")
                sp, ref, m = model_fit_refine_kernel_sum_exact(events_dict, N, T, K, betas, n_alpha, MAX_ITER,
                                                               nodes_mem_true=nodes_mem_true, verbose=False)
                ri_sp = adjusted_rand_score(nodes_mem_true, sp[0])
                ri_ref = adjusted_rand_score(nodes_mem_true, ref[0])
                print(f"K={K}, N={N}: iter# {it}: sp={ri_sp}, ref{ri_ref}")
                ri_sp_avg += ri_sp
                ri_ref_avg += ri_ref
            # average over runs
            ri_ref_avg = ri_ref_avg / runs
            ri_sp_avg = ri_sp_avg / runs
            print(f"Iteration average: sp={ri_sp_avg}, ref{ri_ref_avg}")
            RI_sp[N_idx, T_idx] = ri_sp_avg
            RI_ref[N_idx, T_idx] = ri_ref_avg

    results_dict = {}
    results_dict["sim_param"] = sim_param
    results_dict["RI_sp"] = RI_sp
    results_dict["RI_ref"] = RI_ref
    results_dict["T_range"] = T_range
    results_dict["N_range"] = N_range
    results_dict["MAX_ITER"] = MAX_ITER
    results_dict["runs"] = runs
    with open(f"sp_ref_K{K}_N70_last.p", 'wb') as fil:
        pickle.dump(results_dict, fil)

def test_refinement():
    K, N, T_all = 3, 60, 1000  # number of nodes and duration
    n_alpha = 6
    # p = [1 / K] * K  # balanced node membership
    p = [0.70, 0.15, 0.15]

    # 1) simulate from 6-alpha sum of kernels model
    sim_param = get_simulation_params(K)
    betas = sim_param[-1]
    print(f"{n_alpha}-alpha Sum of Kernels model simultion at K={K}, N={N}, not balanced membership")
    print("betas = ", betas)
    events_dict, nodes_mem_true = MBHP.simulate_sum_kernel_model(sim_param, N, K, p, T_all)
    n_events_all = sum_betas_bp.cal_num_events(events_dict)
    print("n_events simulated = ", n_events_all)
    agg_adj = MBHP.event_dict_to_aggregated_adjacency(N, events_dict)
    MBHP.plot_adj(agg_adj, nodes_mem_true, K, "True membership")

    MAX_ITER = 15
    model_fit_refine_kernel_sum_exact(events_dict, N, T_all, K, betas, n_alpha, MAX_ITER, nodes_mem_true=nodes_mem_true,
                                      verbose=True)
#%% main

if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    test_refinement()




