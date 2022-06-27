"""Simulated Networks MULCH Experiments (Section 5.1 & Appendix B.2)

This script tests the ability of both spectral clustering and our
likelihood refinement procedure to recover true node memberships
on networks simulated from MULCH. Also, we test MULCH's
parameter Estimation Accuracy.

This file contains the following functions - more details in functions docstring:
    * spectral_clustering_accuracy()
    * refinement_accuracy()
    * parameters_estimation_MSE()

@author: Hadeel Soliman
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
import utils_fit_model as fit_model
from utils_generate_model import simulate_mulch
from utils_fit_refine_mulch import fit_refinement_mulch
from utils_fit_bp import cal_num_events


# %% helper functions
def arrage_fit_param(param, n_alpha, K, node_mem_true, node_mem_fit, betas):
    # arrange fit_parameters before print
    ar = [None] * K
    for a in range(K):
        idx = list(node_mem_true).index(a)
        ar[a] = node_mem_fit[idx]
    arranged_fit_param = []
    for i in range(n_alpha + 2):
        arranged_fit_p = np.zeros_like(param[i])
        for a in range(K):
            for b in range(K):
                arranged_fit_p[a, b] = param[i][ar[a], ar[b]]
        arranged_fit_param.append(arranged_fit_p)
    arranged_fit_param.append(betas)
    return arranged_fit_param


def get_simulation_params(n_classes, assortative):
    if assortative:  # assortative mixing - more events between diagonal block pairs
        theta_dia = [0.008, 0.3, 0.3, 0.002, 0.0005, 0.001, 0.0005]
        theta_off = [0.008, 0.1, 0.1, 0.001, 0.0001, 0.001, 0.0001]
    else:  # dissortative mixing - more events between off-diagonal block pairs
        theta_off = [0.008, 0.3, 0.3, 0.002, 0.0005, 0.001, 0.0005]
        theta_dia = [0.008, 0.1, 0.1, 0.001, 0.0001, 0.001, 0.0001]

    mu_sim = np.ones((n_classes, n_classes)) * theta_off[0]
    mu_sim[np.diag_indices_from(mu_sim)] = theta_dia[0]

    alpha_s_sim = np.ones((n_classes, n_classes)) * theta_off[1]
    alpha_s_sim[np.diag_indices_from(mu_sim)] = theta_dia[1]

    alpha_r_sim = np.ones((n_classes, n_classes)) * theta_off[2]
    alpha_r_sim[np.diag_indices_from(mu_sim)] = theta_dia[2]

    alpha_tc_sim = np.ones((n_classes, n_classes)) * theta_off[3]
    alpha_tc_sim[np.diag_indices_from(mu_sim)] = theta_dia[3]

    alpha_gr_sim = np.ones((n_classes, n_classes)) * theta_off[4]
    alpha_gr_sim[np.diag_indices_from(mu_sim)] = theta_dia[4]

    alpha_al_sim = np.ones((n_classes, n_classes)) * theta_off[5]
    alpha_al_sim[np.diag_indices_from(mu_sim)] = theta_dia[5]

    alpha_alr_sim = np.ones((n_classes, n_classes)) * theta_off[6]
    alpha_alr_sim[np.diag_indices_from(mu_sim)] = theta_dia[6]

    C_sim = np.array([[[0.33, 0.33, 0.34]] * n_classes for _ in range(n_classes)])
    betas_recip = np.array([7 * 2, 1, 1 / 12])  # [2week, 1day, 2hour]
    betas = np.reciprocal(betas_recip)

    param = (
    mu_sim, alpha_s_sim, alpha_r_sim, alpha_tc_sim, alpha_gr_sim, alpha_al_sim, alpha_alr_sim,
    C_sim, betas)

    return param


# %% simulation accuracy tests
def spectral_clustering_accuracy(n_run=10, verbose=False, plot=False):
    """
    evaluate spectral clustering accuracy as both n, T increase

    Simulate from MULCH at K=4 and #excitations=6. We use an assortative mixing parameters.

    Steps:
        - generate networks at a varying range of n (#nodes) and T (network's duration).
        - at each simulation, run spectral clustering and compute adjusted Rand Index
          between true and estimeted nodes membership

    :param n_run: # number of simulation per (n, T) values
    :param verbose: print intermediate results details
    :param plot: plot results
    :return:
    """
    K, n_alpha = 4, 6
    percent = [1 / K] * K  # nodes percentage membership
    sim_param = get_simulation_params(K, assortative=True)
    N_range = np.arange(40, 101, 15)
    T_range = np.arange(60, 241, 45)

    RI = np.zeros((len(N_range), len(T_range)))  # hold RI scores while varying n_nodes & duration
    n_events_matrix = np.zeros(
        (len(N_range), len(T_range)))  # hold simulated n_events while varying n_nodes & duration
    for T_idx, T in enumerate(T_range):
        for N_idx, N in enumerate(N_range):
            if verbose:
                print(f"\nAt duration={T}, n_nodes:{N}")
            RI_avg = 0
            n_events_avg = 0
            for it in range(n_run):
                events_dict, node_mem_true = simulate_mulch(sim_param, N, K, percent, T)
                n_events = cal_num_events(events_dict)
                agg_adj = fit_model.event_dict_to_aggregated_adjacency(N, events_dict)
                # if it == 0 and verbose:
                #     fit_model.plot_adj(agg_adj, node_mem_true, K, f"N={N}, T={T}")
                #     print("\t\t#events=", n_events)
                node_mem_spectral = fit_model.spectral_cluster1(agg_adj, K, n_kmeans_init=500,
                                                                normalize_z=True,
                                                                multiply_s=True)
                rand_i = adjusted_rand_score(node_mem_true, node_mem_spectral)
                if verbose:
                    print(f"\titer# {it}: RI={rand_i:.3f}, #events={n_events}")
                RI_avg += rand_i
                n_events_avg += n_events
            # average over runs
            RI_avg = RI_avg / n_run
            n_events_avg = n_events_avg / n_run
            if verbose:
                print("\t--> Iteration average: ", RI_avg)
            RI[N_idx, T_idx] = RI_avg
            n_events_matrix[N_idx, T_idx] = n_events_avg

    results_dict = {}
    results_dict["sim_param"] = sim_param
    results_dict["RI"] = RI
    results_dict["n_events_matrix"] = n_events_matrix
    results_dict["N_range"] = N_range
    results_dict["T_range"] = T_range
    results_dict["n_run"] = n_run

    if plot:
        fig, ax = plt.subplots(figsize=(5, 4))
        c = ax.pcolor(results_dict['RI'], cmap='Greens')
        ax.set_xticks(np.arange(5) + 0.5)
        ax.set_xticklabels(results_dict['N_range'])
        ax.set_yticks(np.arange(5) + 0.5)
        ax.set_yticklabels(results_dict['T_range'] / 30)
        ax.set_xlabel('number of nodes n')
        ax.set_ylabel('duration T (month)')
        fig.colorbar(c, ax=ax)
        ax.set_title('Spectral Clustering - Adjusted Rand Index')

    return results_dict


def parameters_estimation_MSE(fixed_n=True, n_run=10, verbose=False, plot=False):
    """
    test accuracy of model's MLE

    generate data from the MULCH model with K = 2 and #excitations=6.
    We assume parameters of the two diagonal block pairs are equal,
    and similarly, parameters of the off-diagonal block pairs are equal.
    Then, fit model and compute Mean Square Error for each MULCH parameter.

    Two tests can be done:
        - set n (#nodes in network) fixed and evaluate parameters MSE over
          range of T (duration)
        - set T (network duration) fixed, and vary n (#nodes)

    :param fixed_n: if True, set n fixed, simulate networks while varying T.
        Otherwise, fix T, and vary n.
    :param n_run: number of simulations per a pair of (n, T)
    :param verbose: print all intermediate results
    :param plot: plot results
    :return: results dictionary
    """
    K = 2
    n_alpha = 6
    percent = [1 / K] * K  # nodes percentage membership
    sim_param = get_simulation_params(K, assortative=False)
    betas = sim_param[-1]
    if fixed_n:
        n_range = np.array([70])
        T_range = np.arange(60, 241, 45)
    else:
        n_range = np.arange(40, 101, 15)
        T_range = np.array([150])

    # hold parameter's average MSE for range of (n) and (T)
    mMSE_mu = np.zeros((len(n_range), len(T_range)))
    mMSE_alpha_s = np.zeros((len(n_range), len(T_range)))
    mMSE_alpha_r = np.zeros((len(n_range), len(T_range)))
    mMSE_alpha_tc = np.zeros((len(n_range), len(T_range)))
    mMSE_alpha_gr = np.zeros((len(n_range), len(T_range)))
    mMSE_alpha_al = np.zeros((len(n_range), len(T_range)))
    mMSE_alpha_alr = np.zeros((len(n_range), len(T_range)))
    mMSE_C = np.zeros((len(n_range), len(T_range)))
    # hold parameter's standard deviation MSE for range of (n) and (T)
    sMSE_mu = np.zeros((len(n_range), len(T_range)))
    sMSE_alpha_s = np.zeros((len(n_range), len(T_range)))
    sMSE_alpha_r = np.zeros((len(n_range), len(T_range)))
    sMSE_alpha_tc = np.zeros((len(n_range), len(T_range)))
    sMSE_alpha_gr = np.zeros((len(n_range), len(T_range)))
    sMSE_alpha_al = np.zeros((len(n_range), len(T_range)))
    sMSE_alpha_alr = np.zeros((len(n_range), len(T_range)))
    sMSE_C = np.zeros((len(n_range), len(T_range)))

    for T_idx, T in enumerate(T_range):
        for N_idx, n in enumerate(n_range):
            if verbose:
                print(f"\nAt duration={T}, n_nodes:{n} ")
            # hold parameters' MSE for current run (certain value of n, T)
            MSE_mu = np.zeros(n_run)
            MSE_alpha_s = np.zeros(n_run)
            MSE_alpha_r = np.zeros(n_run)
            MSE_alpha_tc = np.zeros(n_run)
            MSE_alpha_gr = np.zeros(n_run)
            MSE_alpha_al = np.zeros(n_run)
            MSE_alpha_alr = np.zeros(n_run)
            MSE_C = np.zeros(n_run)
            for it in range(n_run):
                # simulate from mulch at a certain n, T
                events_dict, node_mem_true = simulate_mulch(sim_param, n, K, percent, T)
                n_events = cal_num_events(events_dict)
                agg_adj = fit_model.event_dict_to_aggregated_adjacency(n, events_dict)
                # run spectral clustering
                node_mem_spectral = fit_model.spectral_cluster1(agg_adj, K, n_kmeans_init=500,
                                                                normalize_z=True,
                                                                multiply_s=True)
                rand_i = adjusted_rand_score(node_mem_true, node_mem_spectral)
                if verbose:
                    print(f"\t\titer# {it}: RI={rand_i:.3f}, #events={n_events}")
                fit_param, ll_train, _ = fit_model.model_fit(n_alpha, events_dict,
                                                             node_mem_spectral, K, T,
                                                             betas)
                MSE_mu[it] = np.sum(np.square(sim_param[0] - fit_param[0]))
                MSE_alpha_s[it] = np.sum(np.square(sim_param[1] - fit_param[1]))
                MSE_alpha_r[it] = np.sum(np.square(sim_param[2] - fit_param[2]))
                MSE_alpha_tc[it] = np.sum(np.square(sim_param[3] - fit_param[3]))
                MSE_alpha_gr[it] = np.sum(np.square(sim_param[4] - fit_param[4]))
                MSE_alpha_al[it] = np.sum(np.square(sim_param[5] - fit_param[5]))
                MSE_alpha_alr[it] = np.sum(np.square(sim_param[6] - fit_param[6]))
                MSE_C[it] = np.sum(np.square(sim_param[7][:, :, :-1] - fit_param[7][:, :, :-1]))

            mMSE_mu[N_idx, T_idx] = np.mean(MSE_mu)
            mMSE_alpha_s[N_idx, T_idx] = np.mean(MSE_alpha_s)
            mMSE_alpha_r[N_idx, T_idx] = np.mean(MSE_alpha_r)
            mMSE_alpha_tc[N_idx, T_idx] = np.mean(MSE_alpha_tc)
            mMSE_alpha_gr[N_idx, T_idx] = np.mean(MSE_alpha_gr)
            mMSE_alpha_al[N_idx, T_idx] = np.mean(MSE_alpha_al)
            mMSE_alpha_alr[N_idx, T_idx] = np.mean(MSE_alpha_alr)
            mMSE_C[N_idx, T_idx] = np.mean(MSE_C)

            sMSE_mu[N_idx, T_idx] = np.std(MSE_mu)
            sMSE_alpha_s[N_idx, T_idx] = np.std(MSE_alpha_s)
            sMSE_alpha_r[N_idx, T_idx] = np.std(MSE_alpha_r)
            sMSE_alpha_tc[N_idx, T_idx] = np.std(MSE_alpha_tc)
            sMSE_alpha_gr[N_idx, T_idx] = np.std(MSE_alpha_gr)
            sMSE_alpha_al[N_idx, T_idx] = np.std(MSE_alpha_al)
            sMSE_alpha_alr[N_idx, T_idx] = np.std(MSE_alpha_alr)
            sMSE_C[N_idx, T_idx] = np.std(MSE_C)

    results_dict = {}
    results_dict["sim_param"] = sim_param
    results_dict["MSE_mean"] = (
    mMSE_mu, mMSE_alpha_s, mMSE_alpha_r, mMSE_alpha_tc, mMSE_alpha_gr, mMSE_alpha_al
    , mMSE_alpha_alr, mMSE_C)
    results_dict["MSE_std"] = (
    sMSE_mu, sMSE_alpha_s, sMSE_alpha_r, sMSE_alpha_tc, sMSE_alpha_gr, sMSE_alpha_al
    , sMSE_alpha_alr, sMSE_C)
    results_dict["N_range"] = n_range
    results_dict["T_range"] = T_range
    results_dict["n_run"] = n_run

    if plot:
        labels = ['baseline', 'self', 'reciprocal', 'turn_conti', 'generalized_recip',
                  'allied', 'allied_recip', 'kernel_scaling']
        if fixed_n:
            for i in range(len(labels)):
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.bar(results_dict['T_range'] / 30, results_dict['MSE_mean'][i].ravel(),
                       yerr=results_dict['MSE_std'][i].ravel() / np.sqrt(results_dict['n_run']))
                ax.set_xticks(results_dict['T_range'] / 30)
                ax.set_xticklabels(results_dict['T_range'] / 30)
                ax.set_xlabel('duration T (month)')
                ax.set_ylabel('mean-squared error')
                ax.set_title(f'{labels[i]} at Fixed n={results_dict["N_range"][0]}')

        else:
            for i in range(len(labels)):
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.bar(results_dict['N_range'], results_dict['MSE_mean'][i].ravel(),
                       yerr=results_dict['MSE_std'][i].ravel() / np.sqrt(results_dict['n_run']),
                       width=5,
                       color='grey')
                ax.set_xticks(results_dict['N_range'])
                ax.set_xticklabels(results_dict['N_range'])
                ax.set_xlabel('number of nodes n', fontsize=14)
                ax.set_ylabel('mean-squared error', fontsize=14)
                ax.set_title(f'{labels[i]} at fixed T={results_dict["T_range"][0]}')

    return results_dict


def refinement_accuracy(fixed_n=True, max_refine_iter=7, n_run=10, verbose=False, plot=False):
    """ Nodes membership accuracy after running log-likelihood alg

    simulate networks from MULCH at K=4 & #excitations=6, then compute adjusted rand
    index (RI) between true and estimated nodes membership. We compare RI score
    of estimated node membership after running both spectral clustering and our
    refinement algorithm.

    Two simulation test can be done:
        - simulated at (#nodes) n = 70 and vary T.
        - simulate at (network's duration) T = 105 days and vary n

    :param fixed_n: if True, set n fixed, simulate networks at varying T.
        Otherwise, fix T, and vary n.
    :param max_refine_iter: Maximum number of refinement iterations
    :param n_run: number of simulations per a pair of (n, T)
    :param verbose: print all intermediate results
    :param plot: plot results
    :return: result dictionary
    """
    K, n_alpha = 4, 6
    p = [1 / K] * K  # balanced node membership
    if fixed_n:
        N_range = np.array([70])
        T_range = np.arange(60, 241, 45)
    else:
        N_range = np.arange(40, 101, 15)
        T_range = np.array([105])

    # hold average RI scores while varying n_nodes & duration
    RI_sp = np.zeros((len(N_range), len(T_range)))
    RI_ref = np.zeros((len(N_range), len(T_range)))
    # holds RU scores over simulations
    RI_sp_runs = np.zeros((len(N_range), len(T_range), n_run))
    RI_ref_runs = np.zeros((len(N_range), len(T_range), n_run))

    # 1) simulate from 6-alpha sum of kernels model
    sim_param = get_simulation_params(K, assortative=True)
    betas = sim_param[-1]
    for T_idx, T in enumerate(T_range):
        for N_idx, N in enumerate(N_range):
            if verbose:
                print(f"\n At T={T}, N={N}:")
            ri_sp_avg = 0
            ri_ref_avg = 0
            for it in range(n_run):
                events_dict, nodes_mem_true = simulate_mulch(sim_param, N, K, p, T)
                n_events_all = cal_num_events(events_dict)
                if verbose:
                    print(f"\titer {it}: #simulated events={n_events_all}")
                # agg_adj = mulch_fit.event_dict_to_aggregated_adjacency(N, events_dict)
                # MBHP.plot_adj(agg_adj, nodes_mem_true, K, "True membership")
                sp, ref, m = fit_refinement_mulch(events_dict, N, T, K, betas, n_alpha,
                                                  max_refine_iter,
                                                  nodes_mem_true=nodes_mem_true, verbose=verbose)
                ri_sp = adjusted_rand_score(nodes_mem_true, sp[0])
                ri_ref = adjusted_rand_score(nodes_mem_true, ref[0])
                RI_sp_runs[N_idx, T_idx, it] = ri_sp
                RI_ref_runs[N_idx, T_idx, it] = ri_ref
                if verbose:
                    print(f"\t\tadjusted rand index: spectral={ri_sp:.3f}, refinement={ri_ref:.3f}")
                ri_sp_avg += ri_sp
                ri_ref_avg += ri_ref
            # average over runs
            ri_ref_avg = ri_ref_avg / n_run
            ri_sp_avg = ri_sp_avg / n_run
            if verbose:
                print(f"-->Iteration average: sp={ri_sp_avg}, ref={ri_ref_avg}")
            RI_sp[N_idx, T_idx] = ri_sp_avg
            RI_ref[N_idx, T_idx] = ri_ref_avg

    results_dict = {}
    results_dict["sim_param"] = sim_param
    results_dict["RI_sp"] = RI_sp
    results_dict["RI_ref"] = RI_ref
    results_dict["RI_sp_runs"] = RI_sp_runs
    results_dict["RI_ref_runs"] = RI_ref_runs
    results_dict["T_range"] = T_range
    results_dict["N_range"] = N_range
    results_dict["MAX_ITER"] = max_refine_iter
    results_dict["runs"] = n_run

    if plot:
        if fixed_n:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.errorbar(results_dict['T_range'], results_dict['RI_sp'].ravel(), fmt='k.-',
                        markersize=7
                        , yerr=np.std(results_dict['RI_sp_runs'], axis=2).ravel() / np.sqrt(
                    results_dict['runs'])
                        , elinewidth=1, label='spectral clustering')
            ax.errorbar(results_dict['T_range'], results_dict['RI_ref'].ravel(), fmt='g.-',
                        markersize=7
                        , yerr=np.std(results_dict['RI_ref_runs'], axis=2).ravel() / np.sqrt(
                    results_dict['runs'])
                        , elinewidth=1, label='refinement')
            plt.legend()
            plt.ylim([0, 1.05])
            ax.set_xlabel('duration T (month)')
            ax.set_ylabel('adjusted rand index')
            ax.set_xticks(results_dict['T_range'])
            ax.set_xticklabels(results_dict['T_range'] / 30)
            ax.set_title(f'fixed N={results_dict["N_range"][0]}')
        else:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.errorbar(results_dict['N_range'], results_dict['RI_sp'].ravel(), fmt='k.-',
                        markersize=7
                        , yerr=np.std(results_dict['RI_sp_runs'], axis=2).ravel() / np.sqrt(
                    results_dict['runs'])
                        , elinewidth=1, label='spectral clustering')
            ax.errorbar(results_dict['N_range'], results_dict['RI_ref'].ravel(), fmt='g.-',
                        markersize=7
                        , yerr=np.std(results_dict['RI_ref_runs'], axis=2).ravel() / np.sqrt(
                    results_dict['runs'])
                        , elinewidth=1, label='refinement')
            plt.legend()
            plt.ylim([0, 1.05])
            ax.set_xlabel('number of nodes n', fontsize=11)
            ax.set_ylabel('adjusted rand index', fontsize=11)
            ax.set_xticks(results_dict['N_range'])
            ax.set_xticklabels(results_dict['N_range'])
            ax.set_title(f'fixed T={results_dict["T_range"][0]}')

    return results_dict


# %% Main

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    # res = spectral_clustering_accuracy(verbose=True, plot=True)
    # res = parameters_estimation_MSE(fixed_n=False, verbose=False, plot=True)
    res = refinement_accuracy(fixed_n=False, verbose=True, plot=True)
