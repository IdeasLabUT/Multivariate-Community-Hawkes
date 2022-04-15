##########################

# Implementation of MAP EM algorithm for Hawkes process
#  described in:
#  https://stmorse.github.io/docs/orc-thesis.pdf
#  https://stmorse.github.io/docs/6-867-final-writeup.pdf
# For usage see README
# For license see LICENSE
# Author: Steven Morse
# Email: steventmorse@gmail.com
# License: MIT License (see LICENSE in top folder)


# modified by Hadeel Soliman Feb 7, 2021
# modified by Hadeel Soliman May 23, 2021

##########################


import numpy as np
import matplotlib.pyplot as plt


class MHP_Kernels_2:
    def __init__(self, mu=[0.1], alpha=[[0.5]], C=[1], C_r=None, betas=[1]):
        '''
        mu: (M,) np.array
        alpha: (MxM) np.array
        C = (Q,) np.array
        C_r = (Q,) np.array for reciprocal block pair
        betas = (Q,) np.array
        '''
        self.data = []
        self.alpha, self.mu = np.array(alpha), np.array(mu),
        self.C, self.betas = np.array(C), np.array(betas)
        self.C_r = C_r
        self.dim = self.mu.shape[0]
        self.check_stability()

    def check_stability(self):
        ''' check stability of process (max alpha eigenvalue < 1)'''
        w, v = np.linalg.eig(self.alpha)
        me = np.amax(np.abs(w))
        if me >= 1.:
            print('(WARNING) Unstable.', me)


    def calc_rates(self, s):
        '''calclate processes intensity at time s'''
        rates = self.mu.copy()
        if self.C_r is None:
            for event in self.data:
                tj, uj = event[0], event[1]
                exp_term = np.sum(self.C * self.betas * np.exp(-self.betas * (s - tj)))
                rates += self.alpha[:, uj] * exp_term
        else:
            for event in self.data:
                M = int(self.dim/2)
                tj, uj = event[0], event[1]
                exp_term = np.sum(self.C * self.betas * np.exp(-self.betas * (s - tj)))
                exp_term_r = np.sum(self.C_r * self.betas * np.exp(-self.betas * (s - tj)))
                rates[:M] += self.alpha[:M,uj]*exp_term
                rates[M:] += self.alpha[M:,uj]*exp_term_r
        return rates

    def generate_seq(self, horizon):
        '''Generate a sequence based on mu, alphas, C, beta values.
        Uses Ogata's thinning method, with some speedups'''

        ## initializations ##
        self.data = []
        # Istare : hold maximum intensity (initially equal sum of baselines)
        Istar = np.sum(self.mu)
        # s : hold new event time
        # intertimes follow exponential distribution
        s = np.random.exponential(scale=1. / Istar)
        # attribute (weighted random sample, since sum(mu)==Istar)
        # n0 is process with first event
        n0 = np.random.choice(np.arange(self.dim), 1, p=(self.mu / Istar))
        n0 = int(n0)
        self.data.append([s, n0])

        # last_rates : (M,) np.array
        # holds values of lambda(t_k) where k is most recent event for each process
        # starts with just the base rate
        last_rates = self.mu.copy()
        Q = len(self.betas)
        M = int(self.dim/2) # only for off-diagonal simulation
        exp_sum_Q_last = np.zeros((Q, self.dim))

        # decrease I* (I* decreases if last event was rejected)
        decIstar = False
        while True:
            tj, uj = self.data[-1][0], self.data[-1][1]

            if decIstar:
                # if last event was rejected, decrease Istar
                Istar = np.sum(rates)
                decIstar = False
                # print(f"rejected - I* = {Istar}")
            else:
                # just had an event, increase Istar by summing over column of process uj
                if self.C_r is None: # diagonal block pair
                    Istar = np.sum(last_rates) + np.sum(self.alpha[:, uj])*np.sum(self.betas*self.C)
                    # print(f"not rejected - I* = {Istar}")
                else: # two off-diagonal block pairs
                    Istar = np.sum(last_rates) + np.sum(self.alpha[:M, uj]) * np.sum(self.betas * self.C) + \
                            np.sum(self.alpha[M:, uj]) * np.sum(self.betas * self.C_r)
                    # print(f"not rejected - I* = {Istar}")

            # generate new event
            s += np.random.exponential(scale=1. / Istar)

            # calc rates at time s (use trick to take advantage of rates at last event)
            # rates : (M,) np.array holds intensity of each process at t=s
            exp_term = np.exp(-self.betas * (s - tj)).reshape((Q, 1))
            # exp_term_repeat: (Q, M) np.array
            exp_term_repeat = np.tile(exp_term, (1, self.dim))
            if self.C_r is None:
                C_alpha = (self.C*self.betas).reshape(Q, 1) @ self.alpha[:, uj].reshape(1, self.dim)
            else:
                C_alpha_left = (self.C*self.betas).reshape(Q, 1)@self.alpha[:M, uj].reshape(1,M)
                C_alpha_right = (self.C_r*self.betas).reshape(Q, 1)@self.alpha[M:, uj].reshape(1,M)
                C_alpha = np.c_[C_alpha_left,C_alpha_right]
            exp_sum_Q = exp_term_repeat * (C_alpha + exp_sum_Q_last)
            rates = self.mu + np.sum(exp_sum_Q, axis=0)
            # print(f"trick  = {rates}")
            # print(f"detail = {self.calc_rates(s)}\n")

            # attribution/rejection test
            # handle attribution and thinning in one step as weighted random sample
            diff = Istar - np.sum(rates)
            try:
                n0 = np.random.choice(np.arange(self.dim + 1), 1,
                                      p=(np.append(rates, diff) / Istar))
                n0 = int(n0)
            except ValueError:
                # by construction this should not happen
                print('Probabilities do not sum to one.')
                self.data = np.array(self.data)
                return self.data

            if n0 < self.dim:
                # s is accepted
                self.data.append([s, n0])
                # update last_rates
                last_rates = rates.copy()
                exp_sum_Q_last = exp_sum_Q
            else:
                decIstar = True

            # if past horizon, done
            if s >= horizon:
                self.data = np.array(self.data, dtype=np.float)
                self.data = self.data[self.data[:, 0] < horizon]
                return self.data

    # -----------
    # VISUALIZATION METHODS
    # -----------

    def get_rate(self, ct, d):
        # return rate at time ct in dimension d
        seq = np.array(self.data)
        if not np.all(ct > seq[:, 0]): seq = seq[seq[:, 0] < ct]
        return self.mu[d] + \
               np.sum([self.alpha_beta[d, int(j)] * self.omega * np.exp(-self.omega * (ct - t)) for t, j in seq])

    def plot_rates(self, horizon=-1):

        if horizon < 0:
            horizon = np.amax(self.data[:, 0])

        f, axarr = plt.subplots(self.dim * 2, 1, sharex='col',
                                gridspec_kw={'height_ratios': sum([[3, 1] for i in range(self.dim)], [])},
                                figsize=(8, self.dim * 2))
        xs = np.linspace(0, horizon, (horizon / 100.) * 1000)
        for i in range(self.dim):
            row = i * 2

            # plot rate
            r = [self.get_rate(ct, i) for ct in xs]
            axarr[row].plot(xs, r, 'k-')
            axarr[row].set_ylim([-0.01, np.amax(r) + (np.amax(r) / 2.)])
            axarr[row].set_ylabel('$\lambda(t)_{%d}$' % i, fontsize=14)
            r = []

            # plot events
            subseq = self.data[self.data[:, 1] == i][:, 0]
            axarr[row + 1].plot(subseq, np.zeros(len(subseq)) - 0.5, 'bo', alpha_beta=0.2)
            axarr[row + 1].yaxis.set_visible(False)

            axarr[row + 1].set_xlim([0, horizon])

        plt.tight_layout()

    def plot_events(self, horizon=-1, showDays=True, labeled=True):
        if horizon < 0:
            horizon = np.amax(self.data[:, 0])
        fig = plt.figure(figsize=(10, 2))
        ax = plt.gca()
        for i in range(self.dim):
            subseq = self.data[self.data[:, 1] == i][:, 0]
            plt.plot(subseq, np.zeros(len(subseq)) - i, 'bo', alpha_beta=0.2)

        if showDays:
            for j in range(1, int(horizon)):
                plt.plot([j, j], [-self.dim, 1], 'k:', alpha_beta=0.15)

        if labeled:
            ax.set_yticklabels('')
            ax.set_yticks(-np.arange(0, self.dim), minor=True)
            ax.set_yticklabels([r'$e_{%d}$' % i for i in range(self.dim)], minor=True)
        else:
            ax.yaxis.set_visible(False)

        ax.set_xlim([0, horizon])
        ax.set_ylim([-self.dim, 1])
        ax.set_xlabel('Days')
        plt.tight_layout()
