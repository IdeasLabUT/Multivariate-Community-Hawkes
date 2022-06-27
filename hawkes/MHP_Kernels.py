"""
A class to simulate from multivariate Hawkes process (exponential kernel)

Modified by Hadeel Soliman to extend simulation to sum of exponential kernels
with scaling parameters.

# For license see LICENSE
# Original Author: Steven Morse
# Email: steventmorse@gmail.com
# License: MIT License (see LICENSE in top folder)

Modified by Hadeel Soliman

"""


import numpy as np


class MHP_Kernels:
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


