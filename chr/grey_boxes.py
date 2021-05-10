import numpy as np
from tqdm.autonotebook import tqdm
import copy

def smallestSubWithSum(arr, x, include=None):
    """
    Credit: https://www.geeksforgeeks.org/minimum-length-subarray-sum-greater-given-value/
    """
    n = len(arr)

    # Initialize weights if not provided
    if include is None:
        end_init = 0
        start_max = n
    else:
        end_init = include[1]
        start_max = include[0]

    # Initialize optimal solution
    start_best = 0
    end_best = n
    min_len = n + 1

    # Initialize starting index
    start = 0

    # Initialize current sum
    curr_sum = np.sum(arr[start:end_init])

    for end in range(end_init, n):
        curr_sum += arr[end]
        while (curr_sum >= x) and (start <= end) and (start <= start_max):
            if (end - start + 1 < min_len):
                min_len = end - start + 1
                start_best = start
                end_best = end

            curr_sum -= arr[start]
            start += 1

    if end_best == n:
        print("Error in smallestSubWithSum(): no solution! This may be a bug.")
        quit()

    return start_best, end_best

class HistogramAccumulator:
    def __init__(self, pi, breaks, alpha, delta_alpha=0.001):
        self.n, self.K = pi.shape
        self.breaks = breaks
        self.pi = pi
        self.alpha = alpha

        # Define grid of alpha values
        self.alpha_grid = np.round(np.arange(delta_alpha, 1.0, delta_alpha),4)

        # Make sure the target value is included
        self.alpha_grid = np.unique(np.sort(np.concatenate((self.alpha_grid,[alpha]))))

        # This is only used to predict sets rather than intervals
        self.order = np.argsort(-pi, axis=1)
        self.ranks = np.empty_like(self.order)
        for i in range(self.n):
            self.ranks[i, self.order[i]] = np.arange(len(self.order[i]))
        self.pi_sort = -np.sort(-pi, axis=1)
        self.Z = np.round(self.pi_sort.cumsum(axis=1),9)

    def compute_interval_sequence(self, epsilon=None):
        alpha_grid = self.alpha_grid
        n_grid = len(alpha_grid)
        k_star = np.where(alpha_grid==self.alpha)[0][0]
        S_grid = -np.ones((n_grid,self.n,2)).astype(int)
        S_grid_random = -np.ones((n_grid,self.n,2)).astype(int)

        # First compute optimal set for target alpha
        S, S_random = self.predict_intervals_single(alpha_grid[k_star], epsilon=epsilon)
        S_grid[k_star] = S
        S_grid_random[k_star] = S_random

        # Compute smaller sets
        for k in range(k_star+1, n_grid):
            a = S_grid[k-1,:,0]
            b = S_grid[k-1,:,1]
            S, S_random = self.predict_intervals_single(alpha_grid[k], epsilon=epsilon, a=a, b=b)
            S_grid[k] = S
            S_grid_random[k] = S_random

        # Compute larger sets
        for k in range(0,k_star)[::-1]:
            alpha = alpha_grid[k]
            S, S_random = self.predict_intervals_single(alpha, epsilon=epsilon, include=S_grid_random[k+1])
            S_grid[k] = S
            S_grid_random[k] = S_random

        return S_grid_random

    def predict_intervals_single(self, alpha, a=None, b=None, include=None, epsilon=None):

        if a is None:
            a = np.zeros(self.n).astype(int)
        if b is None:
            b = (np.ones(self.n) * len(self.pi[0])).astype(int)

        if include is None:
            include = [None] * self.n

        start = -np.ones(self.n).astype(int)
        end = -np.ones(self.n).astype(int)
        for i in range(self.n):
            start_offset = a[i]
            end_offset = b[i]+1
            if np.sum(self.pi[i][start_offset:end_offset]) < 1.0-alpha:
                print("Error: incorrect probability normalization. This may be a bug.")
                quit()
            start[i], end[i] = smallestSubWithSum(self.pi[i][start_offset:end_offset], 1.0-alpha, include=include[i])
            start[i] += start_offset
            end[i] += start_offset
        S = np.concatenate((start.reshape((len(start),1)),end.reshape((len(start),1))),1)
        S_random = copy.deepcopy(S)

        # Randomly remove one bin (either first or last) to seek exact coverage
        if (epsilon is not None):
            for i in range(self.n):
                if((S[i,-1]-S[i,0])<=1):
                    continue
                tot_weight = np.sum(self.pi[i][S[i,0]:(S[i,-1]+1)])
                excess_weight = tot_weight-(1.0-alpha)
                weight_left = self.pi[i][S[i,0]]
                weight_right = self.pi[i][S[i,-1]]
                # Remove the endpoint with the least weight (more likely to be removed)
                if weight_left < weight_right:
                    pi_remove = excess_weight / (weight_left + 1e-5)
                    if epsilon[i] <= pi_remove:
                        S_random[i,0] += 1
                else:
                    pi_remove = excess_weight / (weight_right + 1e-5)
                    if epsilon[i] <= pi_remove:
                        S_random[i,-1] -= 1

        return S, S_random

    def predict_intervals(self, alpha, epsilon=None):
        # Pre-compute list of predictive intervals
        bands = np.zeros((self.n,2))
        S_grid = self.compute_interval_sequence(epsilon=epsilon)
        j_star_idx = np.where(self.alpha_grid <= alpha)[0]
        if len(j_star_idx)==0:
            S = np.tile(np.arange(self.K), (self.n,1))
        else:
            j_star = np.max(j_star_idx)
            S = S_grid[j_star]

        bands[:,0] = self.breaks[S[:,0]-1]
        bands[:,1] = self.breaks[S[:,-1]]

        return S, bands

    def calibrate_intervals(self, Y, epsilon=None, verbose=True):
        Y = np.atleast_1d(Y)
        n2 = len(Y)
        alpha_max = np.zeros(n2,)
        S_grid = self.compute_interval_sequence(epsilon=epsilon)
        if verbose:
            print("Computing conformity scores.")
        for j in tqdm(range(len(self.alpha_grid)), disable=(not verbose)):
            alpha = self.alpha_grid[j]
            S = S_grid[j]
            band_left = self.breaks[S[:,0]-1]
            band_right = self.breaks[S[:,-1]]
            idx_inside = np.where((Y>=band_left)*(Y<=band_right))[0]
            if len(idx_inside)>0:
                alpha_max[idx_inside] = alpha

        return 1.0-alpha_max

    def predict_sets(self, alpha, epsilon=None):
        L = np.argmax(self.Z >= 1.0-alpha, axis=1).flatten()
        if epsilon is not None:
            Z_excess = np.array([ self.Z[i, L[i]] for i in range(self.n) ]) - (1.0-alpha)
            p_remove = Z_excess / np.array([ self.pi_sort[i, L[i]] for i in range(self.n) ])
            remove = epsilon <= p_remove
            for i in np.where(remove)[0]:
                L[i] = L[i] - 1
        # Return prediction set
        S = [ self.order[i,np.arange(0, L[i]+1)] for i in range(self.n) ]
        S = [ np.sort(s) for s in S]
        return(S)

    def calibrate_sets(self, Y, epsilon=None):
        Y = np.atleast_1d(Y)
        n2 = len(Y)
        ranks = np.array([ self.ranks[i,Y[i]] for i in range(n2) ])
        pi_cum = np.array([ self.Z[i,ranks[i]] for i in range(n2) ])
        pi = np.array([ self.pi_sort[i,ranks[i]] for i in range(n2) ])
        alpha_max = 1.0 - pi_cum
        if epsilon is not None:
            alpha_max += np.multiply(pi, epsilon)
        else:
            alpha_max += pi
            alpha_max = np.minimum(alpha_max, 1)
        return alpha_max
