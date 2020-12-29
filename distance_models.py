from numba import jit
import numpy as np


# //TODO find more optimal nu, lambda (twed)
# //TODO investigate lcss.
# //TODO find more optimal delta, epsilon (lcss)
# //TODO make nopython JIT work. (Problems caused by multiprocessing)
# Dynamic Time Warp
# https://github.com/MJeremy2017/machine-learning-models/blob/master/Dynamic-Time-Warping/dynamic-time-warping.py
@jit(forceobj=True)
def dtw(s, t, window=15):
    n, m = len(s), len(t)

    w = np.max([window, abs(n - m)])
    dtw_matrix = np.ones((n + 1, m + 1)) * np.inf

    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(np.max([1, i - w]), np.min([m, i + w]) + 1):
            dtw_matrix[i, j] = 0

    for i in range(1, n + 1):
        for j in range(np.max([1, i - w]), np.min([m, i + w]) + 1):
            cost = abs(s[i - 1] - t[j - 1])  # Just distance
            last_min = np.min([dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]])
            dtw_matrix[i, j] = cost + last_min

    return dtw_matrix[n, m]


# Time warp edit distance
# https://en.wikipedia.org/wiki/Time_Warp_Edit_Distance
@jit(forceobj=True)
def twed(s, t, nu=1, _lambda=0.001, time_sa=None, time_sb=None):

    if time_sa is None:
        time_sa = np.arange(len(s))

    if time_sb is None:
        time_sb = np.arange(len(t))

    # Add padding
    s = np.array([0] + list(s))
    time_sa = np.array([0] + list(time_sa))
    t = np.array([0] + list(t))
    time_sb = np.array([0] + list(time_sb))

    n, m = len(s), len(t)

    # Dynamic programming
    dp = np.zeros((n, m))

    # Initialize DP Matrix and set first row and column to infinity
    dp[0, :] = np.inf
    dp[:, 0] = np.inf
    dp[0, 0] = 0

    # Compute minimal cost
    for i in range(1, n):
        for j in range(1, m):
            # Calculate and save cost of various operations
            c = np.ones((3, 1)) * np.inf

            # Deletion in A
            c[0] = (
                    dp[i - 1, j]
                    + dlp(s[i - 1], s[i])
                    + nu * (time_sa[i] - time_sa[i - 1])
                    + _lambda
            )

            # Deletion in B
            c[1] = (
                    dp[i, j - 1]
                    + dlp(t[j - 1], t[j])
                    + nu * (time_sb[j] - time_sb[j - 1])
                    + _lambda
            )

            # Keep data points in both time series
            c[2] = (
                    dp[i - 1, j - 1]
                    + dlp(s[i], t[j])
                    + dlp(s[i - 1], t[j - 1])
                    + nu * (abs(time_sa[i] - time_sb[j]) + abs(time_sa[i - 1] - time_sb[j - 1]))
            )

            # Choose the operation with the minimal cost and update DP Matrix
            dp[i, j] = np.min(c)

    distance = dp[n - 1, m - 1]

    return distance


@jit(forceobj=True)
def dlp(a, b, p=2):
    cost = np.sum(np.power(np.abs(a - b), p))
    return np.power(cost, 1 / p)


# Longest Common Subsequence for time-series
# https://github.com/ymtoo/ts-dist/blob/master/ts_dist.py
@jit(forceobj=True)
def lcss(s, t, delta, epsilon):
    n, m = len(s), len(t)
    dp = np.zeros([n+1, m+1])

    for i in range(1, n+1):
        for j in range(1, m+1):
            if np.all(np.abs(s[i-1]-t[j-1])<epsilon) and (np.abs(i-j) < delta):
                dp[i, j] = dp[i-1, j-1] + 1
            else:
                dp[i, j] = max(dp[i, j-1], dp[i-1, j])
    return 1-dp[n, m]/min(n, m)
