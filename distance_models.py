from numba import jit
import numpy as np


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
def twed(a, b, nu=1, _lambda=0.001):
    time_sa = np.arange(len(a))
    time_sb = np.arange(len(b))

    # Add padding
    a = np.array([0] + list(a))
    time_sa = np.array([0] + list(time_sa))
    b = np.array([0] + list(b))
    time_sb = np.array([0] + list(time_sb))

    n = len(a)
    m = len(b)
    # Dynamical programming
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
                + dlp(a[i - 1], a[i])
                + nu * (time_sa[i] - time_sa[i - 1])
                + _lambda
            )
            # Deletion in B
            c[1] = (
                dp[i, j - 1]
                + dlp(b[j - 1], b[j])
                + nu * (time_sb[j] - time_sb[j - 1])
                + _lambda
            )
            # Keep data points in both time series
            c[2] = (
                dp[i - 1, j - 1]
                + dlp(a[i], b[j])
                + dlp(a[i - 1], b[j - 1])
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
