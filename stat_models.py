import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from numba import jit


# //TODO K* see why -1 doesn't work.

@jit(forceobj=True)
def knn(distances, column):
    predictions = np.zeros(10)
    for neighbours in range(1, 11):
        neigh = KNeighborsRegressor(n_neighbors=neighbours, weights='distance')
        neigh.fit(np.array(distances[column]).reshape(-1, 1), np.array(distances.returns))
        predictions[neighbours - 1] = neigh.predict([[np.mean(np.sort(np.array(distances[column]))[:neighbours])]])
    if np.sum(predictions > 0) >= 5:
        return 1
    else:
        return -1


#  https://github.com/kfirkfir/k-Star-Nearest-Neighbors/blob/master/kStarNN.m
@jit(forceobj=True)
def kstar(distances, column):

    l_c = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]

    predictions = np.zeros(len(l_c))

    for count, lc in enumerate(l_c):
        distances = distances.sort_values(by=column)

        n = len(distances)

        beta = lc * distances[column]
        l_lambda = beta[1] + 1  # Otherwise it will never go into the while loop - go figure research papers...
        k, sum_beta, sum_beta_square = 0, 0, 0
        np.seterr(invalid='ignore')  # High change sqrt has negative value in it.
        while l_lambda > beta[k + 1] and k <= n - 3:  # -3 because otherwise will run into issues.
            k += 1
            sum_beta = sum_beta + beta[k]
            sum_beta_square = sum_beta_square + (beta[k]) ** 2
            l_lambda = (1 / k) * (sum_beta + np.sqrt(k + sum_beta ** 2 - k * sum_beta_square))

        alpha = np.zeros(n)

        for i in range(n):
            alpha[i] = np.max(l_lambda - lc * distances[column][i], 0)
            alpha[i] = alpha[i] / np.sum(alpha[i])

        predictions[count] = np.sum(alpha * distances.returns)

    if np.sum(predictions > 0) >= 5:
        return 1
    else:
        return -1
