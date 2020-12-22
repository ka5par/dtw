import numpy as np
import pandas as pd

from numba import jit
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm
from functools import reduce

# Step 1 separate each month of each stock.
# Calculate the distance A with DTW, DDTW, IDTW with the last month as the input and compares it against all other months.
# Choose n closest months, predict next month's return using KNN given distance and return.
# If next month's return is positive, buy one stock. Otherwise sell one unit.
# Calculate real return.
# Proceed to next month and repeat 12 times.

# https://github.com/MJeremy2017/machine-learning-models/blob/master/Dynamic-Time-Warping/dynamic-time-warping.py

# Dynamic Time Warp
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


def twed(s, t):
    return twed_matrix[n, m]


# cuTWED
# cuda driven time warp edit distance.
# using cuda because of the nature of O(n^2) which can be divided by the amount of cuda cores.


# aux functions
def read_data(adj_close_file, returns_file, stock_f=None):
    stocks_f = pd.read_csv(adj_close_file, index_col=0)
    stocks_f["monthID"] = pd.to_datetime(stocks_f.index.values).year * 100 + pd.to_datetime(stocks_f.index.values).month
    returns_f = pd.read_csv(returns_file, index_col=0)

    if stock_f is None:
        return stocks_f, returns_f
    else:
        return stocks_f[["monthID", stock_f]], returns_f[["monthID", stock_f]]


def calculate_distances_dtw(train_x, train_y, test_x, train_labels, predict_id):
    distances_f = np.zeros([len(train_labels), 3])

    for count, unique_id in enumerate(train_labels):
        distances_f[count, 0] = unique_id
        distances_f[count, 1] = dtw(train_x[train_x["monthID"] == unique_id]["sp500"],
                                    test_x[test_x["monthID"] == predict_id]["sp500"])
        distances_f[count, 2] = train_y[train_y["monthID"] == unique_id]["sp500"]

    output = pd.DataFrame(distances_f)
    output.columns = ["monthID", "distance_dtw", "returns"]

    return output


def knn(distances_f, column):

    predictions = np.zeros(10)
    for neighbours in range(1, 10):
        neigh = KNeighborsRegressor(n_neighbors=neighbours, weights='distance')
        neigh.fit(np.array(distances_f[column]).reshape(-1, 1), np.array(distances_f.returns))
        predictions[neighbours - 1] = neigh.predict([[np.mean(np.sort(np.array(distances_f[column]))[:neighbours])]])

    if np.sum(predictions > 0) >= 5:
        return 1
    else:
        return -1


#  https://github.com/kfirkfir/k-Star-Nearest-Neighbors/blob/master/kStarNN.m
def kstar(distances_f, L_C):
    predictions = np.zeros(len(L_C))
    for count, lc in enumerate(L_C):
        distances_f = distances_f.sort_values(by=["distance_dtw"])

        n = len(distances_f)

        beta = lc * distances_f.distance_dtw
        l_lambda = beta[1] + 1  # Otherwise it will never go into the while loop - go figure research papers...
        k, sum_beta, sum_beta_square = 0, 0, 0
        np.seterr(invalid='ignore')  # High change sqrt has negative value in it.
        while l_lambda > beta[k + 1] and k <= n - 3: #  -3 because otherwise will run into issues.
            k += 1
            sum_beta = sum_beta + beta[k]
            sum_beta_square = sum_beta_square + (beta[k]) ** 2
            l_lambda = (1 / k) * (sum_beta * np.sqrt(k + sum_beta ** 2 - k * sum_beta_square))
        alpha = np.zeros(n)
        for i in range(n):
            alpha[i] = max(l_lambda - lc * distances_f.distance_dtw[i], 0)

        predictions[count] = np.sum(alpha * distances_f.returns)

    if np.sum(predictions > 0) >= 4:
        return 1
    else:
        return -1


def predict_trades_knn(type_train, train_x, train_y, test_x, train_labels, test_labels):
    trades = np.zeros([len(test_labels), 2])

    if type_train == "DTW":
        for count, predict_id in enumerate(tqdm(test_labels)):
            distances = calculate_distances_dtw(train_x,
                                                train_y,
                                                test_x,
                                                train_labels,
                                                predict_id)
            trades[count, 0] = predict_id
            trades[count, 1] = knn(distances, "distance_dtw")

    output = pd.DataFrame(trades)
    output.columns = ["monthID", "BuyOrSell"]

    return output


def predict_trades_kstar(type_train, train_x, train_y, test_x, train_labels, test_labels, L_C):
    trades = np.zeros([len(test_labels), 2])

    if type_train == "DTW":
        for count, predict_id in enumerate(tqdm(test_labels)):
            distances = calculate_distances_dtw(train_x,
                                                train_y,
                                                test_x,
                                                train_labels,
                                                predict_id)
            trades[count, 0] = predict_id
            trades[count, 1] = kstar(distances, L_C)

    output = pd.DataFrame(trades)
    output.columns = ["monthID", "BuyOrSell"]

    return output


def difference(df_stocks, tv_split=60):
    df_diff = df_stocks.diff().iloc[1:].fillna(0)  # First row is NA for
    df_diff["monthID"] = df_stocks["monthID"].iloc[1:]

    train_ids = np.unique(df_diff.monthID)[:-tv_split]
    test_ids = np.unique(df_diff.monthID)[-tv_split:-1]

    return df_diff, train_ids, test_ids


def indexing(df_stocks, tv_split=60):
    df_index = df_stocks
    df_index.index = pd.to_datetime(df_index.index)
    last_month_closes = df_index.loc[df_index.groupby(df_index.index.to_period('M')).apply(lambda x: x.index.max())]
    ids = last_month_closes.monthID
    last_month_closes = last_month_closes.sp500.shift(1)  # <-- get rid of sp500.

    for i in range(len(ids)):
        mask = df_index["monthID"] == ids[i]
        df_index.loc[mask, "index"] = df_index.loc[mask, "sp500"] / last_month_closes[i]

    df_index = df_index.dropna()

    train_ids = np.unique(df_index.monthID)[:-tv_split]
    test_ids = np.unique(df_index.monthID)[-tv_split:-1]
    df_index.sp500 = df_index["index"]  # <-- get rid of this bs

    return df_index, train_ids, test_ids


if __name__ == '__main__':

    stocks, returns = read_data("data/combined_sp500.csv", "data/next_month_returns.csv", stock_f="sp500")

    months_out_of_sample = 120

    train_ids = np.unique(stocks.monthID)[:-months_out_of_sample]
    test_ids = np.unique(stocks.monthID)[-months_out_of_sample:-1]

    diff_stocks, diff_train_ids, diff_test_ids = difference(stocks, tv_split=months_out_of_sample)
    index_stocks, index_train_ids, index_test_ids = indexing(stocks, tv_split=months_out_of_sample)

    L_C = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]  # Anava & Levy

    print("DTW KNN")
    trade_predictions_dtw = predict_trades_knn("DTW",
                                               stocks[stocks["monthID"].isin(train_ids)],
                                               returns[returns["monthID"].isin(train_ids)],
                                               stocks[stocks["monthID"].isin(test_ids)],
                                               train_ids,
                                               test_ids
                                               )
    print("DDTW KNN")
    trade_predictions_ddtw = predict_trades_knn("DTW",
                                                diff_stocks[diff_stocks["monthID"].isin(diff_train_ids)],
                                                returns[returns["monthID"].isin(diff_train_ids)],
                                                diff_stocks[diff_stocks["monthID"].isin(diff_test_ids)],
                                                diff_train_ids,
                                                diff_test_ids
                                                )
    print("IDTW KNN")
    trade_predictions_idtw = predict_trades_knn("DTW",
                                                index_stocks[index_stocks["monthID"].isin(index_train_ids)],
                                                returns[returns["monthID"].isin(index_train_ids)],
                                                index_stocks[index_stocks["monthID"].isin(index_test_ids)],
                                                index_train_ids,
                                                index_test_ids
                                                )
    print("DTW K*NN")
    trade_predictions_dtw_kstar = predict_trades_kstar("DTW",
                                                       stocks[stocks["monthID"].isin(train_ids)],
                                                       returns[returns["monthID"].isin(train_ids)],
                                                       stocks[stocks["monthID"].isin(test_ids)],
                                                       train_ids,
                                                       test_ids,
                                                       L_C
                                                       )
    print("DDTW K*NN")
    trade_predictions_ddtw_kstar = predict_trades_kstar("DTW",
                                                        diff_stocks[diff_stocks["monthID"].isin(diff_train_ids)],
                                                        returns[returns["monthID"].isin(diff_train_ids)],
                                                        diff_stocks[diff_stocks["monthID"].isin(diff_test_ids)],
                                                        diff_train_ids,
                                                        diff_test_ids,
                                                        L_C
                                                        )
    print("IDTW K*NN")
    trade_predictions_idtw_kstar = predict_trades_kstar("DTW",
                                                        index_stocks[index_stocks["monthID"].isin(index_train_ids)],
                                                        returns[returns["monthID"].isin(index_train_ids)],
                                                        index_stocks[index_stocks["monthID"].isin(index_test_ids)],
                                                        index_train_ids,
                                                        index_test_ids,
                                                        L_C
                                                        )

    data_frames = [trade_predictions_dtw,
                   trade_predictions_ddtw,
                   trade_predictions_idtw,
                   trade_predictions_dtw_kstar,
                   trade_predictions_ddtw_kstar,
                   trade_predictions_idtw_kstar]

    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['monthID'], how='outer'), data_frames)

    pd.DataFrame.to_csv(df_merged, 'data/trade_preds.csv', sep=',', na_rep='.', index=False)
