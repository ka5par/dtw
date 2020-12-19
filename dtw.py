import numpy as np
import pandas as pd
from numba import jit
from sklearn.neighbors import KNeighborsRegressor


# Step 1 separate each month of each stock.
# Calculate the distance A with DTW, DDTW, IDTW with the last month as the input and compares it against all other months.
# Choose n closest months, predict next month's return using KNN given distance and return.
# If next month's return is positive, buy one stock. Otherwise sell one unit.
# Calculate real return.
# Proceed to next month and repeat 12 times.

# https://github.com/MJeremy2017/machine-learning-models/blob/master/Dynamic-Time-Warping/dynamic-time-warping.py
# Distance calculations

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


def ndtw(s, t):
    return dtw(s,t)




# aux functions
def read_data(adj_close_file, returns_file, stock_f=None):

    stocks_f = pd.read_csv(adj_close_file, index_col=0)
    stocks_f["monthID"] = pd.to_datetime(stocks_f.index.values).year * 100 + pd.to_datetime(stocks_f.index.values).month
    returns_f = pd.read_csv(returns_file, index_col=0)

    if stock_f is None:
        return stocks_f, returns_f
    else:
        return stocks_f[["monthID", stock_f]], returns_f[["monthID", stock_f]]


def calculate_distances_dtw(stocks_f, returns_f, stocks_test_f, unique_ids, predict_id_1):
    distances_f = np.zeros([len(unique_ids), 3])

    for count, unique_id in enumerate(unique_ids):
        distances_f[count, 0] = unique_id
        distances_f[count, 1] = dtw(stocks_f[stocks_f["monthID"] == unique_id]["sp500"], stocks_test_f[stocks_test_f["monthID"] == predict_id_1]["sp500"])
        distances_f[count, 2] = returns_f[returns_f["monthID"] == unique_id]["sp500"]

    output = pd.DataFrame(distances_f)
    output.columns = ["monthID", "distance_dtw", "returns"]

    return output


def b_s_order(distances_f, column):

    predictions = np.zeros(10)
    for neighbours in range(1, 11):
        neigh = KNeighborsRegressor(n_neighbors=neighbours, weights='distance')
        neigh.fit(np.array(distances_f[column]).reshape(-1, 1), np.array(distances_f.returns))
        predictions[neighbours - 1] = neigh.predict([[np.mean(np.sort(np.array(distances_f[column]))[:neighbours])]])

    if np.sum(predictions > 0) > 5:
        return 1
    else:
        return -1


def predict_trades(main_label, train_x, train_y, test_x, train_labels, test_labels):
    trades = np.zeros([len(test_labels), 4])

    for count, predict_id in enumerate(test_labels):

        distances = calculate_distances_dtw(train_x,
                                            train_y,
                                            test_x,
                                            train_labels,
                                            predict_id)
        trades[count, 0] = predict_id
        trades[count, 1] = returns_test[returns_test["monthID"] == predict_id]["sp500"].values
        trades[count, 2] = b_s_order(distances, "distance_dtw")

        if count > 0:
            trades[count, 3] = trades[count - 1, 3] * (trades[count, 1] + 1)
        else:
            trades[count, 3] = trades[count, 1] + 1

        print(main_label, np.round(trades[count, :], 3))

    return trades


if __name__ == '__main__':

    stocks, returns = read_data("data/combined_sp500.csv", "data/next_month_returns.csv", stock_f="sp500")

    train_ids = np.unique(stocks.monthID)[1:-60] # Diffs don't have first row.
    test_ids = np.unique(stocks.monthID)[-60:-1]

    returns_test = returns[returns["monthID"].isin(test_ids)]

    trade_predictions = predict_trades("DTW, no augmentations.",
                                       stocks[stocks["monthID"].isin(train_ids)],
                                       returns[returns["monthID"].isin(train_ids)],
                                       stocks[stocks["monthID"].isin(test_ids)],
                                       train_ids,
                                       test_ids)
    print(trade_predictions)





