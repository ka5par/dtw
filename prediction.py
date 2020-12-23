import numpy as np
import pandas as pd
import stat_models
import distance_models


from tqdm import tqdm
from functools import reduce

# Step 1 separate each month of each stock.
# Calculate the distance A with DTW, DDTW, IDTW with the last month as the input and compares it against all other months.
# Choose n closest months, predict next month's return using KNN given distance and return.
# If next month's return is positive, buy one stock. Otherwise sell one unit.
# Calculate real return.
# Proceed to next month and repeat 12 times.


def read_data(prices_file, returns_file, instrument=None):

    if instrument is None:
        print("Error, choose index/etf/stock/commodity")
        return

    stocks = pd.read_csv(prices_file, index_col=0)
    stocks["monthID"] = pd.to_datetime(stocks.index.values).year * 100 + pd.to_datetime(stocks.index.values).month
    returns = pd.read_csv(returns_file, index_col=0)

    if instrument is None:
        return stocks, returns
    else:
        return stocks[["monthID", instrument]], returns[["monthID", instrument]]


def difference(df_stocks, tv_split=60):

    df_diff = df_stocks.diff().iloc[1:].fillna(0)  # First row is NA for
    df_diff["monthID"] = df_stocks["monthID"].iloc[1:]

    train_ids = np.unique(df_diff.monthID)[:-tv_split]
    test_ids = np.unique(df_diff.monthID)[-tv_split:-1]

    return df_diff, train_ids, test_ids


def indexing(df_stocks, tv_split=60, column="^GSPC"):

    df_index = df_stocks.copy()
    df_index.index = pd.to_datetime(df_index.index)
    last_month_closes = df_index.loc[df_index.groupby(df_index.index.to_period('M')).apply(lambda x: x.index.max())]
    ids = last_month_closes.monthID
    last_month_closes = last_month_closes[column].shift(1)

    for i in range(len(ids)):
        mask = df_index["monthID"] == ids[i]
        df_index.loc[mask, "index"] = df_index.loc[mask, column] / last_month_closes[i]

    df_index = df_index.dropna()

    train_ids = np.unique(df_index.monthID)[:-tv_split]
    test_ids = np.unique(df_index.monthID)[-tv_split:-1]
    df_index[column] = df_index["index"]  # <-- get rid of this bs

    return df_index, train_ids, test_ids


def calculate_distances(train_x, train_y, test_x, train_labels, predict_id, instrument):
    distances_f = np.zeros([len(train_labels), 4])

    for count, unique_id in enumerate(train_labels):
        distances_f[count, 0] = unique_id
        distances_f[count, 1] = distance_models.dtw(train_x[train_x["monthID"] == unique_id][instrument],
                                                    test_x[test_x["monthID"] == predict_id][instrument])
        distances_f[count, 2] = distance_models.twed(train_x[train_x["monthID"] == unique_id][instrument],
                                                     test_x[test_x["monthID"] == predict_id][instrument])
        distances_f[count, 3] = train_y[train_y["monthID"] == unique_id][instrument]

    output = pd.DataFrame(distances_f)
    output.columns = ["monthID", "distance_dtw", "distance_twed", "returns"]

    return output


def predict_trades(type_train, train_x, train_y, test_x, train_labels, test_labels, instrument):

    trades = np.zeros([len(test_labels), 5])

    train_x, test_x, train_y, train_labels, test_labels = [train_x.dropna(),
                                                           test_x.dropna(),
                                                           train_y.dropna(),
                                                           train_labels[~(np.isnan(train_labels))],
                                                           test_labels[~(np.isnan(test_labels))]]

    for count, predict_id in enumerate(tqdm(test_labels)):
        distances = calculate_distances(train_x,
                                        train_y,
                                        test_x,
                                        train_labels,
                                        predict_id,
                                        instrument)
        trades[count, 0] = predict_id
        trades[count, 1] = stat_models.knn(distances, "distance_dtw")
        trades[count, 2] = stat_models.knn(distances, "distance_twed")
        trades[count, 3] = stat_models.kstar(distances, "distance_dtw")
        trades[count, 4] = stat_models.kstar(distances, "distance_twed")

    output = pd.DataFrame(trades)
    output.columns = ["monthID", type_train+"dtw_knn", type_train+"twed_knn", type_train+"dtw_kstar", type_train+"twed_kstar"]

    return output


if __name__ == '__main__':

    combined_data = "data/combined_data.csv"
    next_months_returns = "data/next_month_returns.csv"
    stock_index = "^GSPC"

    stocks_f, returns_f = read_data(combined_data, next_months_returns, stock_index)

    months_out_of_sample = 120

    train_ids = np.unique(stocks_f.monthID)[:-months_out_of_sample]
    test_ids = np.unique(stocks_f.monthID)[-months_out_of_sample:-1]

    diff_stocks, diff_train_ids, diff_test_ids = difference(stocks_f, tv_split=months_out_of_sample)
    index_stocks, index_train_ids, index_test_ids = indexing(stocks_f, tv_split=months_out_of_sample)

    print("DTW")
    trade_predictions_dtw = predict_trades("DTW",
                                           stocks_f[stocks_f["monthID"].isin(train_ids)],
                                           returns_f[returns_f["monthID"].isin(train_ids)],
                                           stocks_f[stocks_f["monthID"].isin(test_ids)],
                                           train_ids,
                                           test_ids,
                                           stock_index
                                           )
    print("DDTW")
    trade_predictions_ddtw = predict_trades("DDTW",
                                            diff_stocks[diff_stocks["monthID"].isin(diff_train_ids)],
                                            returns_f[returns_f["monthID"].isin(diff_train_ids)],
                                            diff_stocks[diff_stocks["monthID"].isin(diff_test_ids)],
                                            diff_train_ids,
                                            diff_test_ids,
                                            stock_index
                                            )
    print("IDTW")
    trade_predictions_idtw = predict_trades("IDTW",
                                            index_stocks[index_stocks["monthID"].isin(index_train_ids)],
                                            returns_f[returns_f["monthID"].isin(index_train_ids)],
                                            index_stocks[index_stocks["monthID"].isin(index_test_ids)],
                                            index_train_ids,
                                            index_test_ids,
                                            stock_index
                                            )

    data_frames = [trade_predictions_dtw,
                   trade_predictions_ddtw,
                   trade_predictions_idtw]

    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['monthID'], how='outer'), data_frames)

    out_filename = stock_index + "_preds.csv"
    pd.DataFrame.to_csv(df_merged, out_filename, sep=',', na_rep='.', index=False)
