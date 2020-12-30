import numpy as np
import pandas as pd
import os
import stat_models
import distance_models

from tqdm import tqdm
from functools import reduce
from joblib import Parallel, delayed

# Step 1 separate each month of each stock.
# Calculate the distance A with DTW, DDTW, IDTW with the last month as the input and compares it against all other months.
# Choose n closest months, predict next month's return using KNN given distance and return.
# If next month's return is positive, buy one stock. Otherwise sell one unit.
# Calculate real return.
# Proceed to next month and repeat x times.


def read_data(prices_file, returns_file):

    stocks = pd.read_csv(prices_file, index_col=0)
    stocks["monthID"] = pd.to_datetime(stocks.index.values).year * 100 + pd.to_datetime(stocks.index.values).month
    returns = pd.read_csv(returns_file, index_col=0)

    return stocks, returns


def difference(df_stocks, tv_split=60):

    df_diff = df_stocks.diff().iloc[1:]  # First row is NA
    df_diff["monthID"] = df_stocks["monthID"].iloc[1:]

    train_ids = np.unique(df_diff.monthID)[:-tv_split]
    test_ids = np.unique(df_diff.monthID)[-tv_split:-1]

    return df_diff, train_ids, test_ids


def indexing(df_stocks, tv_split=60):

    df_index = df_stocks.copy()
    df_index.index = pd.to_datetime(df_index.index)
    last_month_closes = df_index.loc[df_index.groupby(df_index.index.to_period('M')).apply(lambda x: x.index.max())]

    ids = last_month_closes.monthID
    last_month_closes = last_month_closes["Close"].shift(1)

    for i in range(len(ids)):
        mask = df_index["monthID"] == ids[i]
        df_index.loc[mask, "Close"] = df_index.loc[mask, "Close"] / last_month_closes[i]

    df_index = df_index.dropna()

    train_ids = np.unique(df_index.monthID)[:-tv_split]
    test_ids = np.unique(df_index.monthID)[-tv_split:-1]

    return df_index, train_ids, test_ids


# //TODO Tidy up
def calculate_distances(train_x, train_y, test_x, train_labels, predict_id):
    distances_f = np.zeros([len(train_labels), 5])

    for count, unique_id in enumerate(train_labels):

        distances_f[count, 0] = unique_id
        distances_f[count, 1] = distance_models.dtw(train_x[train_x["monthID"] == unique_id]["Close"],
                                                    test_x[test_x["monthID"] == predict_id]["Close"])
        distances_f[count, 2] = distance_models.twed(train_x[train_x["monthID"] == unique_id]["Close"],
                                                     test_x[test_x["monthID"] == predict_id]["Close"])
        distances_f[count, 3] = distance_models.lcss(np.array(train_x[train_x["monthID"] == unique_id]["Close"], dtype=np.float),
                                                     np.array(test_x[test_x["monthID"] == predict_id]["Close"], dtype=np.float), np.inf, 0.5)
        distances_f[count, 4] = train_y[list(train_y.index) == unique_id]["Returns"]

    output = pd.DataFrame(distances_f)
    output.columns = ["monthID", "distance_dtw", "distance_twed", "distance_lcss", "returns"]

    return output


# //TODO Tidy up
def predict_trades(type_train, train_x, train_y, test_x, train_labels, test_labels, instrument):

    output = pd.DataFrame(columns=list(["monthID", "instrument", "data_normalization", "distance_model", "stat_model", "result"]))

    for count, predict_id in enumerate(tqdm(test_labels)):

        distances = calculate_distances(train_x,
                                        train_y,
                                        test_x,
                                        train_labels,
                                        predict_id
                                        )

        output.loc[len(output)] = [predict_id, instrument, type_train, "dtw", "knn",
                                   stat_models.knn(distances, "distance_dtw")]

        output.loc[len(output)] = [predict_id, instrument, type_train, "twed", "knn",
                                   stat_models.knn(distances, "distance_twed")]

        output.loc[len(output)] = [predict_id, instrument, type_train, "lcss", "knn",
                                   stat_models.knn(distances, "distance_lcss")]

        output.loc[len(output)] = [predict_id, instrument, type_train, "dtw", "kstar",
                                   stat_models.kstar(distances, "distance_dtw")]

        output.loc[len(output)] = [predict_id, instrument, type_train, "twed", "kstar",
                                   stat_models.kstar(distances, "distance_twed")]

        output.loc[len(output)] = [predict_id, instrument, type_train, "lcss", "kstar",
                                   stat_models.kstar(distances, "distance_lcss")]

    return output


# //TODO Tidy up
def main(stock_index):
    print(stock_index)

    months_out_of_sample = 120
    stock_df = "data/stock_dfs/{}.csv".format(stock_index)
    next_month_returns_df = "data/returns/{}.csv".format(stock_index)

    if not os.path.exists('data/predictions'):
        os.makedirs('data/predictions')

    out_filename = "data/predictions/{}_predictions.csv".format(stock_index)

    stocks_f, returns_f = read_data(stock_df, next_month_returns_df)
    train_ids = np.unique(stocks_f.monthID)[:-months_out_of_sample]
    test_ids = np.unique(stocks_f.monthID)[-months_out_of_sample:-1]

    diff_stocks, diff_train_ids, diff_test_ids = difference(stocks_f, tv_split=months_out_of_sample)
    index_stocks, index_train_ids, index_test_ids = indexing(stocks_f, tv_split=months_out_of_sample)

    print("No data normalization")
    trade_predictions_dtw = predict_trades("None",
                                           stocks_f[stocks_f["monthID"].isin(train_ids)],
                                           returns_f[returns_f.index.isin(train_ids)],
                                           stocks_f[stocks_f["monthID"].isin(test_ids)],
                                           train_ids,
                                           test_ids,
                                           stock_index
                                           )

    print("Difference")
    trade_predictions_ddtw = predict_trades("Difference",
                                            diff_stocks[diff_stocks["monthID"].isin(diff_train_ids)],
                                            returns_f[returns_f.index.isin(diff_train_ids)],
                                            diff_stocks[diff_stocks["monthID"].isin(diff_test_ids)],
                                            diff_train_ids,
                                            diff_test_ids,
                                            stock_index
                                            )

    print("Index")
    trade_predictions_idtw = predict_trades("Index",
                                            index_stocks[index_stocks["monthID"].isin(index_train_ids)],
                                            returns_f[returns_f.index.isin(index_train_ids)],
                                            index_stocks[index_stocks["monthID"].isin(index_test_ids)],
                                            index_train_ids,
                                            index_test_ids,
                                            stock_index
                                            )

    data_frames = [trade_predictions_dtw,
                   trade_predictions_ddtw,
                   trade_predictions_idtw]

    df_merged = reduce(lambda left, right: pd.merge(left, right, how='outer'), data_frames)
    pd.DataFrame.to_csv(df_merged, out_filename, sep=',', na_rep='.', index=False)


yahoo_indexes = ["^GSPC", "^DJI", "^GDAXI", "^FCHI", "^N225"]


if __name__ == '__main__':

    # //TODO also add parallel to distance calculations.
    # //TODO fix TQDM with parallel pools.

    Parallel(n_jobs=len(yahoo_indexes))(delayed(main)(stock_index) for stock_index in yahoo_indexes)
