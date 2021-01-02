import numpy as np
import pandas as pd
import os
import stat_models
import distance_models

from tqdm import trange
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

    ids = last_month_closes["monthID"]
    last_month_closes = last_month_closes["Close"].shift(1)

    for i in range(len(ids)):
        mask = df_index["monthID"] == ids[i]
        df_index.loc[mask, "Close"] = df_index.loc[mask, "Close"] / last_month_closes[i]

    df_index = df_index.dropna()

    train_ids = np.unique(df_index.monthID)[:-tv_split]
    test_ids = np.unique(df_index.monthID)[-tv_split:-1]

    return df_index, train_ids, test_ids


# https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
def normalizing(df_stocks, tv_split):
    df_norm = df_stocks.copy()

    ids = np.unique(df_stocks["monthID"])

    for i in range(len(ids)):
        mask = df_norm["monthID"] == ids[i]
        x = df_norm.loc[mask, "Close"]
        df_norm.loc[mask, "Close"] = 2 * ((x - np.min(x)) / (np.max(x)-np.min(x))) - 1

    df_norm = df_norm.dropna()

    train_ids = np.unique(df_norm.monthID)[:-tv_split]
    test_ids = np.unique(df_norm.monthID)[-tv_split:-1]

    return df_norm, train_ids, test_ids


# //TODO Tidy up
def calculate_distances(train_x, train_y, test_x, train_labels, predict_id):
    distances_f = np.zeros([len(train_labels), 5])

    for count, unique_id in enumerate(train_labels):
        train = train_x[train_x["monthID"] == unique_id]["Close"]
        test = test_x[test_x["monthID"] == predict_id]["Close"]
        distances_f[count, 0] = unique_id
        distances_f[count, 1] = distance_models.dtw(train, test)
        distances_f[count, 2] = distance_models.twed(train, test, nu=1, _lambda=0.001,)
        distances_f[count, 3] = distance_models.lcss(np.array(train, dtype=np.float), np.array(test, dtype=np.float), np.inf, 0.5)
        distances_f[count, 4] = train_y[list(train_y.index) == unique_id]["Returns"]

    output = pd.DataFrame(distances_f)
    output.columns = ["monthID", "dtw", "twed", "lcss", "returns"]

    return output


def multiproc_predict_trades(predict_id, train_labels, test_labels, stocks, returns, type_train, instrument, distance_models_to_run):

    train_x = stocks[stocks["monthID"].isin(train_labels)]
    train_y = returns[returns.index.isin(train_labels)]
    test_x = stocks[stocks["monthID"].isin(test_labels)]

    distances = calculate_distances(train_x, train_y, test_x, train_labels, predict_id)

    output = pd.DataFrame(columns=list(["monthID", "instrument", "data_normalization", "distance_model", "stat_model", "result"]))
    string_distance_models = ["dtw", "twed", "lcss"]
    string_stat_models = ["knn", "kstar"]

    for i in range(len(string_distance_models)):
        for j in range(len(string_stat_models)):
            if j == 0:
                output.loc[len(output)] = [predict_id, instrument, type_train, string_distance_models[i], string_stat_models[j],
                                           stat_models.knn(distances, string_distance_models[i])]
            else:
                output.loc[len(output)] = [predict_id, instrument, type_train, string_distance_models[i], string_stat_models[j],
                                           stat_models.kstar(distances, string_distance_models[i])]

    return output


def predict_trades(type_train, stocks, returns, instrument, distance_models_to_run, months_out_of_sample=60):

    if type_train == "None":
        train_labels = np.unique(stocks["monthID"])[:-months_out_of_sample]
        test_labels = np.unique(stocks["monthID"])[-months_out_of_sample:-1]
    elif type_train == "Difference":
        stocks, train_labels, test_labels = difference(stocks, tv_split=months_out_of_sample)
    elif type_train == "Index":
        stocks, train_labels, test_labels = indexing(stocks, tv_split=months_out_of_sample)
    elif type_train == "Normalization":
        stocks, train_labels, test_labels = normalizing(stocks, tv_split=months_out_of_sample)
    else:
        raise TypeError("No/Wrong type_train chosen.")

    labels = np.append(train_labels, test_labels)

    output = pd.concat(Parallel(n_jobs=-1)(
        delayed(multiproc_predict_trades)
        (
            test_labels[i],
            labels[:(-months_out_of_sample + i + 1)],
            labels[-(months_out_of_sample + 1 + i):],
            stocks,
            returns,
            type_train,
            instrument,
            distance_models_to_run
        ) for i in trange(len(test_labels), desc="".join([instrument, " ", type_train]))))

    return output


def main(instrument, distance_models_to_run):

    months_out_of_sample = 120
    instrument_file = "data/stock_dfs/{}.csv".format(instrument)
    next_month_returns_file = "data/returns/{}.csv".format(instrument)

    stocks_f, returns_f = read_data(instrument_file, next_month_returns_file)

    trade_predictions = [predict_trades(normalization_type, stocks_f, returns_f, instrument, distance_models_to_run, months_out_of_sample=months_out_of_sample) for normalization_type in normalization_types]

    if not os.path.exists('data/predictions'):
        os.makedirs('data/predictions')

    out_filename = "data/predictions/{}_predictions.csv".format(instrument)

    df_merged = reduce(lambda left, right: pd.merge(left, right, how='outer'), trade_predictions)
    pd.DataFrame.to_csv(df_merged, out_filename, sep=',', na_rep='.', index=False)


yahoo_indexes = ["^GSPC", "^DJI", "^GDAXI", "^FCHI", "^N225"]
normalization_types = ["Index", "Normalization"]  # ["None", "Difference", "Index"]
list_of_distance_models = ["DTW", "TWED", "LCSS"]

if __name__ == '__main__':

    for stock_index in yahoo_indexes:
        main(stock_index, list_of_distance_models)
