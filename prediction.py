import numpy as np
import pandas as pd
import os
import stat_models
import distance_models
import utils

from tqdm import trange
from functools import reduce
from joblib import Parallel, delayed


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


def calculate_distances(train_x, train_y, test_x, train_labels, predict_id, settings_dict):
    distances_f = np.zeros([len(train_labels), 6])

    for count, unique_id in enumerate(train_labels):

        train = train_x[train_x["monthID"] == unique_id]["Close"]
        test = test_x[test_x["monthID"] == predict_id]["Close"]
        distances_f[count, 0] = unique_id

        if settings_dict["lcss_delta"] == "variable":
            settings_dict["lcss_delta"] = np.max([1, np.round(0.25 * np.mean([len(train), len(test)]), 0)])  # At least 1, 25% of the length of the average length of train&test.

        if settings_dict["lcss_epsilon"] == "variable":
            settings_dict["lcss_epsilon"] = np.min([np.std(train), np.std(test)])  # Smallest standard deviation of train/test

        if "dtw" in settings_dict["list_of_distance_models"]:
            distances_f[count, 1] = distance_models.dtw(train, test, window=15)
        else:
            distances_f[count, 1] = np.nan
        if "twed" in settings_dict["list_of_distance_models"]:
            distances_f[count, 2] = distance_models.twed(train,
                                                         test,
                                                         nu=float(settings_dict["twed_nu"]),
                                                         _lambda=float(settings_dict["twed_lambda"]),)
        else:
            distances_f[count, 2] = np.nan
        if "lcss" in settings_dict["list_of_distance_models"]:
            distances_f[count, 3] = distance_models.lcss(train,
                                                         test,
                                                         delta=int(settings_dict["lcss_delta"]),
                                                         epsilon=float(settings_dict["lcss_epsilon"]))
        else:
            distances_f[count, 3] = np.nan

        if "corrd" in settings_dict["list_of_distance_models"]:
            distances_f[count, 4] = distance_models.corrd(train, test)
        else:
            distances_f[count, 4] = np.nan

        distances_f[count, 5] = train_y[list(train_y.index) == unique_id]["Returns"]

    output = pd.DataFrame(distances_f)
    output.columns = ["monthID", "dtw", "twed", "lcss","corrd", "returns"]

    return output


def multiprocessing_predict_trades(predict_id, train_labels, test_labels, stocks, returns, type_train, settings_dict):

    train_x = stocks[stocks["monthID"].isin(train_labels)].copy()
    train_y = returns[returns.index.isin(train_labels)].copy()
    test_x = stocks[stocks["monthID"].isin(test_labels)].copy()

    distances = calculate_distances(train_x, train_y, test_x, train_labels, predict_id, settings_dict)

    output = pd.DataFrame(columns=list(["monthID", "instrument", "data_normalization", "distance_model", "stat_model", "result"]))

    for a_distance_model in settings_dict["list_of_distance_models"]:
        for a_stat_model in settings_dict["stat_models"]:
            if a_stat_model == "knn":
                output.loc[len(output)] = [predict_id, settings_dict["instrument"], type_train, a_distance_model, a_stat_model,
                                           stat_models.knn(distances, a_distance_model)]
            else:
                output.loc[len(output)] = [predict_id, settings_dict["instrument"], type_train, a_distance_model, a_stat_model,
                                           stat_models.kstar(distances, a_distance_model)]

    return output


def predict_trades(type_train, stocks, returns, settings_dict, months_out_of_sample=60):

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
        delayed(multiprocessing_predict_trades)
        (
            test_labels[i],
            labels[:(-months_out_of_sample + i + 1)],
            labels[-(months_out_of_sample + 1 + i):],
            stocks,
            returns,
            type_train,
            settings_dict
        ) for i in trange(len(test_labels), desc="".join([settings_dict["instrument"], " ", type_train]))))

    return output


# Perfomance metric = `accuracy` / `total_returns`
def take_best_parameters(config_f, performance_metric='profitability'):
    try:
        best_twed = pd.read_csv("data/param_test/{}_{}_test_acc_table.csv".format(config_f["instrument"], "twed")).sort_values(performance_metric, ascending=False)[['nu', 'lambda']].head(1).values[0]  # nu / lambda
        best_lcss = pd.read_csv("data/param_test/{}_{}_test_acc_table.csv".format(config_f["instrument"], "lcss")).sort_values(performance_metric, ascending=False)[['epsilon', 'delta']] .head(1).values[0]  # Epsilon / Delta
        config_f["twed_nu"] = best_twed[0]
        config_f["twed_lambda"] = best_twed[1]
        config_f["lcss_epsilon"] = best_lcss[0]
        config_f["lcss_delta"] = best_lcss[1]
        return config_f
    except FileNotFoundError:
        return config_f


def main(settings_dict):

    stocks, returns = utils.read_data(settings_dict["instrument"])

    trade_predictions = [predict_trades(
        normalization_type,
        stocks,
        returns,
        settings_dict,
        months_out_of_sample= int(settings_dict["months_out_of_sample"])) for normalization_type in settings_dict["normalization_types"]]

    if not os.path.exists('data/predictions'):
        os.makedirs('data/predictions')

    out_filename = "data/predictions/{}_predictions.csv".format(settings_dict["instrument"])

    df_merged = reduce(lambda left, right: pd.merge(left, right, how='outer'), trade_predictions)
    pd.DataFrame.to_csv(df_merged, out_filename, sep=',', na_rep='.', index=False)


instruments = ["^GSPC", "^DJI", "^GDAXI", "^N225", "^FCHI"]  # ["Brent Oil", "Natural Gas", "Gasoline RBOB", "Carbon Emissions", "Gold", "Copper", "London Wheat"]

if __name__ == '__main__':

    config = utils.read_config("run_parameters")

    config["normalization_types"] = ["Index"]

    for stock_index in instruments:
        config["instrument"] = stock_index
        config = take_best_parameters(config)
        main(config)
