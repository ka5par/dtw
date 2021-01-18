import pandas as pd
import ast
from configparser import ConfigParser


def read_data(instrument, predictions=False):
    instrument_file = "data/stock_dfs/{}.csv".format(instrument)
    next_month_returns_file = "data/returns/{}.csv".format(instrument)

    returns = pd.read_csv(next_month_returns_file, index_col=0)

    if predictions:
        prediction = pd.read_csv("data/predictions/{}_predictions.csv".format(instrument))

        return returns, prediction

    else:
        stocks = pd.read_csv(instrument_file, index_col=0)
        stocks["monthID"] = pd.to_datetime(stocks.index.values).year * 100 + pd.to_datetime(stocks.index.values).month

        return stocks, returns


def convert(x):

    if "[" in x:
        try:
            return ast.literal_eval(x)
        except ValueError:
            pass
    if "." in x:
        try:
            return float(x)
        except ValueError:
            pass
    elif any(char.isdigit() for char in x):
        try:
            return int(x)
        except ValueError:
            return x
    else:
        return x


def read_config(option):

    config = ConfigParser()
    config.optionxform = str
    config.read("configuration.ini")
    config = dict(config.items(option))

    for key in config.keys():
        config[key] = convert(config[key])

    return config
