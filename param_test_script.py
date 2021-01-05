import numpy as np
import pandas as pd
import os
import prediction

normalization_types_c = ["Index"]  # ["None", "Difference", "Index"]
list_of_distance_models_c = ["lcss"]
instrument_file = "data/stock_dfs/{}.csv".format("^GSPC")
next_month_returns_file = "data/returns/{}.csv".format("^GSPC")

stocks, returns = prediction.read_data(instrument_file, next_month_returns_file)

nu_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
_lambda_values = [0, 0.25, 0.5, 0.75, 1]

epsilon_values = ["variable", 0.1, 0.3, 0.5, 0.6, 0.8, 1, 1.5, 2, 2.5, 3]
delta_values = ["variable", 4, 5, 6, 7, 8, 9, np.inf]  # Related works claim more than 20-30% of time-series length has no impact.


if not os.path.exists('data/plot/test_scatter'):
    os.makedirs('data/plot/test_scatter')

config = dict(
    twed_nu=1,
    twed_lambda=0.001,
    lcss_epsilon=0.5,
    lcss_delta=np.inf,
    list_of_distance_models=["twed"],
    normalization_types=["Index"],
    months_out_of_sample=120,
    instrument="^GSPC")

# TWED
# Output is a scatter plot & save output as csv (ACCURACY, PROFITABILITY)

main_df = pd.DataFrame(columns=["monthID", "instrument", "data_normalization", "distance_model", "stat_model", "result", "nu", "lambda", "epsilon", "delta"])

for nu in nu_values:
    config["twed_nu"] = nu
    for _lambda in _lambda_values:
        config["twed_lambda"] = _lambda

        b_s_orders = prediction.predict_trades(
            config["normalization_types"][0],
            stocks,
            returns,
            config,
            months_out_of_sample=config["months_out_of_sample"])

        b_s_orders["nu"] = nu
        b_s_orders["lambda"] = _lambda
        b_s_orders[["epsilon", "delta"]] = np.nan

        main_df = main_df.append(b_s_orders)

main_df.to_csv("data/twed_test.csv")

# LCSS

config["list_of_distance_models"] = ["lcss"]

main_df = pd.DataFrame(columns=["monthID", "instrument", "data_normalization", "distance_model", "stat_model", "result", "nu", "lambda", "epsilon", "delta"])

for epsilon in epsilon_values:
    config["lcss_epsilon"] = epsilon
    for delta in delta_values:
        config["lcss_delta"] = delta

        b_s_orders = prediction.predict_trades(
            config["normalization_types"][0],
            stocks,
            returns,
            config,
            months_out_of_sample=config["months_out_of_sample"])

        b_s_orders[["nu", "lambda"]] = np.nan
        b_s_orders["epsilon"] = epsilon
        b_s_orders["delta"] = delta

        main_df = main_df.append(b_s_orders)

main_df.to_csv("data/lcss_test.csv")
