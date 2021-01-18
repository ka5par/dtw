import numpy as np
import pandas as pd
import os
import prediction
import utils
import param_test_inference

def run_parameter_test(stocks, returns, config, test_parameters, list_tests=None, write_to_file=True):

    if list_tests is None or ("twed" not in list_tests and "lcss" not in list_tests):
        print("Defaulting to twed & lcss tests.")
        list_tests = ["twed", "lcss"]

    if "twed" in list_tests:

        config["list_of_distance_models"] = ["twed"]

        twed_tests = pd.DataFrame(
            columns=["monthID", "instrument", "data_normalization", "distance_model", "stat_model", "result", "nu",
                     "lambda", "epsilon", "delta"])

        for nu in test_parameters["nu_values"]:
            config["twed_nu"] = nu
            for _lambda in test_parameters["lambda_values"]:
                config["twed_lambda"] = _lambda

                b_s_orders = prediction.predict_trades(
                    config["normalization_types"][0],
                    stocks,
                    returns,
                    config,
                    months_out_of_sample=test_parameters["months_out_of_sample"])

                b_s_orders["nu"] = nu
                b_s_orders["lambda"] = _lambda
                b_s_orders[["epsilon", "delta"]] = np.nan

                twed_tests = twed_tests.append(b_s_orders)

        if write_to_file:
            twed_tests.to_csv("data/param_test/{}_twed_test.csv".format(config["instrument"]))

    if "lcss" in list_tests:

        config["list_of_distance_models"] = ["lcss"]

        lcss_tests = pd.DataFrame(
            columns=["monthID", "instrument", "data_normalization", "distance_model", "stat_model", "result", "nu",
                     "lambda", "epsilon", "delta"])

        for epsilon in test_parameters["epsilon_values"]:
            config["lcss_epsilon"] = epsilon
            for delta in test_parameters["delta_values"]:
                config["lcss_delta"] = delta

                b_s_orders = prediction.predict_trades(
                    config["normalization_types"][0],
                    stocks,
                    returns,
                    config,
                    months_out_of_sample=test_parameters["months_out_of_sample"])

                b_s_orders[["nu", "lambda"]] = np.nan
                b_s_orders["epsilon"] = epsilon
                b_s_orders["delta"] = delta

                lcss_tests = lcss_tests.append(b_s_orders)

        if write_to_file:
            lcss_tests.to_csv("data/param_test/{}_lcss_test.csv".format(config["instrument"]))


def main(instrument, config, parameters_to_test):

    stocks, returns = utils.read_data(instrument)
    returns = returns[:-120]
    whats_left = np.array(returns.index)

    stocks = stocks[np.array(stocks.monthID.apply(lambda x: x in whats_left))]

    if not os.path.exists('data/plots/test_scatter'):
        os.makedirs('data/plots/test_scatter')

    if not os.path.exists('data/param_test/'):
        os.makedirs('data/param_test/')

    run_parameter_test(stocks, returns, config, parameters_to_test)


if __name__ == '__main__':
    parameters_to_test = utils.read_config("test_parameters")

    instruments = ["^GSPC", "^DJI", "^GDAXI", "^N225", "^FCHI"]

    sample_config = utils.read_config("run_parameters")
    sample_config["normalization_types"] = ["Index"]

    for instrument in instruments:
        sample_config["instrument"] = instrument
        main(instrument, sample_config, parameters_to_test)

    distance_metrics = ["twed", "lcss"]

    for comm in instruments:
        for distance_metric in distance_metrics:
            param_test_inference.main(distance_metric, comm)
