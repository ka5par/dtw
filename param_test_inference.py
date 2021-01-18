import numpy as np
import pandas as pd
import inference
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import utils

plt.style.use(['seaborn-poster'])


def plot_scatter(param, value, sm, instrument):
    accuracy_table = pd.read_csv("data/param_test/{}_{}_test_acc_table.csv".format(instrument, param))
    mask = accuracy_table["stat_model"] == sm
    if param == "twed":
        param1 = accuracy_table[mask]["nu"]
        param2 = accuracy_table[mask]["lambda"]
    elif param == "lcss":
        param1 = accuracy_table[mask]["delta"]
        param2 = accuracy_table[mask]["epsilon"]
    else:
        raise ValueError("Need input parameter")

    my_cmap = plt.get_cmap('PiYG')

    plt.figure(figsize=(10, 10))
    plt.scatter(param1, param2, c=accuracy_table[mask][value], cmap=my_cmap, s=100)
    if param == "twed":
        plt.xscale('log')
    plt.xlabel(param1.name)
    plt.ylabel(param2.name)
    plt.colorbar()
    plt.title(sm + " " + param + " " + value + " with different parameters.")
    plt.savefig('data/plots/test_scatter/{}_{}_{}_{}_param.png'.format(instrument, sm, param, value))
    if value == "accuracy":
        plt.figure(figsize=(10, 10))
        plt.scatter(accuracy_table[mask][value], accuracy_table[mask]["profitability"])
        plt.xlabel(value)
        plt.ylabel("profitability")
        plt.title(sm + " " + param)
        plt.savefig('data/plots/test_scatter/{}_{}_{}_prof_acc.png'.format(instrument, sm, param))


def main(param, instrument, write_to_file=True):

    next_returns, _ = utils.read_data(instrument, predictions=True)
    b_s_orders = pd.read_csv("data/param_test/{}_{}_test.csv".format(instrument, param))

    b_s_orders = inference.calculate_actual_returns(b_s_orders, next_returns)

    b_s_orders["Labels"] = b_s_orders["stat_model"] + " " + \
                           b_s_orders["nu"].astype(str) + " " + \
                           b_s_orders["lambda"].astype(str) + " " + \
                           b_s_orders["epsilon"].astype(str) + " " + \
                           b_s_orders["delta"].astype(str)

    b_s_orders["cum_returns"] = 0
    accuracy_table = pd.DataFrame(
        columns=["stock_index", "stat_model", "nu", "lambda", "epsilon", "delta", "accuracy", "profitability"])

    for a_label in np.unique(b_s_orders["Labels"]):
        mask = b_s_orders["Labels"] == a_label
        b_s_orders.loc[mask, "cum_returns"] = inference.convert_orders_to_cum_return(b_s_orders[mask]["result"].values,
                                                                                     b_s_orders[mask]["Returns"].values)

        perfect = np.array(b_s_orders[mask]["Returns"]) > 0
        predictions = np.array(b_s_orders[mask]["result"]) > 0
        profitability = np.array(b_s_orders[mask]["cum_returns"])[-1]

        temp_dict = {
            'stock_index': dict_indexes[instrument],
            'stat_model': b_s_orders[mask]["stat_model"].values[-1],
            'nu': b_s_orders[mask]["nu"].values[-1],
            'lambda': b_s_orders[mask]["lambda"].values[-1],
            'epsilon': b_s_orders[mask]["epsilon"].values[-1],
            'delta': b_s_orders[mask]["delta"].values[-1],
            'accuracy': accuracy_score(perfect, predictions),
            'profitability': profitability
        }
        accuracy_table = accuracy_table.append(temp_dict, ignore_index=True)

    if write_to_file:
        accuracy_table.to_csv("data/param_test/{}_{}_test_acc_table.csv".format(instrument, param))
        return
    else:
        return accuracy_table.sort("profitability", ascending=False).loc[0], param


distance_metrics = ["twed", "lcss"]  # ["twed", "lcss"]

dict_indexes = utils.read_config("actual_names")

instruments = ["^GSPC", "^DJI", "^GDAXI", "^N225", "^FCHI"]  # ["Brent Oil", "Natural Gas", "Gasoline RBOB", "Carbon Emissions", "Gold", "Copper", "London Wheat"]


if __name__ == '__main__':

    for comm in instruments:
        for distance_metric in distance_metrics:
            main(distance_metric, comm)
            # plot_scatter(distance_metric, "accuracy", "knn", comm)
            # plot_scatter(distance_metric, "profitability", "knn", comm)
