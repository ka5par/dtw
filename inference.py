from sklearn.metrics import accuracy_score, f1_score
from matplotlib import pyplot as plt
plt.style.use(['fivethirtyeight'])

import datetime as dt
import numpy as np
import pandas as pd
import os


# import tabulate #  If need to pretty pring markdown.

# //TODO #6 cmd line arguments,


def convert_orders_to_cum_return(orders, underlying_returns):
    cum_returns = underlying_returns + 1
    cum_returns[0] = 1 + underlying_returns[0]
    for i in range(1, len(cum_returns)):
        cum_returns[i] = cum_returns[i - 1] * (1 + underlying_returns[i] * orders[i])
    return cum_returns


# Create a date column for plotting
def create_date_from_month_id(month_id):
    years = np.array(month_id // 100).astype(int)
    months = np.array(month_id % 100).astype(int)
    return [dt.datetime(years[i], months[i], 1) for i in range(len(years))]


def main(stock_index):

    next_returns = pd.read_csv("data/returns/{}.csv".format(stock_index))
    b_s_orders = pd.read_csv("data/predictions/{}_predictions.csv".format(stock_index))

    # Correct the returns to actual returns
    actual_returns = next_returns.copy()
    actual_returns["Returns"] = next_returns.Returns.shift(1)

    b_s_orders = pd.merge(b_s_orders, actual_returns, how="left", on="monthID")

    # Instead of multi-index using a "label" column.
    b_s_orders["Labels"] = b_s_orders["data_normalization"] + " " + b_s_orders["distance_model"] + " " + b_s_orders["stat_model"]
    mask = b_s_orders["Labels"] == b_s_orders["Labels"][0]
    baseline_cum_returns = convert_orders_to_cum_return(np.ones(len(b_s_orders[mask])), np.array(b_s_orders[mask]["Returns"]))

    b_s_orders["Date"] = create_date_from_month_id(b_s_orders["monthID"])

    # Calculate the cumulative returns of all the outputs.
    b_s_orders["cum_returns"] = 0

    for data_normalization in np.unique(b_s_orders.data_normalization):
        for distance_model in np.unique(b_s_orders.distance_model):
            for stat_model in np.unique(b_s_orders.stat_model):

                mask = (b_s_orders["data_normalization"] == data_normalization) & \
                       (b_s_orders["distance_model"] == distance_model) & \
                       (b_s_orders["stat_model"] == stat_model)

                b_s_orders.loc[mask, "cum_returns"] = convert_orders_to_cum_return(b_s_orders[mask]["result"].values,
                                                                                   b_s_orders[mask]["Returns"].values)

    month_id = np.unique(b_s_orders["monthID"])

    # Plot the cumulative returns
    if not os.path.exists('data/plots'):
        os.makedirs('data/plots')

    # Top 5
    top_list = np.array(b_s_orders[b_s_orders["monthID"] == month_id[-1]].sort_values(by=["cum_returns"], ascending=False).head(5)["Labels"])

    plt.figure(figsize=(12, 12))
    for spec_label in np.unique(b_s_orders["Labels"]):
        mask = b_s_orders["Labels"] == spec_label
        if spec_label in top_list:
            plt.plot(b_s_orders.Date[mask], b_s_orders.cum_returns[mask], label=spec_label, linewidth=2)
        else:
            plt.plot(b_s_orders.Date[mask], b_s_orders.cum_returns[mask], label=spec_label, alpha=0.3, linestyle='--', linewidth=1)

    plt.plot(b_s_orders.Date[mask], baseline_cum_returns, label="Baseline", alpha=0.5, linestyle="-", linewidth=3, color="r")

    plt.legend()
    plt.title("Cumulative returns of {} 2011-2020".format(dict_indexes[stock_index]))
    plt.ylabel("Cumulative returns (%)")
    plt.savefig('data/plots/cumulative_returns_{}.png'.format(stock_index))

    if not os.path.exists('data/summary_tables'):
        os.makedirs('data/summary_tables')

    # Create the performance metrics for the model outputs.
    accuracy_table = pd.DataFrame(columns=["stock_index", "data_normalization", "distance_model", "stat_model", "accuracy", "f1", "profitability"])

    for spec_label in np.unique(b_s_orders["Labels"]):

        mask = b_s_orders["Labels"] == spec_label

        perfect = np.array(b_s_orders[mask]["Returns"]) > 0
        predictions = np.array(b_s_orders[mask]["result"]) > 0
        profitability = np.array(b_s_orders[mask]["cum_returns"])[-1]

        temp_dict = {
            'stock_index': dict_indexes[stock_index],
            'data_normalization': b_s_orders[mask]["data_normalization"].values[-1],
            'distance_model': b_s_orders[mask]["distance_model"].values[-1],
            'stat_model': b_s_orders[mask]["stat_model"].values[-1],
            'accuracy': accuracy_score(perfect, predictions),
            'f1': f1_score(perfect, predictions),
            'profitability': profitability
        }

        accuracy_table = accuracy_table.append(temp_dict, ignore_index=True)
        accuracy_table.to_csv("data/summary_tables/{}".format(stock_index))


yahoo_indexes = ["^GSPC", "^DJI", "^GDAXI", "^FCHI", "^N225"]
dict_indexes = {"^GSPC": "S&P 500", "^DJI": "Dow Jones Industrial Average", "^GDAXI": "DAX30", "^FCHI": "CAC 40", "^N225": "Nikkei 225"}

if __name__ == '__main__':
    for yahoo_index in yahoo_indexes:
        main(yahoo_index)
