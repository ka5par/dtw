from sklearn.metrics import accuracy_score, f1_score
from matplotlib import pyplot as plt
plt.style.use(['fivethirtyeight'])

import datetime as dt
import numpy as np
import pandas as pd
import os


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


def calculate_actual_returns(orders, returns):
    actual_returns = returns.copy()
    actual_returns["Returns"] = returns.Returns.shift(1)
    return pd.merge(orders, actual_returns, how="left", on="monthID")


def main(stock_index):

    next_returns = pd.read_csv("data/returns/{}.csv".format(stock_index))
    b_s_orders = pd.read_csv("data/predictions/{}_predictions.csv".format(stock_index))

    b_s_orders = calculate_actual_returns(b_s_orders, next_returns)
    b_s_orders["Date"] = create_date_from_month_id(b_s_orders["monthID"])
    month_id = np.unique(b_s_orders["monthID"])

    # Instead of multi-index using a "label" column.
    b_s_orders["Labels"] = b_s_orders["data_normalization"] + " " + b_s_orders["distance_model"] + " " + b_s_orders["stat_model"]
    mask = b_s_orders["Labels"] == b_s_orders["Labels"][0]
    baseline_cum_returns = convert_orders_to_cum_return(np.ones(len(b_s_orders[mask])), np.array(b_s_orders[mask]["Returns"]))

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

    # Plot the cumulative returns
    if not os.path.exists('data/plots'):
        os.makedirs('data/plots')

    # Top 5
    top_list = np.array(b_s_orders[b_s_orders["monthID"] == month_id[-1]].sort_values(by=["cum_returns"], ascending=False).head(5)["Labels"])
    plt.figure(figsize=(12, 12))
    distance_models = np.unique(b_s_orders["distance_model"])
    for i, spec_label in enumerate(np.unique(b_s_orders["Labels"])):
        mask = b_s_orders["Labels"] == spec_label
        if spec_label in top_list:
            plt.plot(b_s_orders.Date[mask], b_s_orders.cum_returns[mask], label=distance_models[i], linewidth=1)
            plt.annotate(distance_models[i], xy=(plt.xticks()[0][-1] + 0.7, b_s_orders.cum_returns[mask].iloc[-1]))
        else:
            plt.plot(b_s_orders.Date[mask], b_s_orders.cum_returns[mask], label=spec_label, alpha=0.3, linestyle='--', linewidth=1)

    plt.plot(b_s_orders.Date[mask], baseline_cum_returns, label="index", alpha=1, linestyle="-", linewidth=3, color="r")

    plt.legend()
    plt.title("Total returns {}".format(dict_indexes[stock_index]))
    plt.ylabel("Total returns (%)")
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

        standard_deviation_of_excess_return = np.std(np.array(b_s_orders[mask]["cum_returns"]) - baseline_cum_returns)
        print(standard_deviation_of_excess_return)
        temp_dict = {
            'stock_index': dict_indexes[stock_index],
            'data_normalization': b_s_orders[mask]["data_normalization"].values[-1],
            'distance_model': b_s_orders[mask]["distance_model"].values[-1],
            'stat_model': b_s_orders[mask]["stat_model"].values[-1],
            'accuracy': accuracy_score(perfect, predictions),
            'std_excess': standard_deviation_of_excess_return,
            'sharpe_ratio': profitability/standard_deviation_of_excess_return,
            'profitability': profitability
        }

        accuracy_table = accuracy_table.append(temp_dict, ignore_index=True)

    accuracy_table.to_csv("data/summary_tables/{}".format(stock_index))


yahoo_indexes = ["^GSPC", "^DJI", "^GDAXI", "^FCHI", "^N225"]
# commodity_indexes = ["Brent Oil", "Natural Gas", "Gasoline RBOB", "Carbon Emissions", "Gold", "Copper", "London Wheat"]
dict_indexes = {"^GSPC": "S&P 500",
                "^DJI": "Dow Jones Industrial Average",
                "^GDAXI": "DAX30",
                "^FCHI": "CAC 40",
                "^N225": "Nikkei 225",
                "Brent Oil": "Brent",
                "Natural Gas": "NG",
                "Gasoline RBOB": "RBOB",
                "Carbon Emissions": "CO2",
                "Gold": "Gold",
                "Copper": "Copper",
                "London Wheat": "Wheat"
                }

if __name__ == '__main__':
    for yahoo_index in yahoo_indexes:
        main(yahoo_index)
    # for commodity in commodity_indexes:
    #     main(commodity)

