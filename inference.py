from matplotlib import pyplot as plt

import datetime as dt
import numpy as np
import pandas as pd


# import tabulate #  If need to pretty pring markdown.

# //TODO #6 cmd line arguments,
# //TODO create accuracy csvs.


def convert_orders_to_cum_return(orders, underlying_returns):
    cum_returns = underlying_returns + 1
    cum_returns[0] = 1 + underlying_returns[0]
    for i in range(1, len(cum_returns)):
        cum_returns[i] = cum_returns[i - 1] * (1 + underlying_returns[i] * orders[i])
    return cum_returns


def plot_lines(df, plot_title, baseline, plot_filter=None):
    plt.figure(figsize=(12, 12))
    if plot_filter is None:
        plt.plot(df)
        plt.plot(df.index, baseline, linewidth=4.0, linestyle="--", color="r")
        plt.legend(list(df.columns) + ["baseline"])
    else:
        plt.plot(df[plot_filter])
        plt.plot(df.index, baseline, linewidth=4.0, linestyle="--", color="r")
        plt.legend(list(plot_filter) + ["baseline"])
    plt.title(plot_title)
    plt.show()


def main(stock_index):
    next_returns = pd.read_csv("data/returns/{}.csv".format(stock_index))
    b_s_orders = pd.read_csv("data/predictions/{}_predictions.csv".format(stock_index))

    # Correct the returns to actual returns
    actual_returns = next_returns.copy()
    actual_returns["Returns"] = next_returns.Returns.shift(1)

    b_s_orders = pd.merge(b_s_orders, actual_returns, how="left", on="monthID")

    years = np.array(b_s_orders.monthID // 100).astype(int)
    months = np.array(b_s_orders.monthID % 100).astype(int)
    b_s_orders["Date"] = [dt.datetime(years[i], months[i], 1) for i in range(len(years))]
    b_s_orders["Labels"] = b_s_orders["data_normalization"] + " " + b_s_orders["distance_model"] + " " + b_s_orders["stat_model"]

    mask = b_s_orders["Labels"] == b_s_orders["Labels"][0]
    baseline_cum_returns = convert_orders_to_cum_return(np.ones(len(b_s_orders[mask])), np.array(b_s_orders[mask]["Returns"]))

    # perfect_cum_returns = convert_orders_to_cum_return(np.array(b_s_orders["Returns"] > 0),
    #                                                    np.array(b_s_orders["Returns"]))

    b_s_orders["cum_returns"] = 0

    for data_normalization in np.unique(b_s_orders.data_normalization):
        for distance_model in np.unique(b_s_orders.distance_model):
            for stat_model in np.unique(b_s_orders.stat_model):
                mask = (b_s_orders["data_normalization"] == data_normalization) & \
                       (b_s_orders["distance_model"] == distance_model) & \
                       (b_s_orders["stat_model"] == stat_model)

                b_s_orders.loc[mask, "cum_returns"] = convert_orders_to_cum_return(b_s_orders[mask]["result"].values,
                                                                                   b_s_orders[mask]["Returns"].values)

    plt.figure(figsize=(12, 12))
    for spec_label in np.unique(b_s_orders["Labels"]):
        mask = b_s_orders["Labels"] == spec_label
        plt.plot(b_s_orders.Date[mask], b_s_orders.cum_returns[mask], label=spec_label)

    plt.plot(b_s_orders.Date[mask], baseline_cum_returns, label="baseline",
             linewidth=4.0, linestyle="--", color="r")

    plt.legend()
    plt.title("Cumulative returns of {} 2011-2020".format(stock_index))
    plt.ylabel("Cumulative returns (%)")
    plt.savefig('data/cumulative_returns_{}.png'.format(stock_index))

    # accuracy = b_s_orders.copy()
    # accuracy.index = accuracy.monthID
    # perfect= accuracy["sp500"] > 0
    # accuracy["baseline"] = 1
    # date = accuracy["Date"]
    # accuracy.drop(["monthID", "Date", "sp500"], axis=1, inplace=True)  # <-- get rid of sp500
    #
    # test = pd.DataFrame(np.sum(accuracy.isin(perfect))/len(accuracy))
    # test.columns = ["accuracy"]
    # test["distance_metric"] = ["dtw", "twed"]*6 + ["baseline"]
    # test["stat_model"] = ["knn", "knn", "kstar", "kstar"]*3 + ["baseline"]
    # test["normalization_method"] = ["none"]*4 + ["diff"]*4 + ["indexing"]*4 + ["baseline"]
    # print(test.pivot_table(index=["distance_metric", "stat_model", "normalization_method"], aggfunc="mean").to_markdown())


if __name__ == '__main__':
    yahoo_indexes = ["^GSPC", "^DJI", "^GDAXI", "^FCHI", "^N225"]

    for yahoo_index in yahoo_indexes:
        main(yahoo_index)
