from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tabulate


def convert_orders_to_cum_return(orders, underlying_returns):
    cum_returns = underlying_returns + 1
    cum_returns[0] = 1 + underlying_returns[0]

    for i in range(1, len(cum_returns)):
        cum_returns[i] = cum_returns[i - 1] * (1 + underlying_returns[i] * orders[i])

    return cum_returns


def plot_lines(df, plot_title, plot_filter=None):
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


# Read in the data
next_returns = pd.read_csv("data/next_month_returns.csv")
b_s_orders = pd.read_csv("data/trade_preds.csv")

# Correct the returns to actual returns
actual_returns = next_returns.copy()
actual_returns.sp500 = next_returns.sp500.shift(1)  # <-- get rid of sp500

b_s_orders = pd.merge(b_s_orders, actual_returns, how="left", on="monthID")

baseline = convert_orders_to_cum_return(np.ones(len(b_s_orders)), np.array(b_s_orders["sp500"]))  # <-- get rid of sp500
perfect = convert_orders_to_cum_return(np.array(b_s_orders["sp500"] > 0), np.array(b_s_orders["sp500"]))  # <-- get rid of sp500
columns = np.array(b_s_orders.columns[b_s_orders.columns.str.contains("DTW")])

trading_cum_returns = pd.DataFrame(columns=columns)


for column in columns:
    trading_cum_returns[column] = convert_orders_to_cum_return(np.array(b_s_orders[column]),
                                                               np.array(b_s_orders["sp500"]))

# trading_cum_returns["baseline"] = baseline
# trading_cum_returns["perfect"] = perfect

trading_cum_returns.index = pd.to_datetime(b_s_orders["Date"])


# Filters
columns_dtw = trading_cum_returns.columns[np.array(trading_cum_returns.columns.str.contains("DTWdtw|baseline", regex=True))]
columns_twed = trading_cum_returns.columns[np.array(trading_cum_returns.columns.str.contains("DTWtwed|baseline", regex=True))]
columns_over250 = trading_cum_returns.columns[(trading_cum_returns.iloc[-1] > 2.5)]


plot_lines(trading_cum_returns, "S&P 500 Cumulative returns of ALL methods")
plot_lines(trading_cum_returns, "S&P 500 Cumulative returns of DTW", columns_dtw)
plot_lines(trading_cum_returns, "S&P 500 Cumulative returns of TWED", columns_twed)
plot_lines(trading_cum_returns, "S&P 500 Cumulative returns that end with over 250%", columns_over250)

accuracy = b_s_orders.copy()
accuracy.index = accuracy.monthID
perfect= accuracy["sp500"] > 0
accuracy["baseline"] = 1
date = accuracy["Date"]
accuracy.drop(["monthID", "Date", "sp500"], axis=1, inplace=True)  # <-- get rid of sp500

test = pd.DataFrame(np.sum(accuracy.isin(perfect))/len(accuracy))
test.columns = ["accuracy"]
test["distance_metric"] = ["dtw", "twed"]*6 + ["baseline"]
test["stat_model"] = ["knn", "knn", "kstar", "kstar"]*3 + ["baseline"]
test["normalization_method"] = ["none"]*4 + ["diff"]*4 + ["indexing"]*4 + ["baseline"]
print(test.pivot_table(index=["distance_metric", "stat_model", "normalization_method"], aggfunc="mean").to_markdown())
