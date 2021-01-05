
# Download data from Yahoo/Investing.
# Put tickers into data/stocks_cfs
# Calculate returns into data/returns

# Links:
# https://stackoverflow.com/questions/54854276/no-data-fetched-web-datareader-panda
# https://medium.com/wealthy-bytes/5-lines-of-python-to-automate-getting-the-s-p-500-95a632e5e567
# https://pythonprogramming.net/combining-stock-prices-into-one-dataframe-python-programming-for-finance/
# https://pythonprogramming.net/sp500-company-price-data-python-programming-for-finance/


import pandas as pd
import pandas_datareader.data as web
import investpy

import datetime as dt
import os

from tqdm import trange


def get_data_from_yahoo(tickers, years=15, force_overwrite=True):

    if not os.path.exists('data/stock_dfs'):
        os.makedirs('data/stock_dfs')

    start = dt.datetime.now() - dt.timedelta(days=years*365)
    end = dt.datetime.now()
    print("Downloading from yahoo.")

    for i in trange(len(tickers)):
        ticker = tickers[i]
        if not os.path.exists('data/stock_dfs/{}.csv'.format(ticker)) or force_overwrite:
            df = web.get_data_yahoo(str(ticker), start, end)
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df.drop(['High', 'Low', 'Open', 'Close', 'Volume'], axis=1, inplace=True)
            df.rename({"Adj Close": "Close"}, axis=1, inplace=True)
            df.to_csv('data/stock_dfs/{}.csv'.format(ticker))


def get_data_from_investing(tickers, countries, years=15, t_type="index"):

    if t_type != "index" and t_type != "commodity":
        print("Currently, only work with indexes from investing.com.")
        return

    if not os.path.exists('data/stock_dfs'):
        os.makedirs('data/stock_dfs')

    start = dt.datetime.now() - dt.timedelta(days=years * 365)
    end = dt.datetime.now()
    print("Downloading from investing.com")

    if t_type == "index":
        for i in trange(len(tickers)):
            ticker = tickers[i]
            if not os.path.exists('data/stock_dfs/{}.csv'.format(ticker)):
                df = investpy.get_index_historical_data(index=ticker,
                                                        from_date=str(start.strftime("%d/%m/%Y")),
                                                        to_date=str(end.strftime("%d/%m/%Y")),
                                                        country=countries[i])
                df.reset_index(inplace=True)
                df.set_index("Date", inplace=True)
                df.drop(['High', 'Low', 'Open', 'Volume', 'Currency'], axis=1, inplace=True)
                df.to_csv('data/stock_dfs/{}.csv'.format(ticker))

    elif t_type == "commodity":

        for i in trange(len(tickers)):

            ticker = tickers[i]
            df = investpy.get_commodity_historical_data(commodity=ticker,
                                                            from_date=str(start.strftime("%d/%m/%Y")),
                                                            to_date=str(end.strftime("%d/%m/%Y")),
                                                            country=countries[i])
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df.drop(['High', 'Low', 'Open', 'Volume', 'Currency'], axis=1, inplace=True)
            df.to_csv('data/stock_dfs/{}.csv'.format(ticker))


def calculate_returns(ticker):

    file = 'data/stock_dfs/{}.csv'.format(ticker)

    if not os.path.exists('data/returns'):
        os.makedirs('data/returns')

    print("Calculating returns for {}.".format(ticker))

    df = pd.read_csv(file, index_col=0)
    df.index = pd.to_datetime(df.index)

    returns = df.loc[df.groupby(df.index.to_period('M')).apply(lambda x: x.index.max())]  # https://stackoverflow.com/questions/48288059/how-to-get-last-day-of-each-month-in-pandas-dataframe-index-using-timegrouper
    returns = returns.pct_change()
    returns = returns.shift(-1)  # Shift them up by 1 because, they are to be used as predictions per id.
    returns = returns[:-1]  # Drop last row. It's NaN.
    returns["monthID"] = pd.to_datetime(returns.index.values).year * 100 + pd.to_datetime(returns.index.values).month
    returns = returns.rename(columns={"Close": "Returns"})

    returns.set_index("monthID", inplace=True)
    returns.to_csv('data/returns/{}.csv'.format(ticker))


tickers_yahoo = ["^GSPC", "^DJI", "^GDAXI", "^FCHI", "^N225", "^VIX"]

tickers_investing = ["Brent Oil", "Natural Gas", "Gasoline RBOB", "Carbon Emissions", "Gold", "Copper", "London Wheat"]
countries_investing = ["united kingdom", "united states", "united states", "united kingdom", "united states", "united states", "united kingdom"]


if __name__ == '__main__':
    get_data_from_investing(tickers_investing, countries_investing, years=40, t_type="commodity")
    get_data_from_yahoo(tickers_yahoo, years=40)

    for ticker_l in tickers_yahoo:
        calculate_returns(ticker_l)

    for ticker_l in tickers_investing:
        calculate_returns(ticker_l)

    # print(investpy.commodities.get_commodity_groups())
    # print(investpy.get_commodities_overview("energy"))
    # print(investpy.get_commodities_overview("metals"))
    # print(investpy.get_commodities_overview("grains"))
    # print(investpy.get_commodities_overview("softs"))
