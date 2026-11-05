# Step 1, get the data and save it in the data folder.
# Bundled uses the CSV already in this repo and needs no internet.
# Yahoo downloads fresh data up to 2025.

import argparse
import os
import numpy as np
import pandas as pd
import config


def load_bundled():
    # Read the S and P 500 and the 20 large company prices saved in the repo.
    path = os.path.join(config.DATA_DIR, "sp500_bundled.csv")
    return pd.read_csv(path, index_col=0, parse_dates=True)


def download_yahoo():
    # Download closing prices and keep only days present in every series.
    import yfinance as yf
    tickers = [config.TARGET_TICKER] + config.RELATED_TICKERS
    raw = yf.download(tickers, start=config.SOURCES["yahoo"]["start"],
                      end=config.SOURCES["yahoo"]["end"], auto_adjust=False, progress=False)
    table = raw["Close"].dropna()
    if table.empty:
        raise SystemExit("Yahoo gave no data. Try again later or use the bundled source.")
    table = table[tickers]
    table = table.rename(columns={config.TARGET_TICKER: "SP500"})
    table.to_csv(os.path.join(config.DATA_DIR, "sp500_yahoo.csv"))
    return table


def load_prices(source):
    # Return prices for the chosen source, cut to the project dates.
    if source == "yahoo":
        path = os.path.join(config.DATA_DIR, "sp500_yahoo.csv")
        table = pd.read_csv(path, index_col=0, parse_dates=True) if os.path.exists(path) else download_yahoo()
    else:
        table = load_bundled()
    dates = config.SOURCES[source]
    return table.loc[dates["start"]:dates["end"]].dropna()


def load_log_prices(source):
    # Log prices make a 10 percent move look the same in any year.
    table = load_prices(source)
    target = np.log(table["SP500"].to_numpy())
    related = np.log(table.drop(columns="SP500").to_numpy())
    return table.index, target, related


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="bundled", choices=["bundled", "yahoo"])
    source = parser.parse_args().source
    prices = load_prices(source)
    print(f"Days {len(prices)}, from {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Related series {prices.shape[1] - 1}")
