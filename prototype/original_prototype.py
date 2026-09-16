"""ORIGINAL PROTOTYPE -- preserved verbatim for provenance / regression reference.

This is the source-of-truth script for the modeling logic. Reformatted only for
line breaks (the version supplied had newlines stripped); no logic altered.
"""
import yfinance as yf
import numpy as np


def analyze_earnings_vol(ticker_symbol):
    # Fetch 5 years of price history
    tkr = yf.Ticker(ticker_symbol)
    hist = tkr.history(period="5y")

    # Converts dataframe to raw numpy arrays
    dates = hist.index.values.astype('datetime64[D]')
    prices = hist['Close'].values

    # Calculates daily log price changes
    log_returns = np.diff(np.log(prices))
    dates = dates[1:]

    # Extracts historical earnings announcement dates
    try:
        edates = tkr.earnings_dates.index.values.astype('datetime64[D]')
    except AttributeError:
        print(f"No earnings data available for {ticker_symbol}.")
        return

    v5_list, v10_list, earn_vol_list = [], [], []
    for edate in edates:
        # Match earnings to next trading day
        future_idx = np.where(dates >= edate)[0]
        if future_idx.size == 0:
            continue
        idx = future_idx[0]

        # Ensure enough data
        if idx < 20 or idx + 10 >= len(log_returns):
            continue

        # Realized vol for 5/10 days post-event
        vol5 = np.std(log_returns[idx+1 : idx+6])
        vol10 = np.std(log_returns[idx+1 : idx+11])
        base_vol = np.std(log_returns[idx-20 : idx])

        # Captures two-day total event movement
        tot_vol = np.std(log_returns[idx : idx+2])

        # Isolates earnings move via variance
        var_diff = (tot_vol**2) - (base_vol**2)
        implied_vol = np.sqrt(var_diff) if var_diff > 0 else 0.0

        v5_list.append(vol5)
        v10_list.append(vol10)
        earn_vol_list.append(implied_vol)

    # Stores results in NumPy arrays
    vol_5d = np.array(v5_list)
    vol_10d = np.array(v10_list)
    earnings_vol = np.array(earn_vol_list)

    # Check for empty data
    if len(vol_5d) == 0:
        print("Not enough valid data points to process.")
        return

    # Displays calculated average vol metrics
    print(f"--- Results for {ticker_symbol} ---")
    print(f"Mean 5d Vol:       {np.mean(vol_5d):.6f}")
    print(f"Mean 10d Vol:      {np.mean(vol_10d):.6f}")
    print(f"Mean Earnings Vol: {np.mean(earnings_vol):.6f}\n")

    # Prints full result arrays
    print("vol_5d array:\n", vol_5d)
    print("\nvol_10d array:\n", vol_10d)
    print("\nearnings_vol array:\n", earnings_vol)


if __name__ == "__main__":
    # Prompt user for stock ticker input
    user_input = input("Enter ticker (e.g., NVDA): ").upper().strip()
    if user_input:
        analyze_earnings_vol(user_input)
