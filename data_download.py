import os
from datetime import datetime
from io import StringIO

import certifi
import joblib
import numpy as np
import pandas as pd
import requests
import torch
import yfinance as yf
from sklearn.preprocessing import StandardScaler


# S&P 500 ticker list from Wikipedia. Fetch through requests + certifi so the
# call works on Python distributions whose stdlib ssl module has no usable CA
# bundle (e.g., uv-managed cpython, some minimal containers). A non-default
# User-Agent is required because Wikipedia returns 403 for python-requests.
# Pass the HTML through StringIO so newer pandas/lxml parse it as a buffer
# rather than re-interpreting it as a URL or file path.
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
resp = requests.get(
    url,
    verify=certifi.where(),
    timeout=30,
    headers={"User-Agent": "gbm-data-prep/0.1 (https://github.com/) python-requests"},
)
resp.raise_for_status()
tables = pd.read_html(StringIO(resp.text))
sp500_table = tables[0]
sp500_tickers = sp500_table['Symbol'].tolist()

# These tickers are known to fail extraction on yfinance for this workflow.
bad_tickers = ['ETR', 'BRK.B', 'BF.B', 'LEN']
tickers = [ticker for ticker in sp500_tickers if ticker not in bad_tickers]
print(tickers)

data = yf.download(tickers, period="max", interval="1d")

# Keep tickers whose first valid price is before start_threshold (i.e., long history).
valid_tickers = []
start_threshold = datetime(1900, 1, 1)

for ticker in tickers:
    try:
        ticker_data = data.xs(ticker, axis=1, level=1)
        col = 'Adj Close' if 'Adj Close' in ticker_data.columns else 'Close'
        first_date = ticker_data[col].first_valid_index()
        if first_date is not None and first_date < start_threshold:
            valid_tickers.append(ticker)
    except Exception as e:
        print(f"{ticker} extraction error: {e}")

print("Tickers with long history:", valid_tickers)
print("Total:", len(valid_tickers))

# Extract sliding-window subsequences of log returns and log prices per ticker.
# A single global StandardScaler is fit across all subsequences.
target_length = 2048
stride = 400
raw_subseq_list = []      # raw log returns per subsequence
raw_logprice_list = []    # raw log prices per subsequence
raw_time_list = []        # timestamps per subsequence
meta_list = []            # (ticker, start_date) per subsequence

for ticker in tickers:
    try:
        ticker_data = data.xs(ticker, axis=1, level=1)
        col = 'Adj Close' if 'Adj Close' in ticker_data.columns else 'Close'
        prices = ticker_data[col]
        log_returns = np.log(prices / prices.shift(1)).dropna()
        # Drop the first log-price to align lengths with log_returns.
        log_prices = np.log(prices).dropna().iloc[1:]

        if len(log_returns) < target_length:
            print(f"{ticker} log-return length {len(log_returns)} < {target_length}; skipping.")
            continue

        for start_idx in range(0, len(log_returns) - target_length + 1, stride):
            subseq = log_returns.iloc[start_idx:start_idx + target_length].values.reshape(-1, 1)
            subseq_logprice = log_prices.iloc[start_idx:start_idx + target_length].values.reshape(-1, 1)
            time_subseq = log_returns.index[start_idx:start_idx + target_length]
            raw_subseq_list.append(subseq)
            raw_logprice_list.append(subseq_logprice)
            raw_time_list.append(time_subseq)
            meta_list.append((ticker, time_subseq[0]))
    except Exception as e:
        print(f"{ticker} subsequence extraction error: {e}")

if len(raw_subseq_list) == 0:
    raise ValueError("No subsequences were produced.")

print(f"Total subsequences: {len(raw_subseq_list)}")

# Fit one global scaler across all log-return subsequences.
all_data = np.vstack(raw_subseq_list)
global_scaler = StandardScaler()
global_scaler.fit(all_data)
print("log-return scaler mean:", global_scaler.mean_, "std:", np.sqrt(global_scaler.var_))
norm_subseq_list = [global_scaler.transform(s).flatten() for s in raw_subseq_list]

# Fit a separate global scaler across all log-price subsequences.
all_logprice_data = np.vstack(raw_logprice_list)
global_log_scaler = StandardScaler()
global_log_scaler.fit(all_logprice_data)
print("log-price scaler mean:", global_log_scaler.mean_, "std:", np.sqrt(global_log_scaler.var_))
norm_logprice_list = [global_log_scaler.transform(s).flatten() for s in raw_logprice_list]

subseq_tensor_list = [torch.tensor(s, dtype=torch.float32) for s in norm_subseq_list]

time_tensor_list = []
for time_seq in raw_time_list:
    # Unix timestamps normalized to [0,1] within each subsequence.
    t_arr = np.array([pd.Timestamp(t).timestamp() for t in time_seq])
    t_arr_norm = (t_arr - t_arr.min()) / (t_arr.max() - t_arr.min() + 1e-8)
    time_tensor_list.append(torch.tensor(t_arr_norm, dtype=torch.float32))

logprice_tensor_list = [torch.tensor(s, dtype=torch.float32) for s in norm_logprice_list]

scaler_file = "global_scaler.pkl"
joblib.dump(global_scaler, scaler_file)
print(f"Saved log-return scaler to {scaler_file}.")

global_log_scaler_file = "global_log_scaler.pkl"
joblib.dump(global_log_scaler, global_log_scaler_file)
print(f"Saved log-price scaler to {global_log_scaler_file}.")

processed_file = "sp500_subseq.pt"
os.makedirs(os.path.dirname(processed_file) or ".", exist_ok=True)
torch.save(
    {'data': subseq_tensor_list, 'timepoints': time_tensor_list, 'meta': meta_list},
    processed_file,
)
print(f"Saved normalized log-return subsequences to '{processed_file}'.")

processed_logprice_file = "sp500_subseq_log.pt"
torch.save(
    {'data': logprice_tensor_list, 'timepoints': time_tensor_list, 'meta': meta_list},
    processed_logprice_file,
)
print(f"Saved normalized log-price subsequences to '{processed_logprice_file}'.")

original_file = "sp500_subseq_original.pt"
torch.save(
    {
        'raw_data': raw_subseq_list,
        'raw_logprice': raw_logprice_list,
        'timepoints': raw_time_list,
        'meta': meta_list,
    },
    original_file,
)
print(f"Saved raw (unscaled) subsequences to '{original_file}'.")