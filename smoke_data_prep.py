"""Smoke-test variant of data_download.py.

Same output schema; only 5 hand-picked long-history tickers instead of the
full S&P 500. Used purely to exercise the downstream pipeline end-to-end
without spending 10+ minutes on yfinance.
"""
import os

import joblib
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from sklearn.preprocessing import StandardScaler

# Long-history tickers that are stable on yfinance.
tickers = ["AAPL", "MSFT", "IBM", "GE", "KO"]
print(f"Smoke tickers: {tickers}")

data = yf.download(tickers, period="max", interval="1d", auto_adjust=False)

target_length = 2048
stride = 400
raw_subseq_list = []
raw_logprice_list = []
raw_time_list = []
meta_list = []

for ticker in tickers:
    try:
        ticker_data = data.xs(ticker, axis=1, level=1)
        col = "Adj Close" if "Adj Close" in ticker_data.columns else "Close"
        prices = ticker_data[col].dropna()
        log_returns = np.log(prices / prices.shift(1)).dropna()
        log_prices = np.log(prices).dropna().iloc[1:]

        if len(log_returns) < target_length:
            print(f"{ticker}: only {len(log_returns)} log-returns; skipping")
            continue

        for start in range(0, len(log_returns) - target_length + 1, stride):
            sub_r = log_returns.iloc[start:start + target_length].values.reshape(-1, 1)
            sub_lp = log_prices.iloc[start:start + target_length].values.reshape(-1, 1)
            t_idx = log_returns.index[start:start + target_length]
            raw_subseq_list.append(sub_r)
            raw_logprice_list.append(sub_lp)
            raw_time_list.append(t_idx)
            meta_list.append((ticker, t_idx[0]))
    except Exception as e:
        print(f"{ticker}: {e}")

print(f"Total subsequences: {len(raw_subseq_list)}")
assert len(raw_subseq_list) > 0, "no subsequences produced"

global_scaler = StandardScaler().fit(np.vstack(raw_subseq_list))
norm_subseq = [global_scaler.transform(s).flatten() for s in raw_subseq_list]

global_log_scaler = StandardScaler().fit(np.vstack(raw_logprice_list))
norm_logprice = [global_log_scaler.transform(s).flatten() for s in raw_logprice_list]

subseq_t = [torch.tensor(s, dtype=torch.float32) for s in norm_subseq]
logprice_t = [torch.tensor(s, dtype=torch.float32) for s in norm_logprice]

time_t = []
for ts in raw_time_list:
    arr = np.array([pd.Timestamp(t).timestamp() for t in ts])
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    time_t.append(torch.tensor(arr, dtype=torch.float32))

os.makedirs("data", exist_ok=True)
joblib.dump(global_scaler, "data/global_scaler.pkl")
joblib.dump(global_log_scaler, "data/global_log_scaler.pkl")

torch.save(
    {"data": subseq_t, "timepoints": time_t, "meta": meta_list},
    "data/sp500_subseq.pt",
)
torch.save(
    {"data": logprice_t, "timepoints": time_t, "meta": meta_list},
    "data/sp500_subseq_log.pt",
)
torch.save(
    {
        "raw_data": raw_subseq_list,
        "raw_logprice": raw_logprice_list,
        "timepoints": raw_time_list,
        "meta": meta_list,
    },
    "data/sp500_subseq_original.pt",
)

print("Wrote: data/sp500_subseq.pt, data/sp500_subseq_log.pt, data/sp500_subseq_original.pt")
print("Wrote: data/global_scaler.pkl, data/global_log_scaler.pkl")
