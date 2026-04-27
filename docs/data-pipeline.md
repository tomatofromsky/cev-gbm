# Data pipeline

Source: [`data_download.py`](../data_download.py).

The pipeline has one responsibility: turn "S&P 500 tickers" into a bag of fixed-length, normalized subsequences that the diffusion model can consume. Running `python data_download.py` once produces three `.pt` files and two `.pkl` scalers.

The reference paper (§3.1.2) describes the same pipeline: pull S&P 500 history from `yfinance`, drop tickers with non-standard symbols, keep the long-history ones, compute log-returns, then cut **length-2048 windows at stride 400**. The paper specifies a "more than 40 years" history filter; the codebase uses a more permissive 1900-01-01 cutoff (effectively the same in practice — both keep multi-decade histories and drop recent IPOs).

## Stages

### 1. Ticker list — `data_download.py:13-21`

Scrapes the current S&P 500 table from Wikipedia with `pandas.read_html`. Four tickers (`ETR`, `BRK.B`, `BF.B`, `LEN`) are hard-excluded; `BRK.B` and `BF.B` are explicitly called out in the paper (§3.1.2) for non-standard symbols, the others fail `yfinance`'s multi-ticker extraction in this workflow.

### 2. Download — `data_download.py:23`

```python
data = yf.download(tickers, period="max", interval="1d")
```

One `yfinance` call for all tickers. Returns a multi-index DataFrame; per-ticker slices are taken with `data.xs(ticker, axis=1, level=1)`. `Adj Close` is preferred when present, otherwise `Close`.

### 3. Long-history filter — `data_download.py:26-40`

Tickers whose first valid price is **before 1900-01-01** are kept. This threshold is permissive; in practice it drops tickers with short histories (e.g., recent IPOs with < 2048 trading days of data). If you expand the universe, tune `start_threshold`.

### 4. Sliding-window extraction — `data_download.py:42-73`

For each kept ticker, the script computes:

- `log_returns = log(price_t / price_{t-1})` — a length-N series.
- `log_prices = log(price_t)` — the same series sans the first point (aligned with returns).

Then it slides a window of length **2048** with stride **400** over `log_returns`:

```python
target_length = 2048
stride        = 400
for start_idx in range(0, len(log_returns) - target_length + 1, stride):
    ...
```

Each window contributes **two** parallel subsequences (one in return-space, one in log-price-space) and one matching timestamp window.

Tickers with fewer than 2048 log-returns are skipped.

### 5. Global normalization — `data_download.py:81-92`

Two independent `StandardScaler`s, each fit across **all** subsequences pooled:

- `global_scaler` over log-returns → `global_scaler.pkl`
- `global_log_scaler` over log-prices → `global_log_scaler.pkl`

The scalers are "global" in two senses: across tickers **and** across windows. This matters: a single ticker's window is not standardized against its own mean/std. The implication is that generated samples, after `inverse_transform`, live in a shared return/log-price space and can be mixed across tickers.

### 6. Time normalization — `data_download.py:96-101`

For each window, the 2048 pandas timestamps are converted to Unix seconds and then min-max scaled to `[0, 1]`:

```python
t_arr_norm = (t_arr - t_arr.min()) / (t_arr.max() - t_arr.min() + 1e-8)
```

These `[0, 1]` timepoints feed into `utils.time_embedding` (a sinusoidal encoding; see [architecture.md](architecture.md#side-information)) and act as the model's "position" signal.

### 7. Output files — `data_download.py:113-138`

| File | Used by | Contents |
|------|---------|----------|
| `sp500_subseq.pt` | `train.py --alpha 0` | Normalized log-returns (list of 1-D tensors), normalized timepoints, meta |
| `sp500_subseq_log.pt` | `train.py --alpha 1` | Normalized log-prices + same timepoints & meta |
| `sp500_subseq_original.pt` | `ori_plot_result.py` | Raw (unscaled) returns & log-prices + raw timestamps — the real-data baseline for evaluation |
| `global_scaler.pkl` | `plot_result.py` (α=0) | `StandardScaler` for log-returns |
| `global_log_scaler.pkl` | `plot_result.py` (α=1) | `StandardScaler` for log-prices |

Each `.pt` shard has the schema:

```python
{
  "data":       [torch.Tensor(2048,)] * N,      # α-0/α-1: normalized; _original: raw
  "timepoints": [torch.Tensor(2048,)] * N,      # in [0, 1], per-window min-max
  "meta":       [(ticker: str, start_timestamp)] * N,
}
```

## Practical notes

- **Output location.** The script writes `sp500_subseq*.pt` to the **current working directory**, not `data/`. The rest of the codebase expects them under `data/` — move them (or `cd data/` before running, or patch the script).
- **Scaler location.** Same deal: `global_scaler.pkl` / `global_log_scaler.pkl` land in CWD; `plot_result.py:271-273` reads them from `data/`.
- **`meta` is informational.** The training loop never reads `meta`; it's there for post-hoc inspection of which ticker/date produced which sample.
- **Windows overlap.** With `target_length = 2048` and `stride = 400`, successive windows share 1648 points. This inflates the dataset size (and the effective correlation between batches).
- **Re-running is expensive.** A full `yfinance` multi-download of ~500 tickers at `period="max"` is slow and rate-limited. Run it once, then work from the cached `.pt` files.
- **Reproducibility.** Wikipedia's S&P 500 list changes; `yfinance` history also updates. Running the script on different dates yields different datasets.

## The two "α" shards, at a glance

```
raw prices  ─┬─► log(p_t / p_{t-1}) ──► StandardScaler ──► sp500_subseq.pt       (α=0)
             └─► log(p_t)           ──► StandardScaler ──► sp500_subseq_log.pt   (α=1)
```

The rest of the project treats these interchangeably as 1-D standardized signals. α=1's post-sampling recovery step (in `plot_result.py`) reverses this: `scaler.inverse_transform → np.diff → log-return`. See [sampling.md](sampling.md#alpha-modes).
