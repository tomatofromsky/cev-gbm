import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})


def normalize_time_series(x):
    """Standardize to mean 0 / std 1."""
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / (std + 1e-8)


def linear_pdf(x, granuality=100):
    """Empirical PDF over [-5, 5] with `granuality` equal-width bins."""
    x_max, x_min = 5.0, -5.0
    dist_x = np.linspace(x_min, x_max, granuality)
    diff = dist_x[1] - dist_x[0]
    dist_x_visual = (dist_x + diff)[:-1]
    dist_y = np.zeros(granuality - 1, dtype=np.float64)
    for e, (x1, x2) in enumerate(zip(dist_x[:-1], dist_x[1:])):
        dist_y[e] = x[np.logical_and(x > x1, x <= x2)].size
    dist_y /= x.size
    return dist_x_visual, dist_y


def compute_autocorrelation(x, max_lag=20):
    x = x - np.mean(x)
    ac = []
    for lag in range(1, max_lag+1):
        if len(x) - lag <= 1:
            ac.append(0)
        else:
            r = np.corrcoef(x[:-lag], x[lag:])[0, 1]
            ac.append(r)
    return np.array(ac)

def compute_volatility_clustering(returns, max_lag=20):
    abs_returns = np.abs(returns)
    return compute_autocorrelation(abs_returns, max_lag)

def compute_heavy_tail(returns, bins=50):
    hist, bin_edges = np.histogram(returns, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
    return bin_centers, hist

def compute_leverage_effect_series(returns, max_lag=10):
    mean_r2 = np.mean(returns**2)
    denom = mean_r2**2
    Lk = []
    for lag in range(1, max_lag+1):
        if len(returns) <= lag:
            Lk.append(np.nan)
        else:
            r_t = returns[:-lag]
            r_tlag = returns[lag:]
            term1 = r_t * (r_tlag**2)
            term2 = np.mean(r_t)
            term3 = np.mean(r_t**2)
            numerator = np.mean(term1) - term2 * term3
            Lk.append(numerator / denom)
    return np.array(Lk)

def compute_coarse_fine_series(returns, tau=5, max_lag=20):
    n = len(returns)
    if n < tau:
        return None, None
    v_coarse = np.array([np.abs(np.sum(returns[i:i+tau])) for i in range(n-tau+1)])
    v_fine   = np.array([np.sum(np.abs(returns[i:i+tau])) for i in range(n-tau+1)])
    lags = np.arange(-max_lag, max_lag+1)
    rho_cf = []
    for lag in lags:
        if lag >= 0:
            if len(v_coarse) - lag <= 1:
                rho_cf.append(np.nan)
            else:
                corr = np.corrcoef(v_coarse[lag:], v_fine[:len(v_fine)-lag])[0, 1]
                rho_cf.append(corr)
        else:
            lag_pos = -lag
            if len(v_coarse) - lag_pos <= 1:
                rho_cf.append(np.nan)
            else:
                corr = np.corrcoef(v_coarse[:len(v_coarse)-lag_pos], v_fine[lag_pos:])[0, 1]
                rho_cf.append(corr)
    return lags, np.array(rho_cf)

def compute_gain_loss_distributions(returns, threshold=0.01):
    waiting_pos = []
    waiting_neg = []
    n = len(returns)
    for i in range(n):
        cum = 0.0
        found_pos = False
        found_neg = False
        for j in range(i, n):
            cum += returns[j]
            if not found_pos and cum >= threshold:
                waiting_pos.append(j-i)
                found_pos = True
            if not found_neg and cum <= -threshold:
                waiting_neg.append(j-i)
                found_neg = True
            if found_pos and found_neg:
                break
    return waiting_pos, waiting_neg

def compute_metrics(returns):
    metrics = {}
    metrics['linear_unpredictability']   = compute_autocorrelation(returns, max_lag=1000)
    metrics['volatility_clustering']     = compute_volatility_clustering(returns, max_lag=1000)
    metrics['leverage_effect_series']    = compute_leverage_effect_series(returns, max_lag=100)
    metrics['coarse_fine_asymmetry']     = compute_coarse_fine_series(returns, tau=5, max_lag=22)
    waiting_pos, waiting_neg = compute_gain_loss_distributions(returns, threshold=0.1)
    metrics['gain_loss_wait_pos'] = waiting_pos
    metrics['gain_loss_wait_neg'] = waiting_neg
    bin_centers, hist = compute_heavy_tail(returns, bins=50)
    metrics['heavy_tail'] = (bin_centers, hist)
    kurtosis = np.mean((returns - np.mean(returns))**4) / (np.mean((returns - np.mean(returns))**2)**2 + 1e-8)
    metrics['kurtosis'] = kurtosis
    return metrics

def plot_all_stylized_facts(metrics_list, plot_folder='csv_plots', save_prefix="from_csv"):
    os.makedirs(plot_folder, exist_ok=True)
    num_samples = len(metrics_list)
    
    # 1. Linear Unpredictability
    plt.figure()
    avg_lin = np.mean([m['linear_unpredictability'] for m in metrics_list], axis=0)
    x_vals = np.arange(1, len(avg_lin)+1)
    plt.scatter(x_vals, avg_lin, color='black', s=10)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.ylim(-1, 1)
    plt.savefig(f"{plot_folder}/{save_prefix}_linear_unpredictability.png",dpi=300, bbox_inches="tight")
    plt.close()
    
    # 2. Heavy-Tailed Distribution (log-log)
    plt.figure()
    all_hist = [m['heavy_tail'][1] for m in metrics_list]
    avg_hist = np.mean(all_hist, axis=0)
    bin_centers = metrics_list[0]['heavy_tail'][0]
    plt.scatter(bin_centers, avg_hist, color='black', s=10)
    plt.xlabel("Return")
    plt.ylabel("Probability Density")
    plt.xscale('log')
    plt.yscale('log')
    #plt.xlim(1e-3, 1e0)
    #plt.ylim(1e-3, 1e2)
    plt.savefig(f"{plot_folder}/{save_prefix}_heavy_tail.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 3. Volatility Clustering (log-log)
    plt.figure()
    avg_vol = np.mean([m['volatility_clustering'] for m in metrics_list], axis=0)
    x_vals = np.arange(1, len(avg_vol)+1)
    plt.scatter(x_vals, avg_vol, color='black', s=10)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.xscale('log')
    plt.yscale('log')
    #plt.ylim(1e-5, 1e0)
    plt.savefig(f"{plot_folder}/{save_prefix}_volatility_clustering.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 4. Leverage Effect
    plt.figure()
    avg_leverage = np.mean([m['leverage_effect_series'] for m in metrics_list], axis=0)
    lags = np.arange(1, len(avg_leverage)+1)
    #plt.scatter(lags, avg_leverage, color='black', s=10)
    plt.plot(lags, avg_leverage, color='black', linewidth=1)
    plt.xlabel("Lag")
    plt.ylabel("L(k)")
    plt.savefig(f"{plot_folder}/{save_prefix}_leverage_effect.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 5. Coarse-Fine Volatility Correlation & Asymmetry
    plt.figure()
    all_rho_cf = np.array([m['coarse_fine_asymmetry'][1] for m in metrics_list])
    avg_rho_cf = np.nanmean(all_rho_cf, axis=0)
    lags = metrics_list[0]['coarse_fine_asymmetry'][0]
    pos_idx = np.where(lags > 0)[0]
    neg_idx = np.where(lags < 0)[0]
    rho_neg_sorted = avg_rho_cf[neg_idx][::-1]
    rho_pos = avg_rho_cf[pos_idx]
    delta_rho = rho_pos - rho_neg_sorted
    pos_lags = lags[pos_idx]
    plt.scatter(lags, avg_rho_cf, color='blue', s=10, label=r'$\rho(k)$')
    plt.scatter(pos_lags, delta_rho, color='orange', s=10, label=r'$\Delta \rho(k)$')
    plt.axhline(y=0.0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    plt.legend()
    plt.savefig(f"{plot_folder}/{save_prefix}_coarse_fine.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 6. Gain/Loss Asymmetry (Waiting Time Distributions)
    plt.figure()
    all_wait_pos = []
    all_wait_neg = []
    for m in metrics_list:
        all_wait_pos.extend(m['gain_loss_wait_pos'])
        all_wait_neg.extend(m['gain_loss_wait_neg'])
    if len(all_wait_pos) + len(all_wait_neg) > 0:
        bins = np.arange(1, max(max(all_wait_pos or [1]), max(all_wait_neg or [1])) + 2) - 0.5
        hist_pos, bin_edges = np.histogram(all_wait_pos, bins=bins, density=True)
        hist_neg, _ = np.histogram(all_wait_neg, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.scatter(bin_centers, hist_pos, color='red', marker='o', alpha=0.7, label="Positive Return")
        plt.scatter(bin_centers, hist_neg, color='blue', marker='o', alpha=0.7, label="Negative Return")
        plt.xscale('log')
        plt.xlim(1, 1000)
        plt.xlabel("Waiting Time")
        plt.ylabel("Probability")
        plt.legend()
    plt.savefig(f"{plot_folder}/{save_prefix}_gain_loss_asymmetry.png", dpi=300, bbox_inches="tight")
    plt.close()


def recover_price_series(log_returns, base_price=100):
    cum_log_returns = np.cumsum(log_returns)
    price_series = base_price * np.exp(cum_log_returns)
    return price_series

def plot_time_series(i, sample_returns, save_folder="csv_timeseries"):
    os.makedirs(save_folder, exist_ok=True)
    price_series = recover_price_series(sample_returns, base_price=100)
    plt.figure(figsize=(8, 4))
    plt.plot(price_series, linewidth=1)
    plt.title(f"Recovered Price Series #{i+1}")
    plt.xlabel("Time step")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"time_series_{i+1}.png"))
    plt.close()


def main():
    csv_folder = "BS_exponential_csv_0623"
    plot_folder = "BS_exponential_plot_0623"
    ts_plot_folder = "BS_exponential_timeseries"

    os.makedirs(plot_folder, exist_ok=True)
    os.makedirs(ts_plot_folder, exist_ok=True)

    metrics_list = []
    pdf_list_x = []
    pdf_list_y = []
    csv_files = sorted([f for f in os.listdir(csv_folder) if f.endswith(".csv")])

    for idx, fname in enumerate(csv_files):
        path = os.path.join(csv_folder, fname)
        df = pd.read_csv(path)

        if "return" in df.columns:
            returns = df["return"].values
        elif "log_return" in df.columns:
            returns = df["log_return"].values
        else:
            print(f"[Warning] '{fname}' has no 'return' or 'log_return' column; skipping.")
            continue

        metrics_list.append(compute_metrics(returns))
        plot_time_series(idx, returns, save_folder=ts_plot_folder)

        x_norm = normalize_time_series(returns)
        dist_x, dist_y = linear_pdf(x_norm, granuality=100)
        pdf_list_x.append(dist_x)
        pdf_list_y.append(dist_y)

    if len(metrics_list) > 0:
        plot_all_stylized_facts(metrics_list, plot_folder=plot_folder, save_prefix="generated_metrics")

        # Average per-sample dist_y; dist_x is identical across samples (fixed [-5,5]).
        avg_dist_y = np.mean(np.vstack(pdf_list_y), axis=0)
        dist_x_common = pdf_list_x[0]
        pos_mask = dist_x_common > 0
        pos_x = dist_x_common[pos_mask]
        pos_y = avg_dist_y[pos_mask]

        plt.figure()
        plt.scatter(pos_x, pos_y, color='black', s=10)
        plt.xlabel("Normalized price return")
        plt.ylabel("Probability density")
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(f"{plot_folder}/generated_metrics_heavy_tail.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Stylized-facts plots saved to: '{plot_folder}/'")
        print(f"Time-series plots saved to:   '{ts_plot_folder}/'")
    else:
        print("No usable CSV samples; nothing to plot.")


if __name__ == "__main__":
    main()
