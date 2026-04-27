import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

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
    x_min, x_max = -5.0, 5.0
    dist_x = np.linspace(x_min, x_max, granuality)
    diff = dist_x[1] - dist_x[0]
    dist_x_vis = (dist_x + diff)[:-1]
    dist_y = np.zeros(granuality - 1, dtype=np.float64)
    for e, (x1, x2) in enumerate(zip(dist_x[:-1], dist_x[1:])):
        dist_y[e] = x[np.logical_and(x > x1, x <= x2)].size
    dist_y /= x.size
    return dist_x_vis, dist_y


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
                waiting_pos.append(j - i)
                found_pos = True
            if not found_neg and cum <= -threshold:
                waiting_neg.append(j - i)
                found_neg = True
            if found_pos and found_neg:
                break
    return waiting_pos, waiting_neg


def compute_heavy_tail(returns, bins=50):
    hist, bin_edges = np.histogram(returns, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
    return bin_centers, hist


def compute_metrics(returns):
    metrics = {}
    metrics['linear_unpredictability'] = compute_autocorrelation(returns, max_lag=1000)
    metrics['volatility_clustering']   = compute_volatility_clustering(returns, max_lag=1000)
    metrics['leverage_effect_series'] = compute_leverage_effect_series(returns, max_lag=100)
    metrics['coarse_fine_asymmetry']  = compute_coarse_fine_series(returns, tau=5, max_lag=22)
    pos, neg = compute_gain_loss_distributions(returns, threshold=0.1)
    metrics['gain_loss_wait_pos'] = pos
    metrics['gain_loss_wait_neg'] = neg
    bin_c, hist = compute_heavy_tail(returns, bins=50)
    metrics['heavy_tail'] = (bin_c, hist)
    kurtosis = np.mean((returns - np.mean(returns))**4) /\
               (np.mean((returns - np.mean(returns))**2)**2 + 1e-8)
    metrics['kurtosis'] = kurtosis
    return metrics


def plot_all_stylized_facts(metrics_list, save_prefix="generated_metrics"):
    os.makedirs("ori_plot", exist_ok=True)
    # 1. Autocorrelation of Returns
    plt.figure()
    avg_lin = np.mean([m['linear_unpredictability'] for m in metrics_list], axis=0)
    plt.scatter(np.arange(1, len(avg_lin)+1), avg_lin, s=10)
    plt.ylim(-1,1)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.savefig(f"ori_plot/{save_prefix}_linear_unpredictability.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Heavy-Tailed Distribution
    plt.figure()
    avg_hist = np.mean([m['heavy_tail'][1] for m in metrics_list], axis=0)
    bin_c = metrics_list[0]['heavy_tail'][0]
    plt.scatter(bin_c, avg_hist, s=10, color='black')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel("Return"); plt.ylabel("Probability Density")
    plt.savefig(f"ori_plot/{save_prefix}_heavy_tail.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Volatility Clustering
    plt.figure()
    avg_vol = np.mean([m['volatility_clustering'] for m in metrics_list], axis=0)
    plt.scatter(np.arange(1, len(avg_vol)+1), avg_vol, s=10, color='black')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel("Lag"); plt.ylabel("Autocorrelation")
    plt.savefig(f"ori_plot/{save_prefix}_volatility_clustering.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Leverage Effect
    plt.figure()
    avg_leverage = np.mean([m['leverage_effect_series'] for m in metrics_list], axis=0)
    plt.plot(np.arange(1, len(avg_leverage)+1), avg_leverage, linewidth=1, color='black')
    plt.xlabel("Lag"); plt.ylabel("L(k)")
    plt.savefig(f"ori_plot/{save_prefix}_leverage_effect.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5. Coarse-Fine Volatility Asymmetry
    plt.figure()
    all_cf = np.array([m['coarse_fine_asymmetry'][1] for m in metrics_list])
    avg_cf = np.nanmean(all_cf, axis=0)
    lags = metrics_list[0]['coarse_fine_asymmetry'][0]
    pos_idx = np.where(lags>0)[0]; neg_idx = np.where(lags<0)[0]
    rho_neg = avg_cf[neg_idx][::-1]; rho_pos = avg_cf[pos_idx]
    delta = rho_pos - rho_neg
    plt.scatter(lags, avg_cf, s=10, label=r'$\rho(k)$')
    plt.scatter(lags[pos_idx], delta, s=10, label=r'$\Delta\rho(k)$')
    plt.axhline(0, linestyle='--', linewidth=1)
    plt.xlabel("Lag"); plt.ylabel("Correlation"); plt.legend()
    plt.savefig(f"ori_plot/{save_prefix}_coarse_fine.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 6. Gain/Loss Asymmetry
    plt.figure()
    pos = []
    neg = []
    for m in metrics_list:
        pos.extend(m['gain_loss_wait_pos'])
        neg.extend(m['gain_loss_wait_neg'])
    if pos or neg:
        max_t = max(max(pos, default=1), max(neg, default=1))
        bins = np.arange(1, max_t+2) - 0.5
        hpos, be = np.histogram(pos, bins=bins, density=True)
        hneg, _  = np.histogram(neg, bins=bins, density=True)
        bc = (be[:-1] + be[1:]) / 2
        plt.scatter(bc, hpos, marker='o', alpha=0.7, label="Positive")
        plt.scatter(bc, hneg, marker='o', alpha=0.7, label="Negative")
        plt.xscale('log'); plt.xlim(1,1000)
        plt.xlabel("Waiting Time"); plt.ylabel("Probability"); plt.legend()
    plt.savefig(f"ori_plot/{save_prefix}_gain_loss_asymmetry.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str,
                        default='data/sp500_subseq_original.pt',
                        help="Path to processed subsequences file")
    parser.add_argument('--scaler_file', type=str,
                        default='data/global_scaler.pkl',
                        help="Path to global scaler file")
    parser.add_argument('--num_samples', type=int,
                        default=1000,
                        help="Number of samples for evaluation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    processed_data = torch.load(
        args.data_file,
        map_location=device,
        weights_only=False,
    )
    subseq_list = processed_data['raw_data']
    print(f"Loaded subsequences: {len(subseq_list)}")

    selected = subseq_list[:args.num_samples]
    
    generated_samples = []
    for seq in selected:
        if isinstance(seq, torch.Tensor):
            arr = seq.flatten().cpu().numpy()
        else:
            arr = np.asarray(seq).flatten()
        generated_samples.append(arr)

    pdf_x = []
    pdf_y = []
    metrics_list = []
    for sample in generated_samples:
        metrics_list.append(compute_metrics(sample))
        dx, dy = linear_pdf(normalize_time_series(sample), granuality=100)
        pdf_x.append(dx)
        pdf_y.append(dy)

    plot_all_stylized_facts(metrics_list, save_prefix="generated_metrics")

    # Heavy-tail via linear PDF on normalized returns.
    avg_dy = np.mean(np.vstack(pdf_y), axis=0)
    common_dx = pdf_x[0]
    mask = common_dx > 0
    pos_x = common_dx[mask]
    pos_y = avg_dy[mask]
    plt.figure()
    plt.scatter(pos_x, pos_y, s=10, color='black')
    plt.xlabel("Normalized price return")
    plt.ylabel("Probability density")
    plt.xscale('log'); plt.yscale('log')
    plt.savefig(f"ori_plot/generated_metrics_heavy_tail_linearpdf.png",
                dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
