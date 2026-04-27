import os
import argparse

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import powerlaw
import torch
from tqdm.auto import tqdm

from generate import predictor_corrector_sampling
from model_config import MODEL_CONFIG
from networks import diff_CSDI
from utils import str2bool

mpl.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})


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
    """L(k) = ( E[r(t) * |r(t+k)|^2] - E[r(t)] * E[|r(t)|^2] ) / (E[|r(t)|^2])^2."""
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
    # v_coarse: |sum r| over window tau; v_fine: sum |r| over window tau.
    v_coarse = np.array([np.abs(np.sum(returns[i:i+tau])) for i in range(n-tau+1)])
    v_fine = np.array([np.sum(np.abs(returns[i:i+tau])) for i in range(n-tau+1)])
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
    metrics['linear_unpredictability'] = compute_autocorrelation(returns, max_lag=1000)
    metrics['volatility_clustering'] = compute_volatility_clustering(returns, max_lag=1000)
    metrics['leverage_effect_series'] = compute_leverage_effect_series(returns, max_lag=100)
    metrics['coarse_fine_asymmetry'] = compute_coarse_fine_series(returns, tau=5, max_lag=22)
    waiting_pos, waiting_neg = compute_gain_loss_distributions(returns, threshold=0.1)
    metrics['gain_loss_wait_pos'] = waiting_pos
    metrics['gain_loss_wait_neg'] = waiting_neg
    bin_centers, hist = compute_heavy_tail(returns, bins=50)
    metrics['heavy_tail'] = (bin_centers, hist)
    kurtosis = np.mean((returns - np.mean(returns))**4) / (np.mean((returns - np.mean(returns))**2)**2 + 1e-8)
    metrics['kurtosis'] = kurtosis
    return metrics

def plot_all_stylized_facts(metrics_list, plot_folder='ori_plot', save_prefix="generated_metrics"):
    os.makedirs(plot_folder, exist_ok=True)

    # 1. Linear unpredictability (autocorrelation of returns).
    plt.figure()
    avg_lin = np.mean([m['linear_unpredictability'] for m in metrics_list], axis=0)
    x_vals = np.arange(1, len(avg_lin)+1)
    plt.scatter(x_vals, avg_lin, color='black', s=10, label="Average")
    plt.title("Linear Unpredictability (Autocorrelation of Returns)")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.ylim(-1, 1)
    plt.savefig(f"{plot_folder}/{save_prefix}_linear_unpredictability.png")
    plt.close()

    # 2. Heavy-tailed return distribution (log-log).
    plt.figure()
    all_hist = [m['heavy_tail'][1] for m in metrics_list]
    avg_hist = np.mean(all_hist, axis=0)
    bin_centers = metrics_list[0]['heavy_tail'][0]
    plt.scatter(bin_centers, avg_hist, color='black', s=10, label="Average")
    plt.xlabel("Return")
    plt.ylabel("Probability density")
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(f"{plot_folder}/{save_prefix}_heavy_tail.png")
    plt.close()

    # 3. Volatility clustering (autocorrelation of absolute returns, log-log).
    plt.figure()
    avg_vol = np.mean([m['volatility_clustering'] for m in metrics_list], axis=0)
    x_vals = np.arange(1, len(avg_vol)+1)
    plt.scatter(x_vals, avg_vol, color='black', s=10, label="Average")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(f"{plot_folder}/{save_prefix}_volatility_clustering.png")
    plt.close()

    # 4. Leverage effect.
    plt.figure()
    avg_leverage = np.mean([m['leverage_effect_series'] for m in metrics_list], axis=0)
    lags = np.arange(1, len(avg_leverage)+1)
    plt.scatter(lags, avg_leverage, color='black', s=10, label="Average")
    plt.xlabel("Lag")
    plt.ylabel("L(k)")
    plt.savefig(f"{plot_folder}/{save_prefix}_leverage_effect.png")
    plt.close()

    # 5. Coarse-fine volatility correlation and asymmetry.
    plt.figure()
    all_rho_cf = np.array([m['coarse_fine_asymmetry'][1] for m in metrics_list])
    avg_rho_cf = np.nanmean(all_rho_cf, axis=0)
    lags = metrics_list[0]['coarse_fine_asymmetry'][0]
    pos_idx = np.where(lags > 0)[0]
    neg_idx = np.where(lags < 0)[0]
    # Reverse negative lags so indices align with positive lags for Δρ(k) = ρ(k) - ρ(-k).
    rho_neg_sorted = avg_rho_cf[neg_idx][::-1]
    rho_pos = avg_rho_cf[pos_idx]
    delta_rho = rho_pos - rho_neg_sorted
    pos_lags = lags[pos_idx]
    plt.scatter(lags, avg_rho_cf, color='blue', s=10, label=r'$\rho(k)$')
    plt.scatter(pos_lags, delta_rho, color='orange', s=10, label=r'$\Delta \rho(k)$')
    plt.axhline(y=0.0, color='gray', linestyle='--', linewidth=1)
    plt.title("Coarse-Fine Volatility Correlation and Asymmetry")
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    plt.legend()
    plt.savefig(f"{plot_folder}/{save_prefix}_coarse_fine.png")
    plt.close()

    # 6. Gain/loss asymmetry (waiting-time distributions).
    plt.figure()
    all_wait_pos = []
    all_wait_neg = []
    for m in metrics_list:
        all_wait_pos.extend(m['gain_loss_wait_pos'])
        all_wait_neg.extend(m['gain_loss_wait_neg'])

    bins = np.arange(1, max(max(all_wait_pos, default=1), max(all_wait_neg, default=1)) + 2) - 0.5
    hist_pos, bin_edges = np.histogram(all_wait_pos, bins=bins, density=True)
    hist_neg, _ = np.histogram(all_wait_neg, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.scatter(bin_centers, hist_pos, color='red', label="Positive Return", marker='o', alpha=0.7)
    plt.scatter(bin_centers, hist_neg, color='blue', label="Negative Return", marker='o', alpha=0.7)
    plt.title("Gain/Loss Asymmetry")
    plt.xlabel("Waiting Time (t')")
    plt.ylabel("Return Time Probability")
    plt.xscale('log')
    plt.xlim(1, 1000)
    plt.legend()
    plt.savefig(f"{plot_folder}/{save_prefix}_gain_loss_asymmetry.png")
    plt.close()


def plot_time_series(i, sample, save_folder="time_series"):
    os.makedirs(save_folder, exist_ok=True)
    plt.figure()
    plt.plot(sample)
    plt.title(f"Generated Sample {i+1}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.savefig(os.path.join(save_folder, f"time_series_{i+1}.png"))
    plt.close()


def recover_price_series(log_returns, base_price=100):
    return base_price * np.exp(np.cumsum(log_returns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='save_model_bs_cosine/model_epoch_1000.pth')
    parser.add_argument('--sde', type=str, default='VE')
    parser.add_argument('--alpha', type=float, default=1, help="0.5 or 1.0")
    parser.add_argument('--sigma_min', type=float, default=0.01)
    parser.add_argument('--sigma_max', type=float, default=1.0)
    parser.add_argument('--emb_time_dim', type=int, default=128)
    parser.add_argument('--emb_feature_dim', type=int, default=32)
    parser.add_argument('--noise_schedule', type=str, default="cosine")
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--snr', type=float, default=0.2)
    parser.add_argument('--n_corr', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=120, help="Number of samples for evaluation")
    parser.add_argument('--is_unconditional', type=str2bool, default=True)
    parser.add_argument('--plot_folder', type=str, default='BS_cosine_plot_', help="Folder to save stylized-fact plots.")
    parser.add_argument('--plot_timeseries', type=str, default='BS_cosine_timeseries', help="Folder to save time-series plots.")
    parser.add_argument('--csv_folder', type=str, default='BS_cosine_csv_', help="Folder to save generated return CSVs.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = dict(MODEL_CONFIG)
    config["side_dim"] = args.emb_time_dim + config["emb_feature_dim"]
    input_dim = 1 if args.is_unconditional else 2
    print(f"is_unconditional={args.is_unconditional!r} -> input_dim={input_dim}")
    model = diff_CSDI(config, input_dim).to(device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    target_length = 2048
    dummy_time = torch.linspace(0, 1, target_length, device=device).unsqueeze(0)

    if args.alpha == 1:
        scaler = joblib.load("data/global_log_scaler.pkl")
    else:
        scaler = joblib.load("data/global_scaler.pkl")

    generated_samples = []
    os.makedirs(args.csv_folder, exist_ok=True)
    for idx in tqdm(range(args.num_samples), desc="Generating Samples"):
        x_clean_dummy = torch.randn(1, 1, target_length, device=device, dtype=torch.float32)

        sample = predictor_corrector_sampling(
            model=model,
            x_clean=x_clean_dummy,  
            x_tp=dummy_time,  # (1,2048)
            sde=args.sde,
            alpha=args.alpha,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            noise_schedule=args.noise_schedule,
            emb_time_dim=args.emb_time_dim,
            is_unconditional=args.is_unconditional,
            steps=args.steps,
            snr=args.snr,
            n_corr=args.n_corr,
            num_samples=1,
            device=device
        )
        # alpha==1 samples live in log-price space; convert to log-returns.
        if args.alpha == 1:
            sample_np = sample.squeeze(0).detach().cpu().numpy().reshape(-1, 1)
            logprice = scaler.inverse_transform(sample_np).flatten()
            log_returns = np.diff(logprice)
            # Pad a leading 0 to keep the original target_length.
            log_returns = np.concatenate(([0.0], log_returns))
            generated_samples.append(log_returns)
            pd.DataFrame({"log_return": log_returns}).to_csv(
                f"{args.csv_folder}/log_returns_sample_{idx+1}.csv", index=False
            )
        else:
            sample_np = sample.squeeze(0).detach().cpu().numpy().reshape(-1, 1)
            returns = scaler.inverse_transform(sample_np).flatten()
            generated_samples.append(returns)
            pd.DataFrame({"return": returns}).to_csv(
                f"{args.csv_folder}/returns_sample_{idx+1}.csv", index=False
            )

    all_returns = np.concatenate(generated_samples)
    data = np.abs(all_returns)
    data = data[data > 0]
    fit = powerlaw.Fit(data, xmin=None)
    print(f"Powerlaw MLE α = {fit.alpha:.2f}, xmin = {fit.xmin:.4f}")

    metrics_list = [compute_metrics(s) for s in generated_samples]
    plot_all_stylized_facts(metrics_list, plot_folder=args.plot_folder, save_prefix="generated_metrics")

    for i, sample in enumerate(generated_samples):
        price_series = recover_price_series(sample, base_price=100)
        plot_time_series(i, price_series, save_folder=args.plot_timeseries)


if __name__ == "__main__":
    main()
