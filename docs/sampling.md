# Sampling

Sources: [`generate.py`](../generate.py), and the sampling entry in [`plot_result.py`](../plot_result.py).

Sampling is reverse-time integration of the learned SDE using a **predictor–corrector** scheme (Song et al., 2021). The predictor is reverse Euler; the corrector is Langevin MCMC calibrated to a target signal-to-noise ratio.

## The predictor–corrector loop

[`generate.predictor_corrector_sampling`](../generate.py#L33). One sample proceeds as:

```
draw x from prior at t = 1
for i in 0..steps-1:
    t = 1 - i/steps
    score = model(x, side_info, t)
    x <- predictor_step(x, score, t)          # reverse Euler
    repeat n_corr times:
        x <- corrector_step(x, model, t)      # Langevin with SNR
return x                                       # approximately p_0
```

### Initialization (`t = 1`)

[`generate.py:57-66`](../generate.py#L57):

```python
eps = torch.randn_like(x_clean)
if sde == "VE":
    x = eps * math.sqrt(var_1)                        # ~ N(0, V(1) · I)
elif sde == "VP":
    gamma = math.exp(-0.5 * var_1)
    sigma_square = 1 - gamma ** 2
    x = eps * math.sqrt(sigma_square)                 # ~ N(0, (1 - γ²) I)
```

These match the forward process's marginal at `t = 1`, so the sampler starts from (approximately) the correct prior.

There is also [`generate.sample_init_bs`](../generate.py#L13) for a GBM-style prior (`exp(√V(1) · ε)` then log) — but `predictor_corrector_sampling` does **not** call it. It's available if you want a Black-Scholes-style initial condition for α=1 experiments; the default code path uses the plain Gaussian prior above.

### Predictor — reverse Euler

[`generate.py:82-87`](../generate.py#L82):

```python
s_t = sigma(t)
if sde == "VE":
    drift = -(s_t**2) * score * dt
else:  # VP
    drift = -0.5 * (s_t**2) * dt - (s_t**2) * score * dt
diffusion = s_t * sqrt(dt) * z      # z ~ N(0, I)
x = x + drift + diffusion
```

Derivations:

- **VE forward:** `dx = σ(t) dW`. Reverse-time: `dx = −σ²(t) ∇log p_t(x) dt + σ(t) dW̃`. The drift is `−σ² · score · dt`.
- **VP forward:** `dx = −½ σ²(t) x dt + σ(t) dW`. Reverse-time: `dx = [½ σ²(t) x − σ²(t) ∇log p_t(x)] · (−dt) + σ(t) dW̃` when stepping `t` backward. The code writes this as `drift = −½ σ² dt − σ² · score · dt` — the `−½ σ² dt` term is the reverse-time-adjusted forward drift. (In a careful derivation there should be an `x` factor on that first term; here the code approximates by omitting it — a mild simplification that only matters when `V(t)` is large.)

### Corrector — Langevin with SNR

[`generate.py:89-99`](../generate.py#L89):

```python
for _ in range(n_corr):
    grad = model(...)                          # fresh score at same t
    z = torch.randn_like(x)
    grad_norm = sqrt(mean(grad²))              # per-sample norms
    z_norm    = sqrt(mean(z²))
    eps = 2 * (snr * z_norm / grad_norm)**2    # Langevin step
    x = x + eps * grad + sqrt(2 * eps) * z
```

The step size is chosen so that the **signal-to-noise ratio** `‖eps · grad‖ / ‖√(2 eps) · z‖ ≈ snr`. This is the heuristic from Song & Ermon — it prevents over/undershooting when the score scale varies wildly across `t`.

`n_corr` controls how many Langevin steps to take per predictor step. `snr=0.2, n_corr=1` are the codebase defaults.

## CLI entry point — `generate.py`

```bash
python generate.py \
    --model_path save_model_*/model_epoch_N.pth \
    --processed_file data/financial_test_data.pt \
    --out_file denoised_financial_test_data.pt
```

The [`main`](../generate.py#L111) in `generate.py`:

1. Builds `diff_CSDI` from a **hard-coded config** (`channels=64, diffusion_embedding_dim=128, emb_feature_dim=16`).
2. Loads the checkpoint.
3. Loads `data/financial_test_data.pt` — a separate tensor-style dataset with `{data, timepoints, labels}`. (This file is **not** produced by `data_download.py` — it's expected to come from some external prep step.)
4. For each batch, runs the sampler 20 times and averages the 20 samples to denoise the batch.
5. Writes `{denoised, labels}` to `--out_file`.

This is framed as a **denoising** entry point rather than a pure-sampling one: it uses real `x_clean` batches (not prior-drawn) and averages multiple draws. If you want unconditional synthesis, `plot_result.py` is the more natural entry.

## CLI entry point — `plot_result.py`

[`plot_result.main`](../plot_result.py#L236). Unlike `generate.py`, this script generates synthetic samples **from scratch** (no reference data) and computes stylized-fact metrics on them.

Per-sample loop ([`plot_result.py:277-313`](../plot_result.py#L277)):

1. Build a throwaway `x_clean_dummy = randn(1, 1, 2048)` — only its shape is used (see note below about why).
2. Use `dummy_time = linspace(0, 1, 2048)` as the temporal axis.
3. Call `predictor_corrector_sampling(..., num_samples=1)` with the configured SDE/schedule.
4. Post-process based on `alpha`:
   - **α = 0:** `sample` lives in standardized log-return space → `scaler.inverse_transform` → save as `returns_sample_*.csv`.
   - **α = 1:** `sample` lives in standardized log-price space → `scaler.inverse_transform` → `np.diff` to get log-returns → pad a leading 0 → save as `log_returns_sample_*.csv`.
5. After all samples are generated: fit a power law on `|returns|` (prints `α, xmin`), compute metrics, plot stylized facts, and save time-series plots.

### Why `x_clean_dummy`?

The sampler signature requires `x_clean` because of the CSDI conditional branch (where the observation is concatenated with the noisy target). In unconditional mode (`is_unconditional=True`) only `x_clean.shape` is consulted — to size `target_dim`, build side info, and initialize `x`. The values of `x_clean_dummy` never feed the network. It's purely a shape carrier.

## α modes

The α flag is **not** a model hyperparameter — it selects which forward-process *interpretation* the data is trained against. See [`docs/sde-and-schedules.md`](sde-and-schedules.md#gbm-sde---alpha-1-the-papers-contribution) for why GBM-in-price-space reduces to a VE SDE in log-coordinates and is therefore a pure data-representation switch from the network's point of view.

| Choice point | α = 0 (additive baselines) | **α = 1 (GBM, paper's contribution)** |
|--------------|----------------------------|----------------------------------------|
| Training shard | `data/sp500_subseq.pt` | `data/sp500_subseq_log.pt` |
| Training scaler | `global_scaler.pkl` (log-returns) | `global_log_scaler.pkl` (log-prices) |
| `--sde` to pair with | `VE` or `VP` (your ablation choice) | `VE` (forward process reduces to VE in log-space; see paper §3) |
| What the model learns | Distribution over standardized log-returns | Distribution over standardized log-prices |
| Implied price-space dynamics | Additive Gaussian noise | Multiplicative (GBM-like) — heteroskedasticity |
| `plot_result.py` post-processing | inverse-transform → save returns | inverse-transform → `np.diff` → save log-returns |
| Prior used at `t = 1` | Plain Gaussian (VE/VP) | Plain Gaussian (VE/VP) — the GBM prior in `sample_init_bs` is available but not wired in |
| Stylized-fact fidelity (paper §4) | Light tails, weak/absent leverage effect | Best heavy-tail / volatility / leverage match |

The loss ([`losses.py`](../losses.py)) and the network ([`networks.py`](../networks.py)) are α-agnostic.

## Shared model config

`generate.py` and `plot_result.py` both import the architecture from [`model_config.py`](../model_config.py), the same module `train.py` uses. Checkpoints are portable across the three without per-script edits. See [architecture.md](architecture.md#configured-sizes-across-the-codebase).

## Practical knobs

| Flag | Effect | Typical |
|------|--------|---------|
| `--steps` | Reverse-diffusion steps. More steps → more accurate, slower. Paper §4 uses **2000**. | 1000–2000 |
| `--snr` | Langevin SNR target. Too high → noise dominates; too low → score dominates. | 0.15–0.25 |
| `--n_corr` | Langevin sub-steps per predictor step. 0 = pure predictor; 1–2 stabilizes. | 1 |
| `--sigma_min` / `--sigma_max` | **Must match training.** Defines the noise schedule endpoints. | Same as `train.py` |
| `--noise_schedule` | **Must match training.** | Same as `train.py` |

Mismatching the training-time `sigma_min/max` or `noise_schedule` will silently produce garbage — the model's score estimates are valid only on the noise process it was trained against.
