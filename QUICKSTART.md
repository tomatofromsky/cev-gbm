# Quickstart

End-to-end recipe to get from a fresh clone to a generated set of synthetic S&P 500 return trajectories with stylized-fact plots. The recommended path is the `run_experiment.sh` driver — one command runs every stage. Manual per-stage invocations are documented at the bottom for users who want finer control. For the *why* behind each step, see [`docs/`](docs/README.md).

## 0. Prerequisites

- **Python ≥ 3.9** (3.12 tested).
- **CUDA GPU** for training. CPU works for `generate.py` / `plot_result.py` at small `--num_samples` but is slow.
- **~5 GB disk** for the data shards and a few model checkpoints.
- Network access to Wikipedia and Yahoo Finance for the one-time data download.

## 1. Install

```bash
git clone <this-repo> gbm
cd gbm
python -m venv .venv && source .venv/bin/activate
pip install -e .
# or: pip install -r requirements.txt
```

For the optional efficient-attention path (only if you set `is_linear=True` in the config):

```bash
pip install -e '.[efficient-attention]'
```

## 2. Run an experiment end-to-end

`run_experiment.sh` (bash) is the driver. It picks one of three targets, runs every stage, is idempotent (safe to re-run), and prints a final summary pointing at the output folders.

```bash
./run_experiment.sh                  # default = TARGET=gbm (paper headline)
TARGET=vp  ./run_experiment.sh       # additive VP baseline
TARGET=ve  ./run_experiment.sh       # additive VE baseline
```

A PowerShell mirror is also provided as `run_experiment.ps1` (Windows or `pwsh` on any platform). Same behavior, PowerShell-native parameter syntax (with env-var fallback for cross-shell consistency):

```powershell
./run_experiment.ps1                            # default = TARGET=gbm
./run_experiment.ps1 -Target vp                 # additive VP baseline
./run_experiment.ps1 -Target ve                 # additive VE baseline
$env:TARGET = 've'; ./run_experiment.ps1        # env-var style also works
```

The remainder of this section uses bash syntax; the PowerShell flags map straightforwardly (`-Target`, `-BatchSize`, `-Epochs`, `-Steps`, `-NumSamples`, `-Schedule`, `-Lr`, `-Snr`, `-NCorr`, `-SigmaMin`, `-SigmaMax`, `-CudaVisibleDevices`).

| `TARGET` | What it trains | Auto-derived `model_dir` | Default schedule |
|---|---|---|---|
| `gbm` (default) | GBM forward process (`--alpha 1`, log-prices) — paper headline | `save_model_bs_cosine_64/` | `cosine` |
| `vp` | VP SDE on log-returns (`--alpha 0`) | `save_model_vp_cosine_64/` | `cosine` |
| `ve` | VE SDE on log-returns (`--alpha 0`) | `save_model_ve_exponential_64/` | `exponential` |

### Pipeline stages the script runs

1. **Data prep** — runs `python data_download.py` and moves the resulting `.pt`/`.pkl` files into `data/`. Skipped if `data/sp500_subseq*.pt` already exist (≈ 10 min the first time, instant after).
2. **Training** — `python train.py` for single-GPU (default), or `torchrun --nproc_per_node=N train.py` for DDP (when `-g N` / `NUM_GPUS=N` with N > 1). Auto-resumes from the highest-numbered checkpoint under the auto-derived `model_dir`. Ctrl+C between epochs is safe (in-progress epoch is discarded; previous checkpoint is intact). Effective batch under DDP is `--batch_size × N`.
3. **Sampling + stylized-fact plots** — `python plot_result.py` against the latest checkpoint, with matching SDE/schedule flags. Writes `<TAG>_csv_/`, `<TAG>_plot_/`, `<TAG>_timeseries/`.
4. **Real-data baseline** — `python ori_plot_result.py` once, writes `ori_plot/`. Skipped on subsequent runs.
5. **Final summary** — prints which folders to compare side by side.

### Knob overrides (env vars)

Every knob has a sensible default — overrides are only needed for non-paper variants or smoke tests.

| Env var | Default | Meaning |
|---|---|---|
| `TARGET` | `gbm` | `gbm` / `vp` / `ve` |
| `ALPHA`, `SDE`, `SCHEDULE` | derived from `TARGET` | Override individually if you want, e.g., `TARGET=gbm SCHEDULE=exponential` |
| `BATCH_SIZE` | 64 | Lower (16, 8) if you hit `CUDNN_STATUS_NOT_INITIALIZED` (GPU OOM at workspace allocation) |
| `EPOCHS` | 1000 | |
| `LR` | 1e-4 | Adam |
| `SIGMA_MIN`, `SIGMA_MAX` | 0.01, 1.0 | Noise schedule endpoints |
| `NUM_SAMPLES` | 120 | Synthetic sequences for evaluation |
| `STEPS` | 2000 | Reverse-diffusion steps |
| `SNR`, `N_CORR` | 0.2, 1 | Langevin corrector parameters |
| `CUDA_VISIBLE_DEVICES` | 0 | Pin to a specific GPU (or empty string for CPU) |

### First-class CLI flags

Three knobs accept both a CLI flag and an env var. The flag wins if both are set; env var wins over the built-in default.

| Knob | Short flag | Long flag | Env var | Default |
|------|-----------|-----------|---------|--------:|
| Training epochs | `-e N` | `--epochs N` | `EPOCHS` | 1000 |
| Training batch size (per rank) | `-b N` | `--batch_size N` (or `--batch-size N`) | `BATCH_SIZE` | 64 |
| Number of GPUs | `-g N` | `--gpus N` | `NUM_GPUS` | 1 |

```bash
./run_experiment.sh -e 2                       # 2 epochs
./run_experiment.sh -b 16                      # batch 16 per rank (helps with VRAM)
./run_experiment.sh -g 4                       # 4-GPU DDP via torchrun
./run_experiment.sh -e 200 --batch_size 32 --gpus 2
EPOCHS=42 BATCH_SIZE=16 NUM_GPUS=4 ./run_experiment.sh    # env-var form (still works)
```

Use `./run_experiment.sh -h` to print the flag-help. All other knobs remain env-var only.

**Multi-GPU note**: when `--gpus N > 1`, the script invokes `torchrun --standalone --nnodes=1 --nproc_per_node=N train.py …` instead of plain `python train.py …`. The training script auto-detects `LOCAL_RANK`/`RANK`/`WORLD_SIZE` and switches into DDP mode (DistributedSampler + DistributedDataParallel + per-rank logging suppression on non-rank-0). Effective batch is `--batch_size × N`. Sampling (`plot_result.py`) is always single-GPU regardless of `--gpus`.

### Quick smoke (verify the pipeline in minutes)

```bash
./run_experiment.sh -e 2 -b 16                                      # via flags
EPOCHS=2 BATCH_SIZE=16 STEPS=100 NUM_SAMPLES=5 ./run_experiment.sh  # all knobs via env
```
```powershell
./run_experiment.ps1 -Epochs 2 -BatchSize 16 -Steps 100 -NumSamples 5
```

Trains for 2 epochs, samples 5 sequences with 100 reverse steps. Fast and confirms every stage works end-to-end. The resulting plots aren't meaningful but the pipeline is verified.

## 3. Outputs

After `./run_experiment.sh` completes (with default `TARGET=gbm`, `SCHEDULE=exponential`):

| Path | Contents |
|------|----------|
| `data/sp500_subseq*.pt`, `data/global_*.pkl` | Preprocessed shards + scalers (created once, reused) |
| `save_model_bs_exponential_64/model_epoch_*.pth` | Per-epoch training checkpoint (only the latest is kept) |
| `BS_exponential_csv_/log_returns_sample_*.csv` | One CSV per generated sequence (column `log_return`; for VE/VP targets it's `return`) |
| `BS_exponential_plot_/generated_metrics_*.png` | Six stylized-fact plots on synthetic samples |
| `BS_exponential_timeseries/time_series_*.png` | Recovered price-series plots |
| `ori_plot/generated_metrics_*.png` | Same six plots, computed on real S&P 500 data — the baseline |

Stdout also prints `Powerlaw MLE α = ...`. The paper reports α = 4.62 for GBM + exponential (closest to empirical α = 4.35), 3.78 for GBM + cosine.

For other targets/schedules, the `BS_exponential` prefix becomes `BS_<schedule>` for GBM, `VP_<schedule>` for `TARGET=vp`, or `VE_<schedule>` for `TARGET=ve`.

## 4. Sanity checks

After the script completes, before declaring success:

1. `ls save_model_*/model_epoch_*.pth | wc -l` — non-zero. Training actually saved a checkpoint.
2. `ls BS_exponential_csv_ | wc -l` — equals `NUM_SAMPLES`. Generation completed.
3. Open `BS_exponential_plot_/generated_metrics_linear_unpredictability.png`. The scatter should be near zero across all lags. If it shows strong autocorrelation at lag 1, the sampler likely diverged (try `STEPS=4000`).
4. Open `BS_exponential_plot_/generated_metrics_volatility_clustering.png`. On log-log it should look like a slowly decaying line, not flat noise.
5. The printed `Powerlaw MLE α` should land in roughly `3 – 5`. Empirical S&P 500 is α=4.35 (paper §4); GBM-exponential reproduces ~4.62 in the paper.

## 5. Re-plot from existing CSVs (skip resampling)

After step 2 you can iterate on plot styling without resampling:

```bash
python csv_to_plot_result.py
# defaults to: csv_folder=BS_exponential_csv_0623, plot_folder=BS_exponential_plot_0623
```

Edit the defaults at the top of [`csv_to_plot_result.py`](csv_to_plot_result.py) or pass arguments to point it at your folder.

## Common pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| `RuntimeError: size mismatch` on `load_state_dict` | `model_config.py` was edited after training | Either retrain or revert `MODEL_CONFIG` to match the saved checkpoint |
| Generated `*.csv` directories empty after run | Sampler crashed mid-loop | Check `STEPS` is an int, `NUM_SAMPLES` ≥ 1, model loaded |
| `FileNotFoundError: data/sp500_subseq*.pt` outside the script | Manual `data_download.py` invocation didn't move outputs | The script handles this automatically; if invoking by hand, `mv sp500_subseq*.pt global_*.pkl data/` |
| `--is_unconditional False` ignored | `argparse` parses to truthy string `"False"` | Don't pass the flag; default is `True`. To train conditional, edit `train.py:177` default to `False` |
| `cuDNN error: CUDNN_STATUS_NOT_INITIALIZED` | Usually GPU OOM at workspace allocation, not a real cuDNN bug | Lower `BATCH_SIZE` (try 16 or 8); free up the GPU; check `nvidia-smi` for other processes |
| Stylized-fact plots look Gaussian (no heavy tails) | Undertrained, or schedule mismatch between train/sample | More epochs; verify `SCHEDULE` and `SIGMA_MIN/MAX` match between training and sampling — if you used the script for both, they automatically do |
| `SSL: CERTIFICATE_VERIFY_FAILED` from `data_download.py` | Python's stdlib `ssl` module has no usable CA bundle (common with uv-managed Python) | The current `data_download.py` already routes through `requests` + `certifi` — re-pull if you have an old version |

## Manual invocations (without the script)

For debugging or custom pipelines, here's what `run_experiment.sh` boils down to. The flags shown are the paper-recipe defaults for `TARGET=gbm`.

### Data prep (one-time)

```bash
mkdir -p data
python data_download.py
mv sp500_subseq*.pt global_*.pkl data/
```

### Train (GBM target — paper headline)

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --alpha 1 --sde VE --noise_schedule exponential \
    --batch_size 64 --epochs 1000 --lr 1e-4 \
    --sigma_min 0.01 --sigma_max 1.0
```

`--sde VE` matches the paper's derivation: with drift `μ_t = ½σ_t²` the GBM forward process reduces to a VE SDE in log-price space, so the matching DSM-loss branch is VE. `--noise_schedule exponential` is the paper's best heavy-tail fit (α≈4.62 vs. empirical 4.35); `cosine` and `linear` are also valid.

The auto-selected `model_dir` per `(alpha, sde, schedule)` — the schedule is baked in so different schedules don't collide:

| `--alpha` | `--sde` | `--noise_schedule` | model dir |
|---|---|---|---|
| 1 | VE | `exponential` | `save_model_bs_exponential_64` (GBM — paper headline) |
| 1 | VE | `cosine` | `save_model_bs_cosine_64` |
| 0 | VP | `cosine` | `save_model_vp_cosine_64` |
| 0 | VE | `exponential` | `save_model_ve_exponential_64` |

(General pattern: `save_model_{bs|vp|ve}_{schedule}_64`.)

### Sample + stylized-fact plots

```bash
CUDA_VISIBLE_DEVICES=0 python plot_result.py \
    --model_path save_model_bs_exponential_64/model_epoch_1000.pth \
    --alpha 1 --sde VE --noise_schedule exponential \
    --sigma_min 0.01 --sigma_max 1.0 \
    --num_samples 120 --steps 2000 --snr 0.2 --n_corr 1 \
    --plot_folder BS_exponential_plot_ --csv_folder BS_exponential_csv_ \
    --plot_timeseries BS_exponential_timeseries
```

`--alpha`, `--sde`, `--noise_schedule`, `--sigma_min`, `--sigma_max` **must match** training, or the score is out-of-distribution.

### Real-data baseline (one-time)

```bash
python ori_plot_result.py \
    --data_file data/sp500_subseq_original.pt \
    --num_samples 1000
```

Writes the six baseline plots to `ori_plot/`. Place them next to `BS_exponential_plot_/` images for visual comparison.

## What to read next

- [arXiv:2507.19003](https://arxiv.org/abs/2507.19003) — the paper this codebase implements.
- [`docs/data-pipeline.md`](docs/data-pipeline.md) — what's actually in those `.pt` files.
- [`docs/sde-and-schedules.md`](docs/sde-and-schedules.md) — VE / VP / GBM SDEs and schedule shapes.
- [`docs/architecture.md`](docs/architecture.md) — model internals + the paper's architecture ablation.
- [`docs/training.md`](docs/training.md) — single-GPU + DDP loop, all CLI flags, presets.
- [`docs/sampling.md`](docs/sampling.md) — the predictor–corrector loop and α-mode post-processing.
