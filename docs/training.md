# Training

Source: [`train.py`](../train.py).

Training auto-detects whether it was launched plain (`python train.py …`) or under `torchrun` (`torchrun --nproc_per_node=N train.py …`). A checkpoint is written at the end of each epoch (only the latest is kept); `load_checkpoint` auto-resumes from the highest-numbered epoch on restart. Pin to a specific device or device set with `CUDA_VISIBLE_DEVICES=<idx>` / `CUDA_VISIBLE_DEVICES=0,1,2,3`.

The reference paper's training recipe (§4): **batch size 64, 1000 epochs, σ_min=0.01, σ_max=1.0, sequence length L=2048**. The defaults in `train.py` match those numbers. Note paper's batch is **per rank** under DDP. The recommended SDE choice is `--alpha 1` (GBM) with cosine or exponential schedule — see the [paper](https://arxiv.org/abs/2507.19003).

## Entry points

```bash
# Single-GPU (or CPU fallback)
python train.py [--options]

# Multi-GPU DDP via torchrun (single node, N processes)
torchrun --standalone --nnodes=1 --nproc_per_node=N train.py [--options]
```

`run_training` parses the CLI, picks a `(model_dir, data_file)` preset based on `(alpha, sde)`, makes the output directory, then calls `train(args)`. `train` checks for the `LOCAL_RANK` env var to decide which branch to take; everything downstream is driven from that boolean.

## Training loop

`train` does, in order:

1. **DDP detection.** If `LOCAL_RANK` in env: `dist.init_process_group(backend='nccl')`, set `device = cuda:LOCAL_RANK`, record `world_size = dist.get_world_size()`. Otherwise: `device = cuda:0` (or CPU), `world_size = 1`.
2. **Dataset.** `PreprocessedFinancialDataset` loads the full `.pt` into memory. Each `__getitem__` returns `(data[i].unsqueeze(0), timepoints[i])` — shape `(1, 2048)` and `(2048,)`.
3. **DataLoader / sampler.** Under DDP: `DistributedSampler(dataset, shuffle=True)` shards the dataset across ranks. Otherwise: plain `DataLoader(shuffle=True)`. `num_workers=0` in both cases.
4. **Model.** `diff_CSDI(config, input_dim)` with `input_dim = 1` (unconditional) or `2`. Then `model.apply(initialize_weights)` — Xavier-uniform on Linear/MHA, ones/zeros on LayerNorm. (The zero-init on `output_projection2` survives because `initialize_weights` doesn't touch `Conv1d`.)
5. **Optimizer.** `Adam(lr=args.lr)`. No weight decay, no scheduler.
6. **Resume — before the DDP wrap.** `load_checkpoint` scans `model_dir/model_epoch_*.pth`, sorts by epoch number (taking the suffix after the last `_`), loads the highest into `model` and `optimizer`, and returns `start_epoch = ckpt['epoch'] + 1`. Doing this before wrapping keeps the load path identical for single-process and DDP modes.
7. **DDP wrap.** Under DDP: `model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)`. The wrap broadcasts rank-0's weights to all ranks (so even if some rank's checkpoint load disagreed, weights converge).
8. **Epoch loop.** For each epoch:
   - Under DDP: `sampler.set_epoch(epoch)` so each rank shuffles consistently.
   - Forward + backward via `denoising_score_matching_loss`. See [sde-and-schedules.md](sde-and-schedules.md#denoising-score-matching-loss).
   - Under DDP: `dist.all_reduce(total_loss, ReduceOp.SUM)` then divide by `world_size × len(dataloader)` for the logged average. Otherwise: just `total_loss / len(dataloader)`.
   - **On rank 0 only**: print the epoch loss, save the new checkpoint, and delete all older `model_epoch_*.pth` files in the dir (keeps only the latest). The saved `state_dict` is always the **unwrapped** form (`model.module.state_dict()` when DDP-wrapped, `model.state_dict()` otherwise) — so checkpoints stay portable between modes and load fine into the bare-model `plot_result.py` / `generate.py`.
9. **SIGINT.** A module-level handler converts Ctrl+C into `KeyboardInterrupt`, which the epoch loop catches. **The in-progress epoch is discarded** — no partial checkpoint. Next run resumes from the last fully-completed epoch.
10. **Cleanup.** Under DDP: `dist.destroy_process_group()`. Otherwise: nothing.

`tqdm` and all `print` calls are guarded by `is_main_process()` so non-rank-0 workers stay silent.

## Checkpoint portability

Because `train.py` always saves the unwrapped `state_dict`, every produced checkpoint is loadable by:

- `python train.py …` (single-GPU, resume) — loads pre-DDP-wrap.
- `torchrun … train.py …` (DDP, resume) — loads pre-DDP-wrap on each rank.
- `python plot_result.py …` (sampling) — loads into the bare `diff_CSDI`.
- `python generate.py …` (sampling) — same.

You can train on N GPUs and sample on 1, or train on 1 and resume on N — no checkpoint conversion needed.

## CLI arguments

| Flag | Default | Effect |
|------|---------|--------|
| `--data_file` | `data/sp500_subseq_log.pt` | Argparse default; overwritten by the preset below based on `(alpha, sde, noise_schedule)` |
| `--model_dir` | `save_model_bs_exponential_64` | Argparse default; overwritten by the preset below |
| `--batch_size` | 64 | Per-rank batch size; effective batch under DDP is `batch_size × world_size`. Lower (16, 8) if you hit `CUDNN_STATUS_NOT_INITIALIZED` — that's usually GPU OOM at workspace allocation, not a real cuDNN bug |
| `--epochs` | 1000 | |
| `--lr` | 1e-4 | Adam |
| `--sde` | `VP` | `VE` or `VP` — see [sde-and-schedules.md](sde-and-schedules.md) |
| `--alpha` | 0 | `0` → return-space training; `1` → log-price-space. Loss doesn't actually read this, it only selects the shard |
| `--sigma_min` | 0.01 | Noise schedule endpoint |
| `--sigma_max` | 1.0 | Noise schedule endpoint |
| `--emb_time_dim` | 128 | Temporal embedding width; fed to `get_side_info` |
| `--emb_feature_dim` | 16 | **Declared but unused** — `MODEL_CONFIG["emb_feature_dim"]` (currently 64) takes precedence |
| `--noise_schedule` | `exponential` | `cosine`, `linear`, or `exponential`. Paper's best heavy-tail fit. The `run_experiment.sh` / `.ps1` wrappers also default `TARGET=gbm` and `TARGET=ve` to `exponential`, and `TARGET=vp` to `cosine`. |
| `--is_unconditional` | `True` | **String-typed — see caveat** |

### Presets

The argparse pass overrides `model_dir` and `data_file` based on `(alpha, sde, noise_schedule)`. The schedule is baked into the dir name so multiple schedules of the same SDE family don't collide and auto-resume picks up the matching checkpoint:

| `alpha` | `sde` | `model_dir` (template) | `data_file` | Paper terminology |
|---------|-------|------------------------|-------------|-------------------|
| 1 | any | `save_model_bs_<schedule>_64` | `data/sp500_subseq_log.pt` | **GBM SDE** (the paper's contribution; "bs" = Black–Scholes) |
| 0 | VE | `save_model_ve_<schedule>_64` | `data/sp500_subseq.pt` | VE SDE baseline on log-returns |
| 0 | VP | `save_model_vp_<schedule>_64` | `data/sp500_subseq.pt` | VP SDE baseline on log-returns |
| other | — | `save_model_cev` | (unchanged from CLI) | (unused) |

Concrete examples:

| Invocation | Resolved `model_dir` |
|------------|----------------------|
| `--alpha 1 --noise_schedule exponential` | `save_model_bs_exponential_64` (GBM, paper headline) |
| `--alpha 1 --noise_schedule cosine` | `save_model_bs_cosine_64` |
| `--alpha 0 --sde VE --noise_schedule exponential` | `save_model_ve_exponential_64` |
| `--alpha 0 --sde VP --noise_schedule cosine` | `save_model_vp_cosine_64` |

The `_64` suffix encodes `emb_feature_dim` from the paper's architecture ablation (§4.1) — the chosen final value. Earlier states of the codebase used `_16` and `_32` for the smaller ablation variants; see [`docs/architecture.md`](architecture.md#architecture-ablation-in-the-paper-41) and the [paper](https://arxiv.org/abs/2507.19003) §4.1.

### Model config

`train` does `config = dict(MODEL_CONFIG)` and overwrites `side_dim = emb_time_dim + emb_feature_dim`. `MODEL_CONFIG` lives in [`model_config.py`](../model_config.py) and is shared with `generate.py` and `plot_result.py`:

```python
MODEL_CONFIG = {
    "channels": 128,
    "diffusion_embedding_dim": 256,
    "target_dim": 1,
    "emb_feature_dim": 64,
    "side_dim": 2,            # overwritten at runtime
    "nheads": 8,
    "is_linear": False,
    "layers": 4,
}
```

The only CLI-tunable part of the architecture is `emb_time_dim`; everything else requires editing `model_config.py`. Since the same module is imported by all three scripts, checkpoints stay portable across training and inference.

## `is_unconditional` pitfall

[`train.py:177`](../train.py#L177):

```python
parser.add_argument('--is_unconditional', type=str, default=True)
```

- Default is the Python bool `True` (truthy).
- But `type=str` means any CLI-supplied value becomes a string. `--is_unconditional False` is the string `"False"` — **truthy**, so the model still runs unconditionally.

Don't pass this flag. If you need conditional training, edit the default to `False` (or fix the parse with `type=lambda s: s.lower() == 'true'`).

## Artifacts written

Under `args.model_dir`:

- `model_epoch_1.pth`, `model_epoch_2.pth`, … — one per epoch.
- `training_log.csv` — declared, but `save_log` / `load_logs` are never actually called from the training loop. The CSV will not be written in current code.

## Memory notes

- `num_workers=0` — data is held entirely in CPU RAM after the initial `torch.load`; no forking.
- Each ticker's ~100 windows of 2048 floats is ~800 KB. A few hundred tickers → ~100 MB total.
- Model + Adam state at `channels=128, layers=4` is small (< 50 MB); the attention activations are the dominant GPU footprint at `L=2048, batch=64`.
- If you hit `cuDNN error: CUDNN_STATUS_NOT_INITIALIZED` on the first batch, it's almost always cuDNN failing to allocate its workspace because the GPU is contended or `--batch_size` is too large. Drop to `--batch_size 16` or smaller.

## Restart semantics

- Killing training mid-epoch (Ctrl+C or process kill) leaves the last completed epoch's checkpoint intact. Re-run the same `python train.py ...` command and it picks up at `epoch + 1`.
- Optimizer state (Adam moments) is also restored — resumption is truly continuous.
- If you change `config` between runs, `load_state_dict` will error and abort before training starts — safer than silently retraining from scratch.
