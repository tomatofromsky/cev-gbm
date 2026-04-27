# Documentation

Companion docs for the GBM project. The top-level [`README.md`](../README.md) is the orientation; these pages go deeper into each subsystem. The underlying paper is [arXiv:2507.19003](https://arxiv.org/abs/2507.19003).

| Topic | File | What it covers |
|-------|------|----------------|
| Data pipeline | [data-pipeline.md](data-pipeline.md) | `data_download.py` — ticker fetch, sliding windows, global scalers, the three `.pt` shards |
| Model architecture | [architecture.md](architecture.md) | `networks.py` — `diff_CSDI`, `ResidualBlock`, diffusion/positional embeddings, side-info conditioning, paper's architecture ablation |
| SDEs & schedules | [sde-and-schedules.md](sde-and-schedules.md) | `utils.py` + `losses.py` — VE / VP / GBM forward dynamics, the three noise schedules, closed-form `var_integral`, denoising score matching |
| Training | [training.md](training.md) | `train.py` — single-GPU + DDP (auto-detected), checkpointing, auto-selected presets, CLI arguments |
| Sampling | [sampling.md](sampling.md) | `generate.py` — predictor–corrector reverse SDE integration, GBM prior init, α=1 post-processing |

## Reading order

If you're new to the code, read in roughly this order:

1. **data-pipeline** — you need to know what `x` is before any of the model makes sense.
2. **sde-and-schedules** — grounds the training loss and sampler in the underlying SDE math; the GBM section is the paper's contribution.
3. **architecture** — once you know the I/O and conditioning, the `ResidualBlock` falls out.
4. **training** — end-to-end training recipe.
5. **sampling** — generation procedure; mostly the reverse of training.

## Cross-cutting caveats

A few surprises appear in multiple files — flagged once here, referenced in context below.

- **Shared model config.** `train.py`, `generate.py`, and `plot_result.py` all import the canonical architecture from [`model_config.py`](../model_config.py). Edit it once to change widths/depths; checkpoints stay portable across scripts. See [architecture.md](architecture.md#configured-sizes-across-the-codebase).
- **`--alpha` is mostly a path selector.** The argument is declared in every script but the loss is α-agnostic; α only controls (a) which preprocessed shard to train on, (b) which scaler to inverse-transform with, and (c) whether to differentiate log-prices into log-returns after sampling. See [sampling.md](sampling.md#alpha-modes).
- **`K = 1` feature dim.** The model is written as if it can handle multivariate series (`K > 1`), but every caller uses `target_dim = 1`. The feature-attention path (`forward_feature`) references an undefined `self.feature_layer` and would crash if `K > 1`. See [architecture.md](architecture.md#feature-attention-is-a-dead-branch).
- **`is_unconditional` parses as a string.** `argparse` uses `type=str, default=True`; passing `--is_unconditional False` gives the truthy string `"False"`. Rely on the default. See [training.md](training.md#isunconditional-pitfall).
