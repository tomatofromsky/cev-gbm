# Model architecture

Source: [`networks.py`](../networks.py).

The score network is a stack of `ResidualBlock`s operating on a `(channels, K, L)` tensor, conditioned on (i) a diffusion-time embedding and (ii) "side info" (temporal + feature embeddings). This is the CSDI (Conditional Score-based Diffusion for Imputation) structure; here `K = 1` because the data is univariate.

```
  x: (B, input_dim, K, L)           diffusion step t: (B,)       cond_info: (B, side_dim, K, L)
       │                                    │                            │
       ▼                                    ▼                            │
  input_projection (Conv1d)         DiffusionEmbedding                   │
       │                                    │                            │
       ▼                                    │                            │
  ResidualBlock ◄──── diffusion_emb ◄───────┘                            │
       │  ▲                                                              │
       │  └────────────────────── cond_info ◄─────────────────────────────┘
       ▼
  (...repeat `layers` times, collect skip connections)
       │
       ▼
  sum(skips)/√L → output_projection1 → ReLU → output_projection2 → score(B, K, L)
```

## Top-level module — `diff_CSDI`

[`networks.diff_CSDI`](../networks.py#L80). Config dict consumed at construction:

| Key | Meaning |
|-----|---------|
| `channels` | Internal feature width |
| `diffusion_embedding_dim` | Width of the sinusoidal+MLP embedding of `t` |
| `target_dim` | Number of features `K`. Always 1 in practice |
| `emb_feature_dim` | Width of the per-feature learnable embedding |
| `side_dim` | Input width of the conditioning info. Set by the caller to `emb_time_dim + emb_feature_dim` |
| `nheads` | Attention heads per residual block |
| `is_linear` | If true, use `LinearAttentionTransformer` instead of `nn.TransformerEncoder` |
| `layers` | Number of residual blocks |

The model takes `inputdim = 1` (unconditional) or `2` (conditional: observation + masked target).

### Input packing

[`utils.set_input_to_diffmodel`](../utils.py#L89):

```
unconditional:   (B, 1, K, L) = x_noisy
conditional:     (B, 2, K, L) = [x_clean ; 0 · x_noisy]
```

The conditional branch zeroes out `x_noisy` and passes `x_clean` as the observed-value channel — a quirk inherited from the CSDI paper's imputation setup. This project trains unconditionally by default.

### Forward pass — `networks.py:112-132`

1. Flatten `(B, input_dim, K, L) → (B, input_dim, K·L)`, apply `input_projection` (Conv1d 1×1, Kaiming init), reshape back to `(B, channels, K, L)`.
2. Compute `diffusion_emb = DiffusionEmbedding(t)` once per batch.
3. Pass through `residual_layers`, collecting **skip connections** from each block.
4. Aggregate skips: `sum(skips) / √len(layers)` — a normalization trick from WaveNet/CSDI.
5. Two-layer Conv1d head: `output_projection1 → ReLU → output_projection2`.
6. Output shape `(B, K, L)` — the predicted score.

### Zero-init head

[`networks.py:92`](../networks.py#L92):

```python
nn.init.zeros_(self.output_projection2.weight)
```

At initialization the model predicts exactly zero score. This stabilizes early training: the first few gradient steps don't have to overcome a random garbage prediction.

## Diffusion time embedding

[`networks.DiffusionEmbedding`](../networks.py#L32). The usual sinusoidal positional encoding adapted to a **continuous** scalar `t ∈ [0, 1]`:

1. Geometric progression of frequencies: `exp(arange(half_dim) · (−log(10000)/(half_dim − 1)))`.
2. Concatenate `sin, cos` → shape `(B, embedding_dim)`.
3. Two-layer MLP with SiLU activations → `(B, projection_dim)`.

Each `ResidualBlock` further projects this down to `channels` and broadcasts along `K·L`.

## Side information

[`utils.get_side_info`](../utils.py#L74) builds `cond_info ∈ R^{B × side_dim × K × L}`:

- `time_embed`: sinusoidal encoding of the normalized-time values `x_tp ∈ [0,1]^L` from the data shard, `emb_time_dim` wide.
- `feature_embed`: learnable `nn.Embedding(target_dim, emb_feature_dim)` indexed by feature ID. With `K = 1`, this is a single vector broadcast across the sequence.

The two are concatenated along the embedding dim. `side_dim = emb_time_dim + emb_feature_dim` must match the model's configured value.

## `ResidualBlock`

[`networks.ResidualBlock`](../networks.py#L135). Each block:

1. **Inject diffusion time.** `diffusion_emb` is projected to `channels` and broadcast-added to the input (same across `K·L`).
2. **Time attention** (`forward_time`): reshape to `(B·K, L, channels)`, apply positional encoding, pass through a transformer encoder layer. This is the dominant attention path for `K = 1` data.
3. **Feature attention** (`forward_feature`): reshape to `(B·L, K, channels)`. **Skipped when `K = 1`** (see caveat below).
4. **Mid projection.** 1×1 Conv1d to `2·channels`.
5. **Inject side info.** The `(B, side_dim, K, L)` conditioning tensor is passed through a small MLP (`side_dim → 128 → 128 → 2·channels`) per time step and added to the mid-projected features.
6. **Gated activation:** split into two halves and apply `σ(gate) ⊙ tanh(filter)` (WaveNet-style).
7. **Output projection.** 1×1 Conv1d to `2·channels`, split into `residual` and `skip`.
8. Return `(x + residual) / √2`, `skip`.

### Feature attention is a dead branch

`ResidualBlock.forward_feature` ([`networks.py:175`](../networks.py#L175)) references `self.feature_layer`, which is **never defined** in `__init__`. The block constructs `self.time_layer` but not a feature-side analogue. The only reason this doesn't crash is that `K = 1` triggers the early-return at `networks.py:177`:

```python
def forward_feature(self, y, base_shape):
    B, channel, K, L = base_shape
    if K == 1:
        return y
    ...
```

If you ever bump `target_dim` > 1, you must add `self.feature_layer = get_torch_trans(...)` (or `get_linear_trans(...)`) alongside `self.time_layer`.

### Positional encoding `max_len`

`ResidualBlock(max_len=10_000)` is the default, but `diff_CSDI` never forwards a different value. The sinusoidal buffer is precomputed for up to 10 000 positions — sequences longer than this need the default bumped.

## Efficient attention path

[`networks.get_linear_trans`](../networks.py#L15) wraps `LinearAttentionTransformer` with `max_seq_len = 256` hard-coded. That hard-coded cap is below the project's sequence length (2048), so `is_linear = True` only makes sense after bumping that parameter — or swapping the library. All current training presets use `is_linear = False` (standard transformer).

## Configured sizes across the codebase

`train.py`, `generate.py`, and `plot_result.py` all import the canonical config from [`model_config.py`](../model_config.py):

```python
MODEL_CONFIG = {
    "channels": 128,
    "diffusion_embedding_dim": 256,
    "target_dim": 1,
    "emb_feature_dim": 64,
    "side_dim": 2,             # overwritten at runtime
    "nheads": 8,
    "is_linear": False,
    "layers": 4,
}
```

Each script does `config = dict(MODEL_CONFIG)` then overwrites `side_dim = emb_time_dim + emb_feature_dim`. A checkpoint trained under one script is loadable by the others without per-script edits. To change architecture, edit `model_config.py` and retrain.

## Architecture ablation in the paper (§4.1)

The reference paper (Fig. 3, leverage-effect ablation) compares three configurations:

| Config | `channels` | `diffusion_embedding_dim` | `emb_feature_dim` | Reproduces leverage effect? |
|--------|-----------:|--------------------------:|-------------------:|-----------------------------|
| 64 / 128 / 16 (vanilla CSDI) | 64 | 128 | 16 | Volatility clustering & heavy tails fine; **leverage effect fails** |
| 128 / 256 / 32 | 128 | 256 | 32 | Improved but still imperfect |
| **128 / 256 / 64** *(paper default)* | 128 | 256 | **64** | **Most realistic, stable across schedules** |

Key takeaway: the **feature embedding dimension is the bottleneck for the leverage effect**, not channels or diffusion-step embedding. Increasing `emb_feature_dim` lets the network "separate low-frequency drift from high-frequency leverage shocks" (§3.2.1).

> **Note on the default.** `model_config.py` is set to `128 / 256 / 64` — the **paper's final choice** (third row of the ablation). The auto-selected directory names in `train.py` (`save_model_vp_cosine_64/`, etc.) carry the `_64` suffix to match. Earlier ablation values (16, 32) are documented for reference but not the default.

Other architecture-side details from §3.2.1:
- The codebase uses **`n = 4`** gated residual blocks, matching the paper.
- Three explicit embeddings are added before the first transformer block: diffusion-step, sinusoidal time, and a sequence positional encoding. The latter is the project's modification of vanilla CSDI; see `PositionalEncoding` in `networks.py:55`.
- All transformer attention uses 8 heads (`nheads=8`).
