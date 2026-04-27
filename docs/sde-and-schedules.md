# SDEs, noise schedules, and the DSM loss

Sources: [`utils.py`](../utils.py), [`losses.py`](../losses.py).

The forward process (clean → noise) is a Stochastic Differential Equation parameterized by a time-dependent noise magnitude `σ(t)`. The model learns the **score** of the marginal at each `t`. Reverse-time integration (see [sampling.md](sampling.md)) turns that score into samples.

This codebase supports **three SDE families** — VE, VP, and the paper's **GBM** mode (selected via `--alpha 1`) — and **three noise schedules** (exponential, linear, cosine). VE and VP are the standard score-based-modeling forward processes (Song et al., 2020); the GBM variant is the contribution of [the reference paper](https://arxiv.org/abs/2507.19003). All three share the same `σ(t)` and variance integral `V(t) = ∫₀ᵗ σ²(u) du` machinery.

## Variance integral

Most of the math reduces to a single quantity:

```
V(t) := ∫₀ᵗ σ²(u) du
```

Implemented in closed form in [`utils.var_integral`](../utils.py#L30). The closed forms are:

| Schedule | `σ(t)` | `V(t) = ∫₀ᵗ σ²(u) du` |
|----------|--------|------------------------|
| exponential | `σ_min · (σ_max/σ_min)^t` | `σ_min² · (r^t − 1) / log(r)`  where `r = σ_max²/σ_min²` |
| linear (in variance) | `√(σ_min² + t(σ_max² − σ_min²))` | `σ_min² · t + ½(σ_max² − σ_min²) · t²` |
| cosine | `σ_min + (σ_max − σ_min) · (1 − cos(πt))/2` | See [`utils.py:52-59`](../utils.py#L52) for the 3-term expansion |

All three interpolate `σ(0) = σ_min` and `σ(1) = σ_max`. Cosine is the most empirically robust (smoother transition near both endpoints); exponential compounds fastest; linear-in-variance is the most analytically convenient.

Numerical detail worth knowing:

- The exponential branch falls back to `σ_min² · t` when `r ≈ 1` (`torch.allclose(r, one, atol=1e-7)`) — a cheap safeguard against `log(r) → 0`.
- `var_integral` is **clamped** to `≥ 1e-20` in `losses.denoising_score_matching_loss` before taking `sqrt`, to avoid NaN at `t ≈ 0`.

## VE SDE (Variance Exploding)

Forward process:

```
dx = σ(t) dW         →     x(t) = x(0) + √V(t) · ε,    ε ~ N(0, I)
```

Marginal at time `t`:

```
p_t(x | x_0) = N(x | x_0, V(t) · I)
```

So the true score is:

```
∇_x log p_t(x | x_0) = −(x − x_0) / V(t)
```

Used in [`losses.py:32-51`](../losses.py#L32).

**Initialization at `t = 1`** for sampling: `x ~ N(0, V(1) · I)`. In practice `σ_max` is chosen so `V(1)` is large enough that the prior effectively covers the data. See [`generate.py:59-60`](../generate.py#L59).

## VP SDE (Variance Preserving)

Forward process (with `β(t) := σ²(t)`):

```
dx = −½ β(t) x dt + σ(t) dW
```

Marginal:

```
x(t) = γ(t) · x(0) + √(1 − γ(t)²) · ε,   γ(t) := exp(−½ V(t))
```

Implemented in [`losses.py:53-77`](../losses.py#L53). True score becomes:

```
∇_x log p_t(x | x_0) = −(x − γ(t) x_0) / (1 − γ(t)²)
```

As `t → 1`, `V(t) → V(1)` grows; `γ(t) → 0`; the marginal becomes `N(0, (1 − γ(1)²) · I) ≈ N(0, I)` when `V(1)` is large. Thus the reverse-time sampler can start from a standard-normal-ish prior. See [`generate.py:61-64`](../generate.py#L61).

**VE vs VP at a glance:**

| | VE | VP |
|---|---|---|
| Forward drift | 0 | `−½ β(t) x` |
| Marginal std | `√V(t)` (unbounded) | `√(1 − e^{−V(t)})` (bounded by 1) |
| Score | `−(x − x₀)/V(t)` | `−(x − γ x₀)/(1 − γ²)` |
| Prior at `t=1` | `N(0, V(1) I)` — depends on `σ_max` | `N(0, (1 − γ(1)²) I) ≈ N(0, I)` |
| Good default | data unbounded, tails matter | data roughly standardized |

In this project both are used. The training presets (see [training.md](training.md#presets)) default α=0 + VP + cosine, and α=1 + VP + cosine.

## GBM SDE (`--alpha 1`, the paper's contribution)

This is the central novelty of the [reference paper](https://arxiv.org/abs/2507.19003) (§3). The forward process is **Geometric Brownian Motion in price space**:

```
dS_t = μ_t · S_t · dt + σ_t · S_t · dW_t
```

with the multiplicative volatility characteristic of asset prices. Itô's lemma applied to `X_t = log S_t` gives:

```
dX_t = (μ_t − ½ σ_t²) dt + σ_t dW_t
```

Choosing **`μ_t = ½ σ_t²`** cancels the drift, leaving:

```
dX_t = σ_t dW_t      (a VE SDE in log-coordinates)
```

So in log-space the math is *identical* to the VE SDE above — the same `V(t)`, the same closed-form score `−(x − x₀)/V(t)`, the same DSM loss. What changes is the **data representation**: training is on standardized log-prices (`data/sp500_subseq_log.pt`) rather than log-returns. Once samples are drawn, exponentiating back to price space gives trajectories with state-dependent volatility — large prices get proportionally larger fluctuations, naturally producing heteroskedasticity, heavy tails, and the leverage effect.

| Aspect | VE on log-returns (`α=0`) | **GBM (`α=1`)** |
|---|---|---|
| Trains on | Standardized log-returns | Standardized log-prices |
| Implied price-space dynamics | Additive Gaussian noise | Multiplicative (GBM-like) |
| Data shard | `data/sp500_subseq.pt` | `data/sp500_subseq_log.pt` |
| Model dir prefix | `save_model_ve_*` / `save_model_vp_*` | `save_model_bs_*` ("Black–Scholes") |
| Post-sample step | inverse-scale → returns | inverse-scale → `np.diff` → log-returns |

The codebase implements the GBM regime simply by swapping the input shard and the post-processing pipeline — the score network, loss, and sampler are unchanged. See [`docs/sampling.md`](sampling.md#alpha-modes) for the post-processing.

**Empirical comparison (paper §4, tail exponents).** Empirical S&P 500 baseline: `α = 4.35`. Pure VE produces light tails (`α ≈ 8.5–9` for linear/exponential). GBM with the **exponential** schedule lands at `α = 4.62`; with **cosine**, `α = 3.78` — both very close to empirical. Full table in §4 of the [paper](https://arxiv.org/abs/2507.19003).

**Defined but unused:** [`generate.sample_init_bs`](../generate.py#L13) draws an initial state directly from the GBM forward marginal `exp(√V(1)·ε)` (lognormal with `X_0 = 0`). It's intended as a more faithful prior for `α=1` sampling, but `predictor_corrector_sampling` does not currently call it — see [`docs/sampling.md`](sampling.md#initialization-t--1).

## Denoising score matching loss

[`losses.denoising_score_matching_loss`](../losses.py#L5) is the sole training objective. The procedure for each minibatch:

1. Draw `t ~ U(0, 1)` independently per sample.
2. Compute `V(t)` and (for VP) `γ(t)`.
3. Add noise: `x_t = x_0 + √V(t) · ε` (VE) or `x_t = γ(t) x_0 + √(1 − γ²) · ε` (VP).
4. Forward through the model: `score_pred = model(total_input, side_info, t)` (with `total_input` assembled by [`utils.set_input_to_diffmodel`](../utils.py#L89); see [architecture.md](architecture.md#input-packing)).
5. Compute the analytic true score from the noising equation.
6. Loss: `(score_pred − score_true)²`, mean-reduced over `(channel, length)` dims.
7. **Variance weighting:** multiply per-sample loss by `V(t)`, then average. This is `λ(t) = V(t)` in the generalized DSM formulation; it upweights high-`t` (noisier) samples which are otherwise down-weighted by the `1/V(t)²` scale of the score.

### `num_scales`

The function accepts `num_scales` (default 1 from `train.py:146`). When > 1, it averages the loss over that many independent `t` draws per minibatch. Larger values reduce variance of the loss estimator at the cost of more model forward passes per step.

### Caveats

- **`alpha` is unused inside the loss.** The signature takes `alpha` but never reads it. The `α=0` (VE/VP on log-returns) and `α=1` (GBM, log-prices) regimes share this exact loss; α only changes what the input tensor *means*. This works because, as derived in §3 of the paper, the GBM forward process reduces to a VE SDE in log-coordinates — so the analytic score and loss are identical. See [sampling.md](sampling.md#alpha-modes).
- **`1e-8` guard on VP denominator.** When `V(t)` is tiny (small `t`), `1 − exp(−V(t))` is tiny too. The `+ 1e-8` in [`losses.py:71`](../losses.py#L71) prevents the score from blowing up at `t ≈ 0`.
- **Per-sample `V(t)` weighting, not batch average.** The weight is applied element-wise (`weighted_loss = var_t * loss`) before `.mean()`, so the gradient magnitudes scale with the actual `V(t)` each sample saw.
