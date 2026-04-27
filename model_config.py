"""Canonical diff_CSDI config shared by train.py, generate.py, plot_result.py.

Keeping the architecture parameters in one place ensures a checkpoint saved by
train.py loads cleanly into the inference scripts without per-script edits.

`side_dim` is a placeholder; each script overwrites it at runtime as
`side_dim = emb_time_dim + emb_feature_dim` once `emb_time_dim` is known.
"""

MODEL_CONFIG = {
    "channels": 128,
    "diffusion_embedding_dim": 256,
    "target_dim": 1,
    "emb_feature_dim": 64,
    "side_dim": 2,
    "nheads": 8,
    "is_linear": False,
    "layers": 4,
}
