import torch
import math


def str2bool(v):
    """Robust bool parser for argparse — accepts 'true'/'false'/'1'/'0'/'yes'/'no'.

    The original code used `type=str, default=True` which silently converted any
    CLI-supplied value to a truthy string, causing `--is_unconditional False` to
    behave as True. Use this as `type=` to fix that.
    """
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ('true', '1', 'yes', 'y', 't')

def sigma_t(t, sigma_min, sigma_max, noise_schedule):
    """
    t: torch.Tensor or float in [0,1]
    noise_schedule: "exponential", "linear", or "cosine"
    Returns: torch.Tensor of same shape as t
    """
    # convert inputs to tensors
    if not torch.is_tensor(t):
        t = torch.tensor(t, dtype=torch.get_default_dtype())
    if not torch.is_tensor(sigma_min):
        sigma_min = torch.tensor(sigma_min, dtype=t.dtype, device=t.device)
    if not torch.is_tensor(sigma_max):
        sigma_max = torch.tensor(sigma_max, dtype=t.dtype, device=t.device)

    if noise_schedule == "exponential":
        return sigma_min * ((sigma_max / sigma_min) ** t)
    elif noise_schedule == "linear":
        val = sigma_min**2 + t * (sigma_max**2 - sigma_min**2)
        return torch.sqrt(val)
    elif noise_schedule == "cosine":
        pi = torch.pi if hasattr(torch, "pi") else torch.tensor(3.141592653589793, dtype=t.dtype, device=t.device)
        return sigma_min + (sigma_max - sigma_min) * ((1 - torch.cos(pi * t)) / 2)
    else:
        raise ValueError(f"Unknown noise_schedule: {noise_schedule}")


def var_integral(t, sigma_min, sigma_max, noise_schedule):
    """
    Compute \u222b_0^t sigma^2(u) du for given schedule.
    t: torch.Tensor or float in [0,1]
    """
    # convert to tensors
    if not torch.is_tensor(t):
        t = torch.tensor(t, dtype=torch.get_default_dtype())
    if not torch.is_tensor(sigma_min):
        sigma_min = torch.tensor(sigma_min, dtype=t.dtype, device=t.device)
    if not torch.is_tensor(sigma_max):
        sigma_max = torch.tensor(sigma_max, dtype=t.dtype, device=t.device)

    if noise_schedule == "exponential":
        r = sigma_max**2 / (sigma_min**2 + 1e-40)
        one = torch.tensor(1.0, dtype=r.dtype, device=r.device)
        if torch.allclose(r, one, atol=1e-7):
            return sigma_min**2 * t
        else:
            return sigma_min**2 * (r.pow(t) - one) / torch.log(r + 1e-40)
    elif noise_schedule == "linear":
        return sigma_min**2 * t + 0.5 * (sigma_max**2 - sigma_min**2) * (t**2)
    elif noise_schedule == "cosine":
        A = sigma_min
        B = sigma_max - sigma_min
        pi = torch.pi if hasattr(torch, "pi") else torch.tensor(3.141592653589793, dtype=t.dtype, device=t.device)
        term1 = A**2 * t
        term2 = A * B * (t - (1/pi) * torch.sin(pi * t))
        term3 = (B**2 / 4) * ((3/2) * t - (2/pi) * torch.sin(pi * t) + (1/(4*pi)) * torch.sin(2 * pi * t))
        return term1 + term2 + term3
    else:
        raise ValueError(f"Unknown noise_schedule: {noise_schedule}")

def time_embedding(pos, d_model=128):
    pe = torch.zeros(pos.shape[0], pos.shape[1], d_model, device=pos.device, dtype=pos.dtype)
    position = pos.unsqueeze(2)
    div_term = 1 / torch.pow(
        10000.0,
        torch.arange(0, d_model, 2, device=pos.device, dtype=pos.dtype) / d_model
    )
    pe[:, :, 0::2] = torch.sin(position * div_term)
    pe[:, :, 1::2] = torch.cos(position * div_term)
    return pe

def get_side_info(x_tp, x_clean, embed_layer, target_dim, emb_time_dim):
    B, K, L = x_clean.shape

    time_embed = time_embedding(x_tp, emb_time_dim)  # (B,L,emb)
    time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
    feature_embed = embed_layer(
        torch.arange(target_dim).to(x_clean.device)
    )  # (K,emb)
    feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

    side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
    side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

    return side_info

def set_input_to_diffmodel(x_noisy, x_clean, is_unconditional):
    if is_unconditional:
        total_input = x_noisy.unsqueeze(1) # (B,1,K,L)
    else:
        obs = x_clean.unsqueeze(1)
        noisy_target = 0 * x_noisy.unsqueeze(1)
        total_input = torch.cat([obs, noisy_target], dim=1) # (B,2,K,L)
    return total_input