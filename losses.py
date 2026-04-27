import torch
from utils import var_integral, get_side_info, set_input_to_diffmodel


def denoising_score_matching_loss(
    model,
    x_clean,
    x_tp,
    sde,
    alpha,
    sigma_min,
    sigma_max,
    noise_schedule,
    emb_time_dim,
    is_unconditional,
    device,
    num_scales=1,
):
    """Denoising score-matching loss for VE/VP SDEs."""
    B = x_clean.shape[0]
    target_dim = x_clean.shape[1]
    total_loss = 0.0

    if hasattr(model, "module"):
        embed_layer = model.module.embed_layer
    else:
        embed_layer = model.embed_layer

    for _ in range(num_scales):
        t_ = torch.rand(B, device=device)

        if sde == "VE":
            var_t = var_integral(t_, sigma_min, sigma_max, noise_schedule).clamp_min(1e-20)
            std_t = var_t.sqrt()

            eps = torch.randn_like(x_clean)
            std_t_bkl = std_t.view(B, 1, 1)
            x_noisy = x_clean + std_t_bkl * eps

            total_input = set_input_to_diffmodel(x_noisy, x_clean, is_unconditional)
            side_info = get_side_info(x_tp, x_clean, embed_layer, target_dim, emb_time_dim)
            pred_score = model(total_input, side_info, t_)

            var_t_bkl = var_t.view(B, 1, 1)
            score_true = -(x_noisy - x_clean) / var_t_bkl

            loss = (pred_score - score_true) ** 2
            loss = torch.mean(loss, dim=(1, 2))
            weighted_loss = var_t * loss

            total_loss += weighted_loss.mean()

        elif sde == "VP":
            var_t = var_integral(t_, sigma_min, sigma_max, noise_schedule).clamp_min(1e-20)

            gamma_t = torch.exp(-0.5 * var_t)
            std_t_square = 1 - torch.exp(-var_t)
            std_t = std_t_square.sqrt()

            gamma_t_bkl = gamma_t.view(B, 1, 1)
            std_t_bkl = std_t.view(B, 1, 1)

            eps = torch.randn_like(x_clean)
            x_noisy = gamma_t_bkl * x_clean + std_t_bkl * eps

            total_input = set_input_to_diffmodel(x_noisy, x_clean, is_unconditional)
            side_info = get_side_info(x_tp, x_clean, embed_layer, target_dim, emb_time_dim)
            pred_score = model(total_input, side_info, t_)

            std_t_square_bkl = std_t_square.view(B, 1, 1)
            score_true = -(x_noisy - gamma_t_bkl * x_clean) / (std_t_square_bkl + 1e-8)

            loss = (pred_score - score_true) ** 2
            loss = torch.mean(loss, dim=(1, 2))
            weighted_loss = var_t * loss

            total_loss += weighted_loss.mean()

    return total_loss / num_scales
