import os
import math
import torch
import argparse
import numpy as np
from tqdm.auto import tqdm

from model_config import MODEL_CONFIG
from networks import diff_CSDI
from utils import sigma_t, var_integral, get_side_info, set_input_to_diffmodel, str2bool


def sample_init_bs(
    batch_size, shape,
    sigma_min, sigma_max,
    noise_schedule="VE",
    device='cpu',
):
    """
    GBM (alpha=1, Black-Scholes) forward: X_1 = X_0 * exp(sqrt(var(1)) * eps).
    With X_0 = 1 this gives a lognormal(0, var(1)) initial value; returns log-space y.
    """
    var_1 = var_integral(1.0, sigma_min, sigma_max, noise_schedule)

    num_elems = batch_size * np.prod(shape)
    eps_np = np.random.randn(num_elems).astype(np.float32)
    eps_torch = torch.from_numpy(eps_np).view(batch_size, *shape).to(device)

    x_init = torch.exp(eps_torch * math.sqrt(var_1)).clamp_min(1e-8)
    return torch.log(x_init)


def predictor_corrector_sampling(
    model,
    x_clean,
    x_tp,
    sde,
    alpha=0.0,
    sigma_min=0.01,
    sigma_max=10.0,
    noise_schedule="VE",
    emb_time_dim=128,
    is_unconditional=False,
    steps=50,
    snr=0.2,
    n_corr=1,
    num_samples=64,
    device='cuda',
):
    """Song et al. Predictor-Corrector sampler (Reverse Euler predictor + Langevin corrector)."""
    if hasattr(model, "module"):
        embed_layer = model.module.embed_layer
    else:
        embed_layer = model.embed_layer
    target_dim = x_clean.shape[1]

    var_1 = var_integral(1.0, sigma_min, sigma_max, noise_schedule)
    eps = torch.randn_like(x_clean)
    if sde == "VE":
        x = eps * math.sqrt(var_1)
    elif sde == "VP":
        gamma = math.exp(-0.5 * var_1)
        sigma_square = 1 - gamma ** 2
        x = eps * math.sqrt(sigma_square)
    else:
        raise ValueError(f"Unknown sde: {sde}")

    dt = 1.0 / steps

    for i in tqdm(range(steps)):
        t_cur = 1.0 - i / steps
        t_tensor = torch.ones(num_samples, device=device) * t_cur

        with torch.no_grad():
            s_t = sigma_t(t_cur, sigma_min, sigma_max, noise_schedule)

            total_input = set_input_to_diffmodel(x, x_clean, is_unconditional)
            side_info = get_side_info(x_tp, x_clean, embed_layer, target_dim, emb_time_dim)
            score = model(total_input, side_info, t_tensor)
            z = torch.randn_like(x)

            if sde == "VE":
                drift = -(s_t ** 2) * score * dt
            else:  # VP
                drift = -0.5 * (s_t ** 2) * dt - (s_t ** 2) * score * dt
            diffusion = s_t * math.sqrt(dt) * z
            x = x + drift + diffusion

            for _ in range(n_corr):
                total_input = set_input_to_diffmodel(x, x_clean, is_unconditional)
                side_info = get_side_info(x_tp, x_clean, embed_layer, target_dim, emb_time_dim)
                grad = model(total_input, side_info, t_tensor)
                z = torch.randn_like(x)
                grad_norm = torch.sqrt(torch.mean(grad ** 2, dim=(1, 2), keepdim=True))
                z_norm = torch.sqrt(torch.mean(z ** 2, dim=(1, 2), keepdim=True))
                # Song et al. Langevin step size: eps = 2 * (snr * ||z|| / ||grad||)^2
                eps = 2 * (snr * z_norm / (grad_norm + 1e-10)) ** 2
                eps = eps.view(-1, 1, 1)
                x = x + eps * grad + torch.sqrt(2 * eps) * z

    return x


def load_financial_dataset(processed_file="data/sp500_subseq.pt"):
    if not os.path.exists(processed_file):
        raise FileNotFoundError(f"{processed_file} not found!")
    loaded = torch.load(processed_file, weights_only=False)
    return loaded['data'], loaded['timepoints'], loaded['labels']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='save_model_alpha_1/model_epoch_2401.pth')
    parser.add_argument('--processed_file', type=str, default='data/financial_test_data.pt')
    parser.add_argument('--out_file', type=str, default='denoised_financial_test_data.pt')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--sde', type=str, default='VE')
    parser.add_argument('--alpha', type=float, default=0, help="0.5 or 1.0")
    parser.add_argument('--sigma_min', type=float, default=0.01)
    parser.add_argument('--sigma_max', type=float, default=0.5)
    parser.add_argument('--emb_time_dim', type=int, default=128)
    parser.add_argument('--emb_feature_dim', type=int, default=16)
    parser.add_argument('--noise_schedule', type=str, default="exponential")
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--snr', type=float, default=0.2)
    parser.add_argument('--n_corr', type=int, default=1)
    parser.add_argument('--is_unconditional', type=str2bool, default=True)

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

    data, timepoints, labels = load_financial_dataset(args.processed_file)
    dataset = torch.utils.data.TensorDataset(data, timepoints, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    denoised_list = []
    label_list = []
    for batch_data, batch_time, batch_labels in tqdm(dataloader, desc="Denoising Financial Time Series"):
        batch_data = batch_data.to(device)
        batch_time = batch_time.to(device)
        s = 20
        samples = []
        for _ in range(s):
            denoised_sample = predictor_corrector_sampling(
                model=model,
                x_clean=batch_data,
                x_tp=batch_time,
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
                device=device,
            )
            samples.append(denoised_sample)
        denoised_avg = torch.stack(samples, dim=0).mean(dim=0)
        denoised_list.append(denoised_avg.cpu())
        label_list.append(batch_labels.cpu())

    denoised_all = torch.cat(denoised_list, dim=0)
    labels_all = torch.cat(label_list, dim=0)
    torch.save({'denoised': denoised_all, 'labels': labels_all}, args.out_file)
    print(f"Saved denoised data with labels to {args.out_file}")


if __name__ == "__main__":
    main()
