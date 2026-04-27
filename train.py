import os
import glob
import signal
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from tqdm.auto import tqdm

from model_config import MODEL_CONFIG
from networks import diff_CSDI
from losses import denoising_score_matching_loss
from utils import str2bool


class PreprocessedFinancialDataset(torch.utils.data.Dataset):
    """Loads a .pt file of preprocessed S&P 500 subsequences (data/timepoints/meta)."""
    def __init__(self, processed_file):
        if not os.path.exists(processed_file):
            raise FileNotFoundError(f"Preprocessed file {processed_file} not found.")
        loaded = torch.load(processed_file, weights_only=False)
        self.data = loaded['data']              # list of (target_length,) tensors
        self.timepoints = loaded['timepoints']  # list of (target_length,) tensors
        self.meta = loaded['meta']              # list of (ticker, start_date)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx].unsqueeze(0)
        time_sample = self.timepoints[idx]
        return sample, time_sample


def sigint_handler(signum, frame):
    print("SIGINT received. Attempting graceful shutdown...")
    raise KeyboardInterrupt

signal.signal(signal.SIGINT, sigint_handler)

log_file = None


def is_main_process():
    """True for the rank-0 process under DDP, or always True in single-process mode."""
    return (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        for param in m.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)
    elif isinstance(m, nn.LayerNorm):
        if m.elementwise_affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def load_checkpoint(model_dir, model, optimizer, device):
    """Load the highest-numbered checkpoint into model + optimizer; returns the next epoch index."""
    checkpoint_files = glob.glob(os.path.join(model_dir, 'model_epoch_*.pth'))
    if checkpoint_files:
        checkpoint_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]), reverse=True)
        latest_checkpoint = checkpoint_files[0]
        checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if is_main_process():
            print(f"Checkpoint loaded from '{latest_checkpoint}'. Resuming training from epoch {start_epoch}.")
        return start_epoch
    if is_main_process():
        print("No saved model found. Starting training from scratch.")
    return 0


def load_logs():
    train_losses = []
    val_losses = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                train_losses.append(float(row['train_loss']))
                val_losses.append(float(row['val_loss']))
    return train_losses, val_losses


def save_log(epoch, train_loss, val_loss):
    file_exists = os.path.exists(log_file)
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['epoch', 'train_loss', 'val_loss'])
        writer.writerow([epoch, train_loss, val_loss])


def train(args):
    # --- Distributed-or-single setup --------------------------------------
    # When launched via `torchrun`, env vars LOCAL_RANK / RANK / WORLD_SIZE
    # / MASTER_ADDR / MASTER_PORT are pre-populated. Otherwise, run single-process.
    ddp = 'LOCAL_RANK' in os.environ
    if ddp:
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        world_size = dist.get_world_size()
        if is_main_process():
            print(f"DDP enabled: world_size={world_size}, this rank's local_rank={local_rank}")
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        world_size = 1
        print(f"Single-process training on device: {device}")

    # --- Data -------------------------------------------------------------
    dataset = PreprocessedFinancialDataset(processed_file=args.data_file)
    if ddp:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=False)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    else:
        sampler = None
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # --- Model + optimizer ------------------------------------------------
    config = dict(MODEL_CONFIG)
    config["side_dim"] = args.emb_time_dim + config["emb_feature_dim"]

    input_dim = 1 if args.is_unconditional else 2
    if is_main_process():
        print(f"is_unconditional={args.is_unconditional!r} -> input_dim={input_dim}")

    model = diff_CSDI(config, input_dim).to(device)
    # output_projection2 is intentionally zero-initialized inside diff_CSDI;
    # initialize_weights is applied to the rest of the submodules.
    model.apply(initialize_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Load checkpoint BEFORE the DDP wrap so both modes hit the same code path
    # and saved state_dicts are always in unwrapped form.
    start_epoch = load_checkpoint(args.model_dir, model, optimizer, device)

    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    # --- Training loop ----------------------------------------------------
    try:
        for epoch in range(start_epoch, args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            total_loss = 0.0
            for batch_data, batch_time in tqdm(
                dataloader,
                desc=f"Epoch[{epoch+1}/{args.epochs}]",
                disable=not is_main_process(),
            ):
                x = batch_data.to(device)
                timepoints = batch_time.to(device)

                optimizer.zero_grad()
                loss = denoising_score_matching_loss(
                    model=model,
                    x_clean=x,
                    x_tp=timepoints,
                    sde=args.sde,
                    alpha=args.alpha,
                    sigma_min=args.sigma_min,
                    sigma_max=args.sigma_max,
                    noise_schedule=args.noise_schedule,
                    emb_time_dim=args.emb_time_dim,
                    is_unconditional=args.is_unconditional,
                    device=device,
                    num_scales=1,
                )
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Aggregate loss across ranks for logging
            if ddp:
                loss_tensor = torch.tensor([total_loss], dtype=torch.float, device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                avg_loss = (loss_tensor.item() / world_size) / len(dataloader)
            else:
                avg_loss = total_loss / len(dataloader)

            if is_main_process():
                print(f"Epoch [{epoch+1}/{args.epochs}] - Train DSM Loss: {avg_loss:.6f}")

                ckpt_path = os.path.join(args.model_dir, f"model_epoch_{epoch+1}.pth")
                # Always save UNWRAPPED state_dict so checkpoints stay portable
                # between single-GPU and DDP runs.
                state_dict = model.module.state_dict() if ddp else model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, ckpt_path)

                # Keep only the latest checkpoint — drop everything older.
                stale = [
                    f for f in glob.glob(os.path.join(args.model_dir, 'model_epoch_*.pth'))
                    if os.path.abspath(f) != os.path.abspath(ckpt_path)
                ]
                for f in stale:
                    try:
                        os.remove(f)
                    except OSError as e:
                        print(f"Warning: could not remove old checkpoint {f}: {e}")

                if stale:
                    print(f"Saved checkpoint to {ckpt_path} (replaced {len(stale)} older).")
                else:
                    print(f"Saved checkpoint to {ckpt_path}.")

    except KeyboardInterrupt:
        if is_main_process():
            print("Ctrl+C received; exiting without saving checkpoint for this epoch.")

    if ddp:
        dist.destroy_process_group()


def run_training():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='data/sp500_subseq_log.pt',
                        help='Path to the preprocessed S&P 500 subsequence .pt file (log returns).')
    parser.add_argument('--model_dir', type=str, default='save_model_bs_exponential_64')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Per-rank batch size when DDP-launched; effective batch is batch_size * world_size.')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sde', type=str, default='VP')
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--sigma_min', type=float, default=0.01)
    parser.add_argument('--sigma_max', type=float, default=1.0)
    parser.add_argument('--model_channels', type=int, default=64)
    parser.add_argument('--emb_time_dim', type=int, default=128)
    parser.add_argument('--emb_feature_dim', type=int, default=16)
    parser.add_argument('--noise_schedule', type=str, default='exponential')
    parser.add_argument('--is_unconditional', type=str2bool, default=True)

    args = parser.parse_args()

    # Pick a model_dir/data_file preset based on (alpha, sde, noise_schedule).
    # The schedule is baked into the dir name so different schedules don't
    # collide in one folder (and so auto-resume picks the matching checkpoint).
    if args.alpha == 1:
        args.model_dir = f'save_model_bs_{args.noise_schedule}_64'
        args.data_file = 'data/sp500_subseq_log.pt'
    elif args.alpha == 0:
        args.data_file = 'data/sp500_subseq.pt'
        if args.sde == 'VE':
            args.model_dir = f'save_model_ve_{args.noise_schedule}_64'
        elif args.sde == 'VP':
            args.model_dir = f'save_model_vp_{args.noise_schedule}_64'
    else:
        args.model_dir = 'save_model_cev'

    os.makedirs(args.model_dir, exist_ok=True)
    global log_file
    log_file = os.path.join(args.model_dir, 'training_log.csv')

    train(args)


if __name__ == '__main__':
    run_training()
