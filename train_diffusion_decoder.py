"""Train the conditional diffusion decoder: embedding(128ch) conditions a DDPM that
denoises toward the RGB(3ch) target.

Unlike the one-step decoder, this can produce multiple different plausible RGB patches
for the *same* embedding (stochastic decoding) -- the property the novel-map sampler
needs, since decoding a synthesized (never-observed) embedding should give a specific,
detailed-looking place rather than an averaged blur.

Not run locally in this session -- executed on cloud compute by the user.
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

import config
from dataset import load_patches
from models import DiffusionUNet


class GaussianDiffusion:
    """Linear-beta DDPM schedule: q_sample (forward noising), p_losses (training
    objective), and DDIM sampling (fast deterministic-path reverse sampling for eval)."""

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def q_sample(self, x0, t, noise):
        sqrt_ac = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_omac = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_ac * x0 + sqrt_omac * noise

    def p_losses(self, model, x0, embedding, t):
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred_noise = model(x_t, embedding, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def ddim_sample(self, model, embedding, steps=50, generator=None):
        """embedding -> RGB, starting from pure noise, deterministic DDIM path (eta=0).

        Pass a seeded torch.Generator to make the starting noise (and hence the whole
        sample) reproducible -- used by app.py so revisiting the same map tile decodes
        to the same image instead of a fresh random draw each time.
        """
        device = embedding.device
        batch = embedding.shape[0]
        shape = (batch, 3, embedding.shape[2], embedding.shape[3])
        x = torch.randn(shape, device=device, generator=generator)

        step_indices = torch.linspace(
            self.timesteps - 1, 0, steps, dtype=torch.long, device=device
        )
        for i, t in enumerate(step_indices):
            t_batch = t.expand(batch)
            pred_noise = model(x, embedding, t_batch)
            alpha_cumprod_t = self.alphas_cumprod[t]
            x0_pred = (x - torch.sqrt(1 - alpha_cumprod_t) * pred_noise) / torch.sqrt(
                alpha_cumprod_t
            )
            x0_pred = x0_pred.clamp(-1, 1)

            if i == len(step_indices) - 1:
                x = x0_pred
            else:
                t_next = step_indices[i + 1]
                alpha_cumprod_next = self.alphas_cumprod[t_next]
                x = (
                    torch.sqrt(alpha_cumprod_next) * x0_pred
                    + torch.sqrt(1 - alpha_cumprod_next) * pred_noise
                )
        return x


def run_validation(model, diffusion, val_loader, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for embedding, rgb in val_loader:
            embedding, rgb = embedding.to(device), rgb.to(device)
            t = torch.randint(0, diffusion.timesteps, (embedding.size(0),), device=device)
            loss = diffusion.p_losses(model, rgb, embedding, t)
            total += loss.item() * embedding.size(0)
            n += embedding.size(0)
    model.train()
    return total / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="train on all patches (train+val combined), no held-out validation -- for the final production run",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}, compute capability {torch.cuda.get_device_capability(0)}")
        torch.backends.cudnn.benchmark = False

    train_ds, val_ds = load_patches()
    if args.full_dataset:
        train_ds = ConcatDataset([train_ds, val_ds])
        print(f"Full-dataset mode: training on {len(train_ds)} patches, no held-out validation")
    else:
        print(f"Train patches: {len(train_ds)}, val patches: {len(val_ds)}")
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = DiffusionUNet(embedding_dim=config.EMBEDDING_DIM).to(device)
    diffusion = GaussianDiffusion(timesteps=args.timesteps, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for embedding, rgb in train_loader:
            embedding, rgb = embedding.to(device), rgb.to(device)
            t = torch.randint(0, args.timesteps, (embedding.size(0),), device=device)
            loss = diffusion.p_losses(model, rgb, embedding, t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * embedding.size(0)

        train_loss = running / len(train_ds)

        if args.full_dataset:
            print(f"epoch {epoch + 1}/{args.epochs}  train_loss={train_loss:.4f}")
            if train_loss < best_val:
                best_val = train_loss
                torch.save(
                    {"model_state_dict": model.state_dict(), "train_loss": train_loss, "timesteps": args.timesteps},
                    config.DIFFUSION_CHECKPOINT,
                )
                print(f"  saved new best checkpoint (train_loss={train_loss:.4f})")
        else:
            val_loss = run_validation(model, diffusion, val_loader, device)
            print(
                f"epoch {epoch + 1}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            )
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {"model_state_dict": model.state_dict(), "val_loss": val_loss, "timesteps": args.timesteps},
                    config.DIFFUSION_CHECKPOINT,
                )
                print(f"  saved new best checkpoint (val_loss={val_loss:.4f})")

    print(f"Done. Best val_loss={best_val:.4f}, checkpoint at {config.DIFFUSION_CHECKPOINT}")


if __name__ == "__main__":
    main()
