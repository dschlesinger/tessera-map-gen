"""Train the deterministic one-shot decoder: embedding(128ch) -> RGB(3ch) in one forward pass.

Loss is L1 + (1 - SSIM), the standard combo for pixel-accurate-but-not-blurry image
regression. This is the fast baseline / data-pipeline sanity check -- compare its
validation quality against train_diffusion_decoder.py once both have run.

Not run locally in this session -- executed on cloud compute by the user.
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from dataset import load_patches
from models import OneStepDecoder


def ssim(img1, img2, window_size=11, data_range=2.0):
    """Differentiable single-scale SSIM with a uniform (box-filter) window.

    Images are expected in (N, C, H, W) with values in [-1, 1] (data_range=2.0).
    """
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    pad = window_size // 2

    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=pad)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=pad)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=pad) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=pad) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=pad) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def loss_fn(pred, target, ssim_weight=0.5):
    l1 = F.l1_loss(pred, target)
    ssim_term = 1 - ssim(pred, target)
    return l1 + ssim_weight * ssim_term, l1.item(), ssim_term.item()


def run_validation(model, val_loader, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for embedding, rgb in val_loader:
            embedding, rgb = embedding.to(device), rgb.to(device)
            pred = model(embedding)
            loss, _, _ = loss_fn(pred, rgb)
            total += loss.item() * embedding.size(0)
            n += embedding.size(0)
    model.train()
    return total / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}, compute capability {torch.cuda.get_device_capability(0)}")
        torch.backends.cudnn.benchmark = False

    train_ds, val_ds = load_patches()
    print(f"Train patches: {len(train_ds)}, val patches: {len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = OneStepDecoder(in_ch=config.EMBEDDING_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for embedding, rgb in train_loader:
            embedding, rgb = embedding.to(device), rgb.to(device)
            pred = model(embedding)
            loss, l1, ssim_term = loss_fn(pred, rgb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * embedding.size(0)

        train_loss = running / len(train_ds)
        val_loss = run_validation(model, val_loader, device)
        print(
            f"epoch {epoch + 1}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state_dict": model.state_dict(), "val_loss": val_loss}, config.ONESTEP_CHECKPOINT)
            print(f"  saved new best checkpoint (val_loss={val_loss:.4f})")

    print(f"Done. Best val_loss={best_val:.4f}, checkpoint at {config.ONESTEP_CHECKPOINT}")


if __name__ == "__main__":
    main()
