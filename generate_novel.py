"""Generate novel (fictional) satellite-style map tiles from both trained decoders.

A novel embedding grid never corresponds to any real place: it's sampled from a PCA
model fit over real Tessera embeddings, with spatial smoothness imposed by upsampling
a low-resolution noise field (sampling every pixel i.i.d. would decode as static, since
real land-cover has strong spatial autocorrelation). Optionally blended with a crop of
real embeddings for grounding.

The one-step decoder gives one deterministic image per embedding grid. The diffusion
decoder can draw multiple different plausible images from the *same* embedding grid
(stochastic decoding via DDIM) -- pass --diffusion-samples > 1 to see that.

Not run locally in this session -- executed on cloud compute by the user.
"""

import argparse
import json
import os

import numpy as np
import torch
from skimage.io import imsave
from skimage.transform import resize
from sklearn.decomposition import PCA

import config
from models import DiffusionUNet, OneStepDecoder
from train_diffusion_decoder import GaussianDiffusion


def fit_pca(embedding, train_coords, k, max_pixels=200_000, seed=0):
    """Fit PCA over real per-pixel embedding vectors pooled from training patches."""
    p = config.PATCH_SIZE
    pixels = []
    for y, x in train_coords:
        pixels.append(embedding[y : y + p, x : x + p, :].reshape(-1, config.EMBEDDING_DIM))
    pixels = np.concatenate(pixels, axis=0)

    rng = np.random.default_rng(seed)
    if len(pixels) > max_pixels:
        idx = rng.choice(len(pixels), size=max_pixels, replace=False)
        pixels = pixels[idx]

    pca = PCA(n_components=k, random_state=seed)
    pca.fit(pixels)
    return pca


def sample_embedding_grid(pca, size, downscale, rng):
    """Sample a spatially-smooth (size, size, 128) embedding grid from the PCA model."""
    low = max(1, size // downscale)
    std = np.sqrt(pca.explained_variance_)  # (K,)
    low_res = rng.standard_normal((low, low, pca.n_components_)).astype(np.float32) * std
    upsampled = resize(low_res, (size, size), order=3, anti_aliasing=False, mode="edge")
    return pca.mean_ + upsampled @ pca.components_


def ground_with_real_crop(sampled_grid, embedding, size, blend_alpha, rng):
    """Blend the sampled grid with a lerp of two random real crops for grounding.

    blend_alpha is the weight of the *sampled* deviation; (1 - blend_alpha) is the
    weight of the real-interpolated grounding.
    """
    h, w, _ = embedding.shape
    y1, x1 = rng.integers(0, h - size), rng.integers(0, w - size)
    y2, x2 = rng.integers(0, h - size), rng.integers(0, w - size)
    crop1 = embedding[y1 : y1 + size, x1 : x1 + size, :]
    crop2 = embedding[y2 : y2 + size, x2 : x2 + size, :]
    r = rng.uniform(0.0, 1.0)
    real_interp = r * crop1 + (1 - r) * crop2
    return blend_alpha * sampled_grid + (1 - blend_alpha) * real_interp


def to_uint8_image(rgb_tensor):
    """(3, H, W) tensor in [-1, 1] -> (H, W, 3) uint8 array."""
    img = rgb_tensor.clamp(-1, 1).permute(1, 2, 0).cpu().numpy()
    return ((img + 1.0) * 127.5).round().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tiles", type=int, default=1, help="number of novel embedding grids to generate")
    parser.add_argument("--size", type=int, default=128, help="output tile height/width in pixels")
    parser.add_argument("--k-components", type=int, default=24, help="PCA components kept from real embeddings")
    parser.add_argument("--downscale", type=int, default=8, help="low-res noise field is size/downscale before upsampling")
    parser.add_argument("--blend-alpha", type=float, default=0.3, help="weight of PCA-sampled deviation vs. real-crop grounding (0=pure real interp, 1=pure sampled)")
    parser.add_argument("--diffusion-samples", type=int, default=3, help="stochastic decodes per tile from the diffusion model")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="generated")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    embedding = np.load(config.EMBEDDING_CACHE)
    train_coords = np.load(f"{config.DATA_DIR}/train_coords.npy")
    with open(f"{config.DATA_DIR}/norm_stats.json") as f:
        stats = json.load(f)
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    embedding_norm = (embedding - mean) / std

    print(f"Fitting PCA (k={args.k_components}) over real embeddings...")
    pca = fit_pca(embedding_norm, train_coords, args.k_components, seed=args.seed)
    print(f"  explained variance ratio (top {args.k_components}): {pca.explained_variance_ratio_.sum():.3f}")

    onestep = OneStepDecoder(in_ch=config.EMBEDDING_DIM).to(device)
    onestep.load_state_dict(torch.load(config.ONESTEP_CHECKPOINT, map_location=device)["model_state_dict"])
    onestep.eval()

    diffusion_ckpt = torch.load(config.DIFFUSION_CHECKPOINT, map_location=device)
    diffusion_model = DiffusionUNet(embedding_dim=config.EMBEDDING_DIM).to(device)
    diffusion_model.load_state_dict(diffusion_ckpt["model_state_dict"])
    diffusion_model.eval()
    diffusion = GaussianDiffusion(timesteps=diffusion_ckpt["timesteps"], device=device)

    for i in range(args.n_tiles):
        sampled = sample_embedding_grid(pca, args.size, args.downscale, rng)
        grid = ground_with_real_crop(sampled, embedding_norm, args.size, args.blend_alpha, rng)
        emb_tensor = torch.from_numpy(grid).float().permute(2, 0, 1)[None].to(device)

        with torch.no_grad():
            onestep_pred = onestep(emb_tensor)[0]
        imsave(os.path.join(args.out_dir, f"tile{i}_onestep.png"), to_uint8_image(onestep_pred), check_contrast=False)

        with torch.no_grad():
            batch = emb_tensor.repeat(args.diffusion_samples, 1, 1, 1)
            diffusion_preds = diffusion.ddim_sample(diffusion_model, batch, steps=args.ddim_steps)
        for j in range(args.diffusion_samples):
            imsave(
                os.path.join(args.out_dir, f"tile{i}_diffusion_sample{j}.png"),
                to_uint8_image(diffusion_preds[j]),
                check_contrast=False,
            )

        print(f"tile {i + 1}/{args.n_tiles} done")

    print(f"Saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
