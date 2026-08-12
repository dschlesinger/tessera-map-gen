"""Generate novel (fictional) satellite-style map tiles from both trained decoders.

Each tile is a window into the same continuous embedding field the map-drag app
(app.py) walks through -- just sampled at a random world position instead of adjacent
ones. See field.py's docstring for how the field is built (PCA over real embeddings +
deterministic lattice noise + real-crop grounding) and why it's spatially smooth rather
than i.i.d. noise.

The one-step decoder gives one deterministic image per embedding grid. The diffusion
decoder can produce multiple different plausible images from the *same* embedding grid
(stochastic decoding via DDIM) -- pass --diffusion-samples > 1 to see that.

Not run locally in this session -- executed on cloud compute by the user.
"""

import argparse
import json
import os

import numpy as np
import torch
from skimage.io import imsave

import config
import field
from models import DiffusionUNet, OneStepDecoder
from train_diffusion_decoder import GaussianDiffusion


def to_uint8_image(rgb_tensor):
    """(3, H, W) tensor in [-1, 1] -> (H, W, 3) uint8 array."""
    img = rgb_tensor.clamp(-1, 1).permute(1, 2, 0).cpu().numpy()
    return ((img + 1.0) * 127.5).round().astype(np.uint8)


def load_field_inputs(device):
    """PCA is fit pooling pixels across every region (terrain variety); the returned
    embedding_norm used for real-crop grounding is a single region (config.
    GROUNDING_REGION_INDEX) to keep it spatially coherent for periodic tiling."""
    with open(config.REGIONS_FILE) as f:
        n_regions = len(json.load(f)["names"])
    with open(f"{config.DATA_DIR}/norm_stats.json") as f:
        stats = json.load(f)
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)

    embeddings_norm = [(np.load(config.embedding_cache(i)) - mean) / std for i in range(n_regions)]
    train_coords = np.load(f"{config.DATA_DIR}/train_coords.npy")

    pca_mean, components, explained_variance = field.fit_or_load_pca(embeddings_norm, train_coords, device=device)
    return embeddings_norm[config.GROUNDING_REGION_INDEX], pca_mean, components, explained_variance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tiles", type=int, default=1)
    parser.add_argument("--size", type=int, default=config.GEN_TILE_SIZE)
    parser.add_argument("--blend-alpha", type=float, default=config.BLEND_ALPHA)
    parser.add_argument("--diffusion-samples", type=int, default=3, help="stochastic decodes per tile from the diffusion model")
    parser.add_argument("--ddim-steps", type=int, default=config.DDIM_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="generated")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("Loading data and fitting/loading PCA...")
    embedding_norm, pca_mean, components, explained_variance = load_field_inputs(device)

    onestep = OneStepDecoder(in_ch=config.EMBEDDING_DIM).to(device)
    onestep.load_state_dict(torch.load(config.ONESTEP_CHECKPOINT, map_location=device)["model_state_dict"])
    onestep.eval()

    diffusion_ckpt = torch.load(config.DIFFUSION_CHECKPOINT, map_location=device)
    diffusion_model = DiffusionUNet(embedding_dim=config.EMBEDDING_DIM).to(device)
    diffusion_model.load_state_dict(diffusion_ckpt["model_state_dict"])
    diffusion_model.eval()
    diffusion = GaussianDiffusion(timesteps=diffusion_ckpt["timesteps"], device=device)

    for i in range(args.n_tiles):
        x0, y0 = rng.integers(-1_000_000, 1_000_000, size=2)
        grid = field.embedding_window(
            int(x0), int(y0), args.size, args.size, pca_mean, components, explained_variance,
            embedding_norm, blend_alpha=args.blend_alpha, seed=args.seed,
        )
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
