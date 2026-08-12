"""An infinite, continuous embedding field for exploring novel maps by dragging around,
plus the PCA basis it's sampled from.

Independently sampling each tile (generate_novel.py's original approach) has two
problems for a pannable map: revisiting the same spot gives a different result each
time, and adjacent tiles have no relationship to each other -- a house near a tile edge
just gets cut off, since neither tile's embedding grid knows the other exists.

This module fixes both by making the field a deterministic function of world-pixel
coordinates instead of a per-tile random draw:

  - PCA basis (mean, components, per-component variance): fit once over real Tessera
    embeddings via GPU SVD (torch.pca_lowrank), cached to disk so it's not refit on
    every app launch.
  - Lattice noise: every lattice point's PCA-coefficient vector is a deterministic hash
    of its (x, y) coordinate + a seed -- identical no matter which window reads it, and
    stable across process restarts. Bicubic interpolation between lattice points gives
    spatial smoothness (i.i.d. per-pixel noise would decode as static; see
    generate_novel.py's docstring for the same reasoning).
  - Real-embedding grounding: deterministic coordinate-modulo tiling of real embeddings,
    blended with the sampled field, so output stays closer to what the decoders learned
    to render.

Because two overlapping windows always read the same lattice points, requesting a tile
with extra padding on each side and cropping after decoding (see app.py's render_tile)
gives every tile's edge pixels real neighboring context -- which is what makes features
spanning a tile boundary come out coherent, without any explicit stitching.
"""

import os

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

import config


def pool_real_pixels(embeddings, coords, seed=0, max_pixels=200_000):
    """Pool per-pixel embedding vectors from training patches across all regions,
    subsampled for speed. Shared by PCA fitting (this module) and the region
    classifier's reference bank (classify.py) -- both need a representative sample of
    real embeddings.

    embeddings: list of (H_i, W_i, 128) normalized arrays, one per config.REGIONS entry.
    coords: (N, 3) array of (region_idx, y, x) patch top-left corners.
    """
    p = config.PATCH_SIZE
    pixels = np.concatenate(
        [embeddings[r][y : y + p, x : x + p, :].reshape(-1, config.EMBEDDING_DIM) for r, y, x in coords],
        axis=0,
    )
    rng = np.random.default_rng(seed)
    if len(pixels) > max_pixels:
        idx = rng.choice(len(pixels), size=max_pixels, replace=False)
        pixels = pixels[idx]
    return pixels


def fit_or_load_pca(embeddings, coords, k=config.PCA_COMPONENTS, seed=0,
                     path=config.PCA_CACHE, device="cpu", max_pixels=200_000):
    """Fit PCA (GPU SVD via torch.pca_lowrank) over real per-pixel embeddings pooled
    across all regions, or load a cached fit from a previous run. Returns
    (mean, components, explained_variance) as numpy arrays -- components has shape
    (k, 128)."""
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu")
        if ckpt["k"] == k:
            return ckpt["mean"].numpy(), ckpt["components"].numpy(), ckpt["explained_variance"].numpy()

    pixels = pool_real_pixels(embeddings, coords, seed=seed, max_pixels=max_pixels)
    pixels_t = torch.from_numpy(pixels).float().to(device)
    mean = pixels_t.mean(dim=0)
    centered = pixels_t - mean
    _, s, v = torch.pca_lowrank(centered, q=k, center=False)
    components = v.T  # (k, 128)
    explained_variance = (s ** 2) / (centered.shape[0] - 1)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {"mean": mean.cpu(), "components": components.cpu(), "explained_variance": explained_variance.cpu(), "k": k},
        path,
    )
    return mean.cpu().numpy(), components.cpu().numpy(), explained_variance.cpu().numpy()


def _cell_vector(li, lj, k, std, seed):
    h = (int(li) * 0x9E3779B1) ^ (int(lj) * 0x85EBCA6B) ^ seed
    rng = np.random.default_rng(h & 0xFFFFFFFF)
    return rng.standard_normal(k).astype(np.float32) * std


def noise_field_window(x0, y0, w, h, std, spacing=config.LATTICE_SPACING, seed=0):
    """Deterministic, spatially-smooth (h, w, k) field of PCA coefficients covering
    world-pixel window [x0, x0+w) x [y0, y0+h)."""
    k = len(std)
    li0, li1 = x0 // spacing - 2, (x0 + w) // spacing + 3
    lj0, lj1 = y0 // spacing - 2, (y0 + h) // spacing + 3

    lattice = np.stack(
        [np.stack([_cell_vector(li, lj, k, std, seed) for li in range(li0, li1)]) for lj in range(lj0, lj1)]
    )  # (lj1-lj0, li1-li0, k)

    lattice_x = np.arange(li0, li1) * spacing
    lattice_y = np.arange(lj0, lj1) * spacing
    interp = RegularGridInterpolator((lattice_y, lattice_x), lattice, method="cubic")

    query_y, query_x = np.meshgrid(np.arange(y0, y0 + h), np.arange(x0, x0 + w), indexing="ij")
    points = np.stack([query_y.ravel(), query_x.ravel()], axis=-1)
    return interp(points).reshape(h, w, k).astype(np.float32)


def real_grounding_window(x0, y0, w, h, embedding_norm):
    """Deterministic periodic tiling of real embeddings (coordinate-modulo, so it's
    revisit-stable). Real data repeats every (H, W) pixels -- a known prototype-scale
    limitation, not an issue for a small explorable area."""
    H, W, _ = embedding_norm.shape
    ys = np.arange(y0, y0 + h) % H
    xs = np.arange(x0, x0 + w) % W
    return embedding_norm[np.ix_(ys, xs)]


def embedding_window(x0, y0, w, h, mean, components, explained_variance, embedding_norm,
                      blend_alpha=config.BLEND_ALPHA, spacing=config.LATTICE_SPACING, seed=0):
    """(h, w, 128) normalized embedding grid at world-pixel position (x0, y0)."""
    std = np.sqrt(explained_variance)
    noise = noise_field_window(x0, y0, w, h, std, spacing, seed)
    sampled = mean + noise @ components
    grounding = real_grounding_window(x0, y0, w, h, embedding_norm)
    return blend_alpha * sampled + (1 - blend_alpha) * grounding
