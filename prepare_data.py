"""Environment setup + data download + parsing for the Tessera novel-map decoders.

Run once before either training script. Steps:
  1. Install pipeline dependencies (pip install -r requirements.txt).
  2. Download Tessera embedding tiles + a matching Sentinel-2 RGB composite for BBOX/YEAR,
     reprojected onto the same pixel grid as the embeddings.
  3. Parse into a fixed non-overlapping patch grid, split train/val, and compute embedding
     normalization stats -- all cached to disk so the training scripts only need to load,
     never recompute.

Not run locally in this session -- executed on cloud compute by the user.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from geotessera import GeoTessera
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject

import config


def install_requirements():
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    print(f"Installing dependencies from {req_path}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])


def fetch_embeddings():
    """Download Tessera embedding tiles for BBOX/YEAR and mosaic into one array.

    Returns (embedding, transform, crs) where embedding has shape (H, W, 128).
    """
    gt = GeoTessera()
    tiles = gt.registry.load_blocks_for_region(bounds=config.BBOX, year=config.YEAR)

    with tempfile.TemporaryDirectory() as tmp:
        paths = gt.export_embedding_geotiffs(tiles_to_fetch=tiles, output_dir=tmp, bands=None)
        opened = [rasterio.open(p) for p in paths]
        crs = opened[0].crs
        vrts = [WarpedVRT(s, crs=crs, resampling=Resampling.bilinear) for s in opened]
        mosaic, transform = merge(vrts)
        for v in vrts:
            v.close()
        for s in opened:
            s.close()

    # merge() returns (bands, H, W); embeddings are stored as 128 bands -> move to (H, W, 128).
    embedding = np.moveaxis(mosaic, 0, -1).astype(np.float32)
    return embedding, transform, crs


def fetch_rgb_composite(out_shape, transform, crs, max_cloud_cover=20, max_scenes=6):
    """Fetch a low-cloud Sentinel-2 RGB composite reprojected onto the embedding grid.

    out_shape: (H, W) to match the embedding array.
    Returns an (H, W, 3) uint8 array.
    """
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=config.BBOX,
        datetime=f"{config.YEAR}-01-01/{config.YEAR}-12-31",
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        max_items=max_scenes,
    )
    items = list(search.item_collection())
    if not items:
        raise RuntimeError(
            f"No Sentinel-2 scenes found for {config.BBOX} in {config.YEAR} "
            f"under {max_cloud_cover}% cloud cover."
        )

    height, width = out_shape
    stack = np.zeros((len(items), height, width, 3), dtype=np.float32)
    valid = np.zeros((len(items), height, width), dtype=bool)

    for i, item in enumerate(items):
        signed = planetary_computer.sign(item)
        with rasterio.open(signed.assets["visual"].href) as src:
            dst = np.zeros((3, height, width), dtype=np.uint8)
            reproject(
                source=rasterio.band(src, [1, 2, 3]),
                destination=dst,
                dst_transform=transform,
                dst_crs=crs,
                resampling=Resampling.bilinear,
            )
        rgb = np.moveaxis(dst, 0, -1).astype(np.float32)
        stack[i] = rgb
        valid[i] = rgb.sum(axis=-1) > 0  # drop nodata/black pixels from the median

    # Per-pixel median over valid scenes knocks out clouds/shadows that slipped through
    # the cloud-cover filter; falls back to 0 where no scene had a valid pixel.
    stack_masked = np.where(valid[..., None], stack, np.nan)
    composite = np.nanmedian(stack_masked, axis=0)
    composite = np.nan_to_num(composite, nan=0.0)
    return composite.astype(np.uint8)


def _patch_grid(height, width, patch_size):
    ys = range(0, height - patch_size + 1, patch_size)
    xs = range(0, width - patch_size + 1, patch_size)
    return [(y, x) for y in ys for x in xs]


def parse_and_split(embedding):
    """Build the non-overlapping patch grid, split train/val, compute embedding
    normalization stats (channel mean/std) from a sample of training patches only,
    so validation patches never leak into the stats used at training time."""
    height, width = embedding.shape[:2]
    coords = _patch_grid(height, width, config.PATCH_SIZE)

    rng = np.random.default_rng(config.RANDOM_SEED)
    rng.shuffle(coords)
    n_val = max(1, int(len(coords) * config.VAL_FRACTION))
    val_coords = coords[:n_val]
    train_coords = coords[n_val:]

    sample_idx = np.linspace(0, len(train_coords) - 1, min(200, len(train_coords)), dtype=int)
    sample_vecs = []
    for i in sample_idx:
        y, x = train_coords[i]
        p = config.PATCH_SIZE
        sample_vecs.append(embedding[y : y + p, x : x + p, :].reshape(-1, config.EMBEDDING_DIM))
    sample_vecs = np.concatenate(sample_vecs, axis=0)
    mean = sample_vecs.mean(axis=0)
    std = sample_vecs.std(axis=0)
    std[std < 1e-6] = 1e-6

    return np.array(train_coords), np.array(val_coords), mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-install", action="store_true", help="skip pip install -r requirements.txt"
    )
    args = parser.parse_args()

    if not args.skip_install:
        install_requirements()

    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    print(f"Fetching Tessera embeddings for {config.BBOX}, {config.YEAR}...")
    embedding, transform, crs = fetch_embeddings()
    print(f"Embedding grid: {embedding.shape}")

    print("Fetching matching Sentinel-2 RGB composite...")
    rgb = fetch_rgb_composite(embedding.shape[:2], transform, crs)
    print(f"RGB grid: {rgb.shape}")

    print("Parsing into patch grid, train/val split, normalization stats...")
    train_coords, val_coords, mean, std = parse_and_split(embedding)
    print(f"Train patches: {len(train_coords)}, val patches: {len(val_coords)}")

    np.save(config.EMBEDDING_CACHE, embedding)
    np.save(config.RGB_CACHE, rgb)
    np.save(f"{config.DATA_DIR}/train_coords.npy", train_coords)
    np.save(f"{config.DATA_DIR}/val_coords.npy", val_coords)
    with open(f"{config.DATA_DIR}/norm_stats.json", "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
    with open(config.METADATA_CACHE, "w") as f:
        json.dump(
            {
                "bbox": config.BBOX,
                "year": config.YEAR,
                "crs": str(crs),
                "transform": list(transform)[:6],
                "shape": list(embedding.shape),
            },
            f,
            indent=2,
        )

    print(f"Done. Cached to {config.DATA_DIR}/ -- ready for train_onestep_decoder.py / train_diffusion_decoder.py")


if __name__ == "__main__":
    main()
