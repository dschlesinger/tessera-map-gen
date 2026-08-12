"""Paired embedding/RGB patch Dataset. Reads data already downloaded and parsed by
prepare_data.py -- run that first; this module only loads, it never computes.

Patches are pooled across all of config.REGIONS (see prepare_data.py), so coords are
(region_idx, y, x) triples rather than plain (y, x) -- each region has its own
(H_i, W_i, C) array since regions are geographically disjoint and can't be mosaicked
into one grid.
"""

import json

import numpy as np
import torch
from torch.utils.data import Dataset

import config


class TesseraPatchDataset(Dataset):
    """Yields (embedding, rgb) patch pairs.

    embedding: float32 tensor (128, P, P), standardized with train-set channel mean/std.
    rgb:       float32 tensor (3, P, P), scaled to [-1, 1].
    """

    def __init__(self, embeddings, rgbs, coords, mean, std):
        self.embeddings = embeddings  # list of (H_i, W_i, 128) arrays, one per region
        self.rgbs = rgbs  # list of (H_i, W_i, 3) arrays, one per region
        self.coords = coords  # (N, 3): region_idx, y, x
        self.mean = mean.reshape(1, 1, -1)
        self.std = std.reshape(1, 1, -1)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        region_idx, y, x = self.coords[idx]
        p = config.PATCH_SIZE
        emb_patch = self.embeddings[region_idx][y : y + p, x : x + p, :]
        rgb_patch = self.rgbs[region_idx][y : y + p, x : x + p, :]

        emb_patch = (emb_patch - self.mean) / self.std
        rgb_patch = (rgb_patch.astype(np.float32) / 127.5) - 1.0

        emb_tensor = torch.from_numpy(np.ascontiguousarray(emb_patch)).permute(2, 0, 1)
        rgb_tensor = torch.from_numpy(np.ascontiguousarray(rgb_patch)).permute(2, 0, 1)
        return emb_tensor.float(), rgb_tensor.float()


def load_patches():
    """Load arrays + patch coords + normalization stats prepared by prepare_data.py.

    Returns: train_dataset, val_dataset
    """
    with open(config.REGIONS_FILE) as f:
        n_regions = len(json.load(f)["names"])
    embeddings = [np.load(config.embedding_cache(i)) for i in range(n_regions)]
    rgbs = [np.load(config.rgb_cache(i)) for i in range(n_regions)]

    train_coords = np.load(f"{config.DATA_DIR}/train_coords.npy")
    val_coords = np.load(f"{config.DATA_DIR}/val_coords.npy")
    with open(f"{config.DATA_DIR}/norm_stats.json") as f:
        stats = json.load(f)
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)

    train_ds = TesseraPatchDataset(embeddings, rgbs, train_coords, mean, std)
    val_ds = TesseraPatchDataset(embeddings, rgbs, val_coords, mean, std)
    return train_ds, val_ds
