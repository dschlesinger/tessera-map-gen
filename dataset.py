"""Paired embedding/RGB patch Dataset. Reads data already downloaded and parsed by
prepare_data.py -- run that first; this module only loads, it never computes."""

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

    def __init__(self, embedding, rgb, coords, mean, std):
        self.embedding = embedding
        self.rgb = rgb
        self.coords = coords
        self.mean = mean.reshape(1, 1, -1)
        self.std = std.reshape(1, 1, -1)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        y, x = self.coords[idx]
        p = config.PATCH_SIZE
        emb_patch = self.embedding[y : y + p, x : x + p, :]
        rgb_patch = self.rgb[y : y + p, x : x + p, :]

        emb_patch = (emb_patch - self.mean) / self.std
        rgb_patch = (rgb_patch.astype(np.float32) / 127.5) - 1.0

        emb_tensor = torch.from_numpy(np.ascontiguousarray(emb_patch)).permute(2, 0, 1)
        rgb_tensor = torch.from_numpy(np.ascontiguousarray(rgb_patch)).permute(2, 0, 1)
        return emb_tensor.float(), rgb_tensor.float()


def load_patches():
    """Load arrays + patch coords + normalization stats prepared by prepare_data.py.

    Returns: train_dataset, val_dataset
    """
    embedding = np.load(config.EMBEDDING_CACHE)
    rgb = np.load(config.RGB_CACHE)
    train_coords = np.load(f"{config.DATA_DIR}/train_coords.npy")
    val_coords = np.load(f"{config.DATA_DIR}/val_coords.npy")
    with open(f"{config.DATA_DIR}/norm_stats.json") as f:
        stats = json.load(f)
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)

    train_ds = TesseraPatchDataset(embedding, rgb, train_coords, mean, std)
    val_ds = TesseraPatchDataset(embedding, rgb, val_coords, mean, std)
    return train_ds, val_ds
