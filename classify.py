"""Region classifier over the embedding field, for the map app's heatmap overlay.

No labeled land-cover dataset ships with Tessera, so this works the way Cambridge's own
tessera-interactive-map tool does: cluster real embeddings unsupervised (k-means) for a
default view, then let a human click points on the map to name regions -- classification
immediately switches to nearest-labeled-point-in-embedding-space, which needs only a
handful of clicks because Tessera's embeddings already separate land-cover types well
(see arXiv:2506.20380).

Also provides the "realistic" check behind app.py's red tile border: a tile whose
embedding sits very close to the k-means reference bank isn't really novel/fictional --
it's basically reproducing something that was actually observed. The distance threshold
is calibrated from real data itself (the 90th-percentile nearest-reference distance
among real pixels), not a guessed constant.
"""

import colorsys
import hashlib
import json
import os

import numpy as np
import torch
from sklearn.cluster import KMeans

import config
from field import pool_real_pixels


def build_or_load_bank(embeddings, coords, n_clusters=config.CLASSIFY_CLUSTERS,
                        seed=0, path=config.CLUSTER_CACHE, device="cpu"):
    """embeddings/coords: same multi-region format as field.pool_real_pixels.
    Returns (centers (n_clusters, 128) float32, novelty_threshold float)."""
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu")
        if ckpt["centers"].shape[0] == n_clusters:
            return ckpt["centers"].numpy(), float(ckpt["threshold"])

    pixels = pool_real_pixels(embeddings, coords, seed=seed)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=4).fit(pixels)
    centers = kmeans.cluster_centers_.astype(np.float32)

    centers_t = torch.from_numpy(centers).to(device)
    sample_t = torch.from_numpy(pixels[:20_000]).float().to(device)
    dists = torch.cdist(sample_t, centers_t).min(dim=1).values
    threshold = torch.quantile(dists, 0.9).item()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"centers": torch.from_numpy(centers), "threshold": threshold}, path)
    return centers, threshold


def load_labels(path=config.LABELS_FILE):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def save_labels(labels, path=config.LABELS_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(labels, f)


def _color_for_key(key):
    """Deterministic, pleasant color for a label name or cluster id -- same key always
    gets the same color, without needing a hardcoded palette."""
    h = int(hashlib.md5(str(key).encode()).hexdigest(), 16)
    hue = (h % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.9)
    return int(r * 255), int(g * 255), int(b * 255)


def classify_grid(embedding_grid, centers, labels, device="cpu"):
    """embedding_grid: (h, w, 128) normalized embedding.
    Returns (h, w, 3) uint8 color map: nearest labeled point if any labels exist,
    else nearest unsupervised cluster center."""
    h, w, _ = embedding_grid.shape
    flat = torch.from_numpy(embedding_grid.reshape(-1, config.EMBEDDING_DIM)).float().to(device)

    if labels:
        vectors = torch.tensor([lab["vector"] for lab in labels], device=device).float()
        idx = torch.cdist(flat, vectors).argmin(dim=1).cpu().numpy()
        palette = np.array([_color_for_key(lab["name"]) for lab in labels], dtype=np.uint8)
    else:
        centers_t = torch.from_numpy(centers).to(device)
        idx = torch.cdist(flat, centers_t).argmin(dim=1).cpu().numpy()
        palette = np.array([_color_for_key(i) for i in range(len(centers))], dtype=np.uint8)

    return palette[idx].reshape(h, w, 3)


def novelty_score(embedding_grid, centers, device="cpu"):
    """Mean nearest-reference distance over the grid. Low means "close to something
    real that was actually observed" -- i.e. not really a novel/fictional tile."""
    flat = torch.from_numpy(embedding_grid.reshape(-1, config.EMBEDDING_DIM)).float().to(device)
    centers_t = torch.from_numpy(centers).to(device)
    dists = torch.cdist(flat, centers_t).min(dim=1).values
    return dists.mean().item()
