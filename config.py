"""Shared config for the Tessera novel-map prototype pipeline."""

# Prototype region: Cambridge, UK (geotessera's own documented example area).
# (min_lon, min_lat, max_lon, max_lat)
BBOX = (-0.2, 51.4, 0.1, 51.6)
YEAR = 2024

EMBEDDING_DIM = 128
PATCH_SIZE = 64
VAL_FRACTION = 0.15
RANDOM_SEED = 0

DATA_DIR = "data"
EMBEDDING_CACHE = f"{DATA_DIR}/embeddings.npy"
RGB_CACHE = f"{DATA_DIR}/rgb.npy"
METADATA_CACHE = f"{DATA_DIR}/metadata.json"

CHECKPOINT_DIR = "checkpoints"
ONESTEP_CHECKPOINT = f"{CHECKPOINT_DIR}/onestep_decoder.pt"
DIFFUSION_CHECKPOINT = f"{CHECKPOINT_DIR}/diffusion_decoder.pt"
