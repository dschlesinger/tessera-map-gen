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

# Novel-map generation (generate_novel.py, field.py, app.py)
GEN_TILE_SIZE = 128          # output tile size in pixels
LATTICE_SPACING = 16         # px between deterministic noise-field lattice points
DECODE_MARGIN = 32           # px of context padded onto each side before decoding, then cropped
PCA_COMPONENTS = 24          # components kept when fitting PCA over real embeddings
BLEND_ALPHA = 0.3            # weight of PCA-sampled deviation vs. real-crop grounding
DDIM_STEPS = 50
PCA_CACHE = f"{CHECKPOINT_DIR}/pca_model.pt"
