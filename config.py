"""Shared config for the Tessera novel-map prototype pipeline."""

# Prototype regions: several small bboxes chosen for terrain variety (city/hills/forest/
# coast) instead of one bigger contiguous area -- gets the decoder diverse training data
# for a few GB total, instead of ~100GB+ for e.g. all of England.
# (name, (min_lon, min_lat, max_lon, max_lat))
REGIONS = [
    ("cambridge_city", (0.05, 52.15, 0.20, 52.27)),
    ("lake_district_hills", (-3.15, 54.45, -2.95, 54.58)),
    ("new_forest_woodland", (-1.65, 50.83, -1.45, 50.93)),
    ("norfolk_coast", (1.15, 52.90, 1.35, 52.98)),
]
YEAR = 2024

# Which region's real embeddings the live map (app.py) tiles for grounding texture.
# Kept to a single region rather than switching between regions to avoid a visible seam
# where the dominant (grounding-weighted) texture would jump between terrain types.
GROUNDING_REGION_INDEX = 0

EMBEDDING_DIM = 128
PATCH_SIZE = 64
VAL_FRACTION = 0.15
RANDOM_SEED = 0

DATA_DIR = "data"
REGIONS_FILE = f"{DATA_DIR}/regions.json"
METADATA_CACHE = f"{DATA_DIR}/metadata.json"


def embedding_cache(region_idx):
    return f"{DATA_DIR}/embeddings_{region_idx}.npy"


def rgb_cache(region_idx):
    return f"{DATA_DIR}/rgb_{region_idx}.npy"


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

# Region classifier (classify.py, app.py heatmap)
CLASSIFY_CLUSTERS = 300      # unsupervised k-means clusters, used until real labels are added
CLUSTER_CACHE = f"{CHECKPOINT_DIR}/cluster_bank.pt"
LABELS_FILE = f"{CHECKPOINT_DIR}/labels.json"
NOVELTY_BORDER_PX = 6        # red border thickness on tiles too close to a real observed location
