"""Small web UI for generating novel Tessera-decoded map tiles on demand.

Pick a decoder (one-step or diffusion) and generate a gallery of tiles, each from an
independent random PCA sample in embedding space (see generate_novel.py for the sampling
method). Loads data + fits PCA once at startup; decoder checkpoints are loaded lazily on
first use of each model.

Not run locally in this session -- executed on cloud compute by the user. The Flask dev
server binds 0.0.0.0:5000; if running on a remote/cluster node, port-forward or tunnel to
view it in a browser.
"""

import base64
import io
import json

import numpy as np
import torch
from flask import Flask, jsonify, render_template_string, request
from skimage.io import imsave

import config
from generate_novel import fit_pca, ground_with_real_crop, sample_embedding_grid, to_uint8_image
from models import DiffusionUNet, OneStepDecoder
from train_diffusion_decoder import GaussianDiffusion

app = Flask(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_state = {"models": {}}


def load_data_and_pca(k_components=24, seed=0):
    embedding = np.load(config.EMBEDDING_CACHE)
    train_coords = np.load(f"{config.DATA_DIR}/train_coords.npy")
    with open(f"{config.DATA_DIR}/norm_stats.json") as f:
        stats = json.load(f)
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    embedding_norm = (embedding - mean) / std
    pca = fit_pca(embedding_norm, train_coords, k_components, seed=seed)
    return embedding_norm, pca


def get_onestep():
    if "onestep" not in _state["models"]:
        model = OneStepDecoder(in_ch=config.EMBEDDING_DIM).to(DEVICE)
        model.load_state_dict(torch.load(config.ONESTEP_CHECKPOINT, map_location=DEVICE)["model_state_dict"])
        model.eval()
        _state["models"]["onestep"] = model
    return _state["models"]["onestep"]


def get_diffusion():
    if "diffusion" not in _state["models"]:
        ckpt = torch.load(config.DIFFUSION_CHECKPOINT, map_location=DEVICE)
        model = DiffusionUNet(embedding_dim=config.EMBEDDING_DIM).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        diffusion = GaussianDiffusion(timesteps=ckpt["timesteps"], device=DEVICE)
        _state["models"]["diffusion"] = (model, diffusion)
    return _state["models"]["diffusion"]


def image_to_data_uri(img_uint8):
    buf = io.BytesIO()
    imsave(buf, img_uint8, format="png", check_contrast=False)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@app.route("/generate", methods=["POST"])
def generate():
    payload = request.get_json(force=True)
    decoder = payload.get("decoder", "onestep")
    n_tiles = int(payload.get("n_tiles", 4))
    size = int(payload.get("size", 128))
    blend_alpha = float(payload.get("blend_alpha", 0.3))
    downscale = int(payload.get("downscale", 8))
    ddim_steps = int(payload.get("ddim_steps", 50))

    embedding_norm, pca = _state["data"]
    rng = np.random.default_rng()  # fresh randomness per request -> random starting positions
    images = []

    for _ in range(n_tiles):
        sampled = sample_embedding_grid(pca, size, downscale, rng)
        grid = ground_with_real_crop(sampled, embedding_norm, size, blend_alpha, rng)
        emb_tensor = torch.from_numpy(grid).float().permute(2, 0, 1)[None].to(DEVICE)

        if decoder == "diffusion":
            model, diffusion = get_diffusion()
            with torch.no_grad():
                pred = diffusion.ddim_sample(model, emb_tensor, steps=ddim_steps)[0]
        else:
            model = get_onestep()
            with torch.no_grad():
                pred = model(emb_tensor)[0]

        images.append(image_to_data_uri(to_uint8_image(pred)))

    return jsonify({"images": images})


PAGE = """
<!doctype html>
<html>
<head>
<title>Tessera Novel Maps</title>
<style>
  body { font-family: sans-serif; max-width: 900px; margin: 2rem auto; }
  .controls { display: flex; gap: 1rem; align-items: center; margin-bottom: 1rem; flex-wrap: wrap; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 8px; }
  .grid img { width: 100%; border-radius: 4px; image-rendering: pixelated; }
  button { padding: 0.5rem 1rem; }
</style>
</head>
<body>
  <h1>Tessera Novel Maps</h1>
  <div class="controls">
    <label>Decoder:
      <select id="decoder">
        <option value="onestep">One-step</option>
        <option value="diffusion">Diffusion</option>
      </select>
    </label>
    <label>Tiles: <input id="n_tiles" type="number" value="4" min="1" max="16" style="width:4em"></label>
    <button id="go">Generate</button>
    <span id="status"></span>
  </div>
  <div class="grid" id="grid"></div>
<script>
document.getElementById("go").addEventListener("click", async () => {
  const status = document.getElementById("status");
  const grid = document.getElementById("grid");
  status.textContent = "generating...";
  grid.innerHTML = "";
  const resp = await fetch("/generate", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      decoder: document.getElementById("decoder").value,
      n_tiles: parseInt(document.getElementById("n_tiles").value, 10),
    }),
  });
  const data = await resp.json();
  status.textContent = "";
  for (const src of data.images) {
    const img = document.createElement("img");
    img.src = src;
    grid.appendChild(img);
  }
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


if __name__ == "__main__":
    print("Loading data and fitting PCA...")
    _state["data"] = load_data_and_pca()
    print("Ready.")
    app.run(host="0.0.0.0", port=5000)
