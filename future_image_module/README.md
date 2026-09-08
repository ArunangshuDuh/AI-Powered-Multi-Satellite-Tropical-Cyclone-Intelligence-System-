# Future image module (not wired into the running API)

The files in this folder are **not used by `api/main.py`**. They're a ready-to-run
scaffold for the real TCIR-satellite-image half of the brief (Sections 2–5 of the
build brief), which could not be executed in the sandbox this project was built in —
there was no network access, and the TCIR dataset is distributed via Google Drive /
Baidu Pan / university mirrors rather than a plain downloadable URL.

What's here, and what it does:

- **`cyclone_cnn.py`** — `CycloneCNN`: EfficientNet-B0 backbone via `timm`, with the
  first conv layer **weight-inflated** from 3→5 channels (averages the pretrained RGB
  filters and repeats across the 5 TCIR channels — IR1/WV/PMW/VIS/VIS-mask — instead of
  training a fresh stem from random init), plus classification and regression heads
  with metadata fusion. Matches Section 3 of the brief.
- **`train_cnn.py`** — training loop: mixed precision, cosine LR schedule, combined
  cross-entropy + λ·MSE loss, checkpointing every epoch.
- **`trend_gru.py`** — the GRU trend-prediction head described in Section 4 (frozen
  CNN embeddings + numeric state → +6h/+12h/+24h wind/pressure).
- **`gradcam.py`** — real Grad-CAM implementation for the classification head, for
  when there's an actual trained CNN to explain.
- **`config_reference.py`** — the category map, wind thresholds, and hyperparameters
  this code expects (kept separate from the live `../config.py` since the two modules
  don't share a runtime today).

## How to actually wire this in later

1. Get TCIR onto a machine with a GPU and download bandwidth (see the main
   `README.md` "What's real vs generated" section for the exact mirror URL).
2. Run `train_cnn.py` following Section 2–3 of the build brief (resample, per-channel
   z-score, `WeightedRandomSampler`, cache to a single tensor file first).
3. `CycloneCNN.get_feature_extractor()` gives you the penultimate-layer embedding per
   frame — feed 3–4 consecutive frames' embeddings + numeric state into `trend_gru.py`
   the same way `models/model.py`'s GRU already consumes numeric state today. The two
   don't conflict: `models/model.py` (the live, numeric-only model) can be swapped for
   an image+numeric fusion model without changing the two output heads or the API
   contract.
4. Replace `scripts/generate_visuals.py`'s procedural PNGs with real composited TCIR
   frames for `/image`, and real `gradcam.py` output for `/gradcam`. `api/main.py`
   doesn't need to change — it already just serves whatever PNG is on disk.
