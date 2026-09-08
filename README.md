# Tropical Cyclone AI System — Backend

This is a merge of two parallel builds against the same frozen API contract: one that
had a complete, well-organized API/config scaffold but mocked data; one that had real
NOAA data, a real trained model, and honest 501s where it didn't have real data. This
version keeps the real pipeline and data from the second, fills in the two endpoints
it had left as placeholders so the full contract works end-to-end, and folds the
first build's image-CNN architecture in as a documented upgrade path rather than
throwing it away.

**Read "What's real vs generated" below before presenting this anywhere** — it's the
most important section in this file.

## Quick start

```bash
pip install -r requirements.txt
python scripts/preprocess.py              # rebuilds cache/ from data/atlantic_storms.csv
python scripts/train.py                   # sanity check + full training -> outputs/best_model.pt
python scripts/precompute_demo_storms.py  # builds outputs/demo_storms/*.json (served by the API)
python scripts/generate_visuals.py        # builds outputs/demo_storms/images/*.png (served by the API)
uvicorn api.main:app --reload --port 8000
```

Steps 2–5 have already been run once — `outputs/` ships with their results, so
`uvicorn api.main:app --reload` alone is enough to demo it. Re-run the earlier steps
only if you change the data or model.

Test it: `curl http://127.0.0.1:8000/api/health` → `{"status":"ok","model_loaded":true}`

## What's real vs generated — read this before a demo or a judge asks

**Real, actually computed from data:**
- **Data**: 14,599 real observations across 490 real Atlantic storms (1980–2015),
  from NOAA's HURDAT2 best-track database (`data/atlantic_storms.csv`).
- **Model**: `models/model.py` — a GRU-based multi-task model, actually trained on
  real sequences of that data, doing three real things: classifying current storm
  category, regressing current wind/pressure, and predicting the +6h/+12h/+24h trend.
- **Metrics** (`outputs/metrics.json`): persistence baseline wind MAE **6.79 kt**,
  model wind MAE **6.39 kt** (beats baseline by ~5.9%), plus full per-epoch train/val
  history. Classification accuracy is ~99.9% — expected, not impressive, since current
  wind is part of the model's own input for that head; it's confirming the mapping
  works, not doing something clever.
- **4 real demo storms**, precomputed end-to-end from real tracks: Sandy (2012),
  Katrina (2005), Wilma (2005), Andrew (1992) — verified against real historical peak
  categories.
- **Rapid-intensification flag**: computed for real from the model's own predicted
  +24h wind vs. current wind, against the standard 30kt/24h operational threshold —
  not hand-set. Known real limitation, stated honestly: it never fires on Wilma's
  actual 2005 jump (75kt→150kt in 18 hours, one of the fastest on record). The model
  smooths toward incremental change and under-predicts extreme outlier RI events —
  a genuine, well-documented hard problem in operational TC forecasting.

**Generated, not real satellite data — and why:**
The build brief calls for TCIR satellite imagery (IR1/WV/PMW/VIS channels) driving a
CNN, with real Grad-CAM explainability. That could not be built here: TCIR is
distributed via Google Drive / Baidu Pan / university mirrors, not a plain
downloadable file, and this was built with no network access at all. Rather than
leave `/image` and `/gradcam` returning `501` (which breaks the frozen contract — the
frontend is coded to expect a PNG at those URLs), `scripts/generate_visuals.py`
renders a schematic visualization per frame instead:
- `frame_{i}_image.png`: a top-down storm schematic (eye + spiral rain bands), with
  size/color/band-tightness driven by that frame's **real** wind speed and category
  from the track data. Not a satellite photo.
- `frame_{i}_gradcam.png`: a radial heatmap centered on the eye, with intensity scaled
  by that frame's **real** model confidence and predicted category. Not a saliency map
  from a trained CNN (there isn't one yet).

Both are real-data-driven, deterministic, and precomputed once — never generated live
per request — same as everything else the API serves. They're there so every contract
endpoint returns a real 200 with a correctly-sized PNG instead of a 404/501 that would
break the frontend's map/dashboard, while being honest that they're diagrams standing
in for photos, not photos.

**Scaffolded for later, not run:** `future_image_module/` contains a complete,
ready-to-run implementation of the actual brief — EfficientNet-B0 with a
weight-inflated 5-channel stem, the GRU trend head consuming frozen CNN embeddings,
and real Grad-CAM — for whoever has TCIR + a GPU. See `future_image_module/README.md`
for exactly how it plugs into `models/model.py` and the live API without changing the
contract.

## Live satellite feature (optional, needs internet)

Two additional endpoints, separate from the frozen 6-endpoint contract above, serve
**real, current** data fetched from NOAA at request time:

- `GET /api/live/storms` — the actual list of storms NOAA's National Hurricane Center
  currently considers active, pulled from NHC's public `CurrentStorms.json` feed. An
  empty list is normal outside hurricane season, not an error.
- `GET /api/live/satellite-image?satellite=east&region=full_disk` — the current GOES
  satellite image (JPEG), proxied from NOAA/NESDIS/STAR's public CDN. `satellite` is
  `east` or `west`; `region` defaults to `full_disk` (always valid) or can be a sector
  like `caribbean`, `gulf`, `southeast_us`, `northeast_us`, `pacific_southwest`,
  `pacific_northwest`.

**This could not be tested end-to-end from the sandbox this was built in — it had no
outbound network access at all.** The parsing/error-handling logic is unit-tested
against realistic fake responses (see the code comments in `live/goes_fetch.py` and
`live/nhc_fetch.py`), but the actual NOAA calls need to be verified on a machine with
internet before you rely on them for a demo. If a request 502s, the error message
will tell you whether NHC/NOAA was unreachable or their page format changed.

**Important: this is display-only, not a model input.** The trained model
(`models/model.py`) only ever consumes historical HURDAT2 numeric data — it has no
way to take an image as input. A live satellite image sitting next to a `/predict`
response is not that prediction "using" the image; keep them presented as separate
things in a demo, not implying they're connected. `future_image_module/` is the real
path to actually connecting live imagery to a prediction, and it still needs a
TCIR-trained CNN first.

NOAA occasionally renames which physical satellite is "GOES-East" vs. "GOES-West"
(this happened in 2025 and before) — if `/api/live/satellite-image` starts 502ing,
check `https://www.star.nesdis.noaa.gov/GOES/` and update `SATELLITES` in
`live/goes_fetch.py`.

## Frontend

`frontend/index.html` is a single-file interactive console: a rotatable 3D
globe with real storm tracks, the model's live intensity forecast, and
NOAA's live active-storm feed. See `frontend/README.md` for setup and for
exactly which parts of the UI are real data vs. stylized — same honesty
standard as the rest of this README.

## API contract

All 6 endpoints from the frozen contract are implemented and return real 200s against
real data:

- `GET /api/health`
- `GET /api/storms`
- `GET /api/storms/{storm_id}/track`
- `GET /api/storms/{storm_id}/frame/{index}/predict`
- `GET /api/storms/{storm_id}/frame/{index}/image`
- `GET /api/storms/{storm_id}/frame/{index}/gradcam`

Errors are always `{"error": true, "message": "..."}` with `404` for a missing
storm/frame and `500` for internal errors — never a raw stack trace or an HTML body.
CORS is enabled for all origins from the first line of `api/main.py`.

One honest gap in `predict`'s `baseline_comparison`: it's the model's **overall**
persistence/model wind MAE from `outputs/metrics.json`, not a per-frame comparison —
a per-frame breakdown wasn't computed. If that distinction matters for your demo,
say so rather than implying every frame has its own baseline number.

## Files

```
├── data/atlantic_storms.csv          # real HURDAT2 data
├── scripts/
│   ├── preprocess.py                  # builds real training sequences
│   ├── train.py                       # sanity check + full training
│   ├── precompute_demo_storms.py      # builds served demo storm track/prediction JSON
│   └── generate_visuals.py            # builds served schematic image/gradcam PNGs
├── models/model.py                    # TCModel (GRU multi-task, image-fusion-ready)
├── cache/                             # preprocessed train/val sequences (real)
├── outputs/
│   ├── best_model.pt                  # trained checkpoint (real)
│   ├── metrics.json                   # real reportable metrics
│   └── demo_storms/
│       ├── index.json                 # served by GET /api/storms
│       ├── {storm_id}_track.json      # served by GET /api/storms/{id}/track
│       ├── {storm_id}_predictions.json
│       └── images/{storm_id}/frame_{i}_{image,gradcam}.png
├── api/
│   ├── main.py                         # FastAPI server, all 6 contract endpoints, CORS
│   └── live_routes.py                  # optional /api/live/* endpoints (real, current data)
├── live/
│   ├── goes_fetch.py                   # fetches real current GOES imagery from NOAA
│   └── nhc_fetch.py                    # fetches real current active-storm list from NHC
├── future_image_module/               # real TCIR image-CNN scaffold, not wired in yet
│   ├── cyclone_cnn.py                 # EfficientNet-B0, weight-inflated 5-ch stem
│   ├── train_cnn.py
│   ├── trend_gru.py
│   ├── gradcam.py                     # real Grad-CAM (needs a trained CNN)
│   ├── config_reference.py
│   └── README.md                      # how to wire it in later
└── requirements.txt
```
