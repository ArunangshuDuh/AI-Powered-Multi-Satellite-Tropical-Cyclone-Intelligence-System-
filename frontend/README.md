# Frontend — Storm Track Console

A single-file, dependency-free (besides Three.js from a CDN) interactive console:
a rotatable 3D globe showing real historical storm tracks, the trained model's
frame-by-frame intensity forecast, and NOAA's live active-storm feed.

## Quick start

1. Start the backend (see the root `README.md`):
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```
2. Open `frontend/index.html` directly in a browser, or serve it so relative
   paths behave the same as a real deployment:
   ```bash
   cd frontend && python -m http.server 5500
   # then visit http://127.0.0.1:5500
   ```
3. In the top bar, confirm the API field points at your backend
   (`http://127.0.0.1:8000` by default) and press **CONNECT**.

No build step, no `npm install` — it's one HTML file with an inline
`<script>`, loading Three.js r128 from cdnjs.

## What each part of the UI actually shows

- **Globe + track line + colored waypoints** — real HURDAT2 positions from
  `/api/storms/{id}/track`, exactly as stored.
- **Moving white marker** — the storm's position, interpolated smoothly
  between real consecutive track points for a live-motion feel; it is not a
  new estimated position.
- **Model forecast panel (+6h/+12h/+24h wind & pressure, RI flag)** — the
  trained GRU model's real output from `/api/storms/{id}/frame/{i}/predict`.
  **The model forecasts intensity, not future track position** — there is no
  predicted lat/lon in the API, so the globe deliberately never draws a
  fabricated "future path." Don't add one without also adding a real
  track-forecasting head to the model.
- **Wind streak spiral around the active marker** — stylized, decorative.
  It rotates counter-clockwise in the northern hemisphere / clockwise in the
  southern hemisphere, which is real physics, but the streak positions
  themselves are not measured or modeled wind vectors — there is no wind
  field in this project's data. This mirrors the same real-vs-generated
  honesty the backend README already applies to the schematic frame images.
- **Live tab (red pulsing markers, storm list, satellite image)** — real,
  current data from `/api/live/storms` and `/api/live/satellite-image`,
  fetched from NOAA at request time. It is shown for situational awareness
  only; per the backend README, the trained model never consumes it.

## Notes

- CORS is already open (`allow_origins=["*"]`) on the backend, so pointing
  the API field at any reachable backend URL works, including over a LAN for
  a live demo.
- The Earth texture loads from a CDN at runtime; if there's no internet
  access, it falls back to a simple procedural texture automatically so the
  globe never breaks.
- Tested for the API's documented shapes only — if you change field names in
  the backend contract, update the `fillPrediction` / `onFrameChange`
  functions in `index.html` to match.
