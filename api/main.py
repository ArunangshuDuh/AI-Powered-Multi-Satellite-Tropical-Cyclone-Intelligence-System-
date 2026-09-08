"""
FastAPI backend serving real, precomputed tropical cyclone track + prediction data,
plus data-driven schematic imagery. Matches the frozen API contract shared with the
frontend team (see README.md "API contract" section) exactly — field names, nesting,
and status codes.
"""
import json
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from api.live_routes import router as live_router

app = FastAPI(title="Tropical Cyclone AI System API")

# CORS enabled immediately, before any routes — required by the brief.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional live-data endpoints (/api/live/...) — additive, doesn't touch the
# frozen 6-endpoint contract below. See README "Live satellite feature".
app.include_router(live_router)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
DEMO_DIR = os.path.join(OUTPUTS_DIR, "demo_storms")
IMAGES_DIR = os.path.join(DEMO_DIR, "images")


def error_response(status_code, message):
    return JSONResponse(status_code=status_code, content={"error": True, "message": message})


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@app.get("/api/health")
def health():
    model_exists = os.path.exists(os.path.join(OUTPUTS_DIR, "best_model.pt"))
    return {"status": "ok", "model_loaded": model_exists}


@app.get("/api/storms")
def list_storms():
    index = load_json(os.path.join(DEMO_DIR, "index.json"))
    if index is None:
        return error_response(500, "demo storm index not found; run scripts/precompute_demo_storms.py first")
    return index


@app.get("/api/storms/{storm_id}/track")
def get_track(storm_id: str):
    track = load_json(os.path.join(DEMO_DIR, f"{storm_id}_track.json"))
    if track is None:
        return error_response(404, "storm not found")
    return track


@app.get("/api/storms/{storm_id}/frame/{index}/predict")
def get_prediction(storm_id: str, index: int):
    preds = load_json(os.path.join(DEMO_DIR, f"{storm_id}_predictions.json"))
    if preds is None:
        return error_response(404, "storm not found")
    pred = preds.get(str(index))
    if pred is None:
        return error_response(404, "frame not found or insufficient history for prediction")

    # Real overall model metrics (outputs/metrics.json) — not per-frame, since a
    # per-frame persistence/model comparison wasn't computed; stated as such in README.
    metrics = load_json(os.path.join(OUTPUTS_DIR, "metrics.json")) or {}
    response = dict(pred)
    response["gradcam_url"] = f"/api/storms/{storm_id}/frame/{index}/gradcam"
    response["baseline_comparison"] = {
        "persistence_wind_mae": metrics.get("persistence_wind_mae"),
        "model_wind_mae": metrics.get("model_wind_mae"),
    }
    return response


@app.get("/api/storms/{storm_id}/frame/{index}/image")
def get_image(storm_id: str, index: int):
    path = os.path.join(IMAGES_DIR, storm_id, f"frame_{index}_image.png")
    if not os.path.exists(path):
        return error_response(404, "frame not found")
    return FileResponse(path, media_type="image/png")


@app.get("/api/storms/{storm_id}/frame/{index}/gradcam")
def get_gradcam(storm_id: str, index: int):
    path = os.path.join(IMAGES_DIR, storm_id, f"frame_{index}_gradcam.png")
    if not os.path.exists(path):
        return error_response(404, "frame not found")
    return FileResponse(path, media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
