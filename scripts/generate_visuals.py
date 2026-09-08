"""
Generates the PNG images served by /image and /gradcam.

IMPORTANT — read this before assuming these are satellite photos:
Real TCIR satellite imagery (IR1/WV/PMW/VIS channels) could not be obtained in the
environment this was built in (see README "What's real vs generated" section) — it's
distributed via Google Drive / Baidu Pan / university mirrors, not a plain downloadable
file, and this sandbox has no network access at all. Rather than leave /image and
/gradcam returning 501 (which breaks the frozen API contract the frontend is coded
against — it expects a PNG at those URLs), this script renders a schematic, data-driven
visualization instead:

  - frame_{i}_image.png: a top-down schematic of the storm system at that frame —
    eye + spiral rain bands — sized and colored using that frame's REAL wind speed,
    pressure, and category (from outputs/demo_storms/{storm_id}_track.json). This is
    not a satellite photo. It is a real-data-driven diagram standing in for one.
  - frame_{i}_gradcam.png: a radial heatmap centered on the storm eye, intensity
    scaled by that frame's REAL model confidence and predicted category (from
    outputs/demo_storms/{storm_id}_predictions.json) rather than an actual saliency
    map from a trained CNN (no image-CNN exists yet — see future_image_module/).

Both are cheap to generate, deterministic per frame, and produced once (not per
request) into outputs/demo_storms/images/, matching the brief's precomputation
requirement. When the real TCIR-trained image-CNN in future_image_module/ exists,
swap this script for real Grad-CAM output (future_image_module/README.md explains
exactly where to plug it in) — the API layer does not need to change either way.

Usage: python scripts/generate_visuals.py
"""
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

DEMO_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "demo_storms")
IMG_DIR = os.path.join(DEMO_DIR, "images")
SIZE = 256

# category_code -> base RGB color (cooler = weaker, hotter = stronger; matches
# conventional storm-category color scales used in TC dashboards)
CATEGORY_COLOR = {
    0: (120, 170, 230),  # TD   - blue
    1: (110, 210, 190),  # TS   - teal
    2: (240, 220, 90),   # Cat1 - yellow
    3: (240, 170, 60),   # Cat2 - orange
    4: (235, 110, 50),   # Cat3 - dark orange
    5: (220, 60, 60),    # Cat4 - red
    6: (170, 40, 130),   # Cat5 - magenta/purple
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def render_storm_image(wind_kt, category_code, seed):
    """Schematic top-down storm system, sized/colored from real wind + category."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (SIZE, SIZE), (8, 10, 22))
    draw = ImageDraw.Draw(img)
    cx, cy = SIZE // 2, SIZE // 2
    color = CATEGORY_COLOR.get(category_code, (150, 150, 150))

    # Overall system radius scales with wind speed (real value), clamped to canvas.
    max_radius = min(SIZE * 0.46, 40 + wind_kt * 0.9)

    # Spiral rain bands: several logarithmic-spiral arcs of small dots, real wind
    # speed controls how tightly wound + how many bands (stronger storm = tighter).
    n_bands = 3 + category_code
    tightness = 0.15 + category_code * 0.03
    for band in range(n_bands):
        band_offset = (2 * math.pi / n_bands) * band
        n_points = 140
        band_color = tuple(min(255, int(c * (0.55 + 0.08 * band))) for c in color)
        for p in range(n_points):
            t = p / n_points
            radius = 14 + t * max_radius
            angle = band_offset + t * (8 + category_code) + tightness * radius * 0.05
            jitter = rng.normal(0, 1.5)
            x = cx + (radius + jitter) * math.cos(angle)
            y = cy + (radius + jitter) * math.sin(angle)
            r = max(1, 2.2 - t * 1.3)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=band_color)

    img = img.filter(ImageFilter.GaussianBlur(1.1))
    draw = ImageDraw.Draw(img)

    # Eye: only a clear/small eye for storms at hurricane strength or above (Cat1+),
    # matching real storm structure — weaker storms don't have a defined eye.
    eye_r = 6 + category_code * 1.8 if category_code >= 2 else 3
    draw.ellipse([cx - eye_r, cy - eye_r, cx + eye_r, cy + eye_r], fill=(230, 230, 235))
    draw.ellipse([cx - eye_r, cy - eye_r, cx + eye_r, cy + eye_r], outline=(90, 90, 100), width=1)

    return img


def render_gradcam(confidence, category_code, seed):
    """Radial heatmap centered on the eye, intensity scaled by real confidence."""
    rng = np.random.default_rng(seed + 1)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    cx, cy = SIZE // 2, SIZE // 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    spread = 30 + category_code * 6
    heat = np.exp(-(dist ** 2) / (2 * spread ** 2))
    heat *= 0.35 + 0.65 * confidence  # real per-frame model confidence scales peak intensity
    heat += 0.04 * rng.normal(size=heat.shape).clip(-1, 1)
    heat = np.clip(heat, 0, 1)

    # Jet-like colormap without matplotlib: blue -> cyan -> yellow -> red
    r = np.clip(1.5 - np.abs(4 * heat - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * heat - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * heat - 1), 0, 1)
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    alpha = (heat * 255).astype(np.uint8)

    heatmap_img = Image.fromarray(rgb, mode="RGB").convert("RGBA")
    heatmap_img.putalpha(Image.fromarray(alpha))

    base = Image.new("RGBA", (SIZE, SIZE), (15, 15, 20, 255))
    composite = Image.alpha_composite(base, heatmap_img)
    return composite.convert("RGB")


def main():
    if not os.path.isdir(DEMO_DIR):
        raise SystemExit("outputs/demo_storms/ not found — run scripts/precompute_demo_storms.py first")

    index = load_json(os.path.join(DEMO_DIR, "index.json"))
    total = 0
    for storm in index["storms"]:
        storm_id = storm["storm_id"]
        track = load_json(os.path.join(DEMO_DIR, f"{storm_id}_track.json"))
        preds_path = os.path.join(DEMO_DIR, f"{storm_id}_predictions.json")
        preds = load_json(preds_path) if os.path.exists(preds_path) else {}

        storm_dir = os.path.join(IMG_DIR, storm_id)
        os.makedirs(storm_dir, exist_ok=True)

        for i, frame in enumerate(track["frames"]):
            img = render_storm_image(frame["wind_kt"], frame["category_code"], seed=i + 1000)
            img.save(os.path.join(storm_dir, f"frame_{i}_image.png"))

            pred = preds.get(str(i))
            confidence = pred["confidence"] if pred else 0.5
            cat_code = pred["predicted_category_code"] if pred else frame["category_code"]
            cam = render_gradcam(confidence, cat_code, seed=i + 1000)
            cam.save(os.path.join(storm_dir, f"frame_{i}_gradcam.png"))
            total += 2

        print(f"  {storm_id}: {len(track['frames'])} frames -> {storm_dir}")

    print(f"\n{total} PNGs written to {IMG_DIR}")


if __name__ == "__main__":
    main()
